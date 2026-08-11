/// @file tests/cuda/gpw/test_kpoint_h_residency.cu
/// @brief Unit: device assemble_H → ingest_H_k_device_rm → residual_from_device.
///
/// Single-function path only — not a full multi-k SCF solve. Chains
/// `KpointIntegrateDevice` H fuse with `KpointDiisEigenDevice` device residual
/// and checks parity vs host residual(H) after download_H (oracle for the
/// layout transpose). cuda-gpw-kpoint-h-residency.
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cuda/gpw/internal/kpoint_diis_eigen.cuh"
#include "atomiverse/cuda/gpw/internal/kpoint_integrate.cuh"

#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::MultiGrid;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::make_multigrid;
using ATV::cpu::gpw::internal::CplxMatrix;
using ATV::cpu::gpw::internal::KOrth;
using ATV::cpu::gpw::internal::KPoint;
using ATV::cpu::gpw::internal::KpointBasisGrid;
using ATV::cpu::gpw::internal::build_kpoint_basis_grid;
using ATV::cpu::gpw::internal::diagonalize_k;
using ATV::cuda::gpw::internal::KpointDiisEigenDevice;
using ATV::cuda::gpw::internal::KpointIntegrateDevice;

namespace {

using Cplx = std::complex<double>;

void check_cuda(cudaError_t e, const char* what) {
    ASSERT_EQ(e, cudaSuccess) << what << ": " << cudaGetErrorString(e);
}

std::vector<ATV::cpu::gpw::GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {ATV::cpu::gpw::GpwAtom(rA, basis, pot, 1),
            ATV::cpu::gpw::GpwAtom(rB, basis, pot, 1)};
}

std::vector<GridBasisFunction>
flatten_basis_local(const std::vector<ATV::cpu::gpw::GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

// density_from_fixed_k is private to gpw_solver; rebuild closed-shell P here.
CplxMatrix density_closed(const CplxMatrix& C, int n_occ, int nb) {
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    const int m = static_cast<int>(C.shape(1));
    const int nfill = std::min(m, n_occ);
    for (int o = 0; o < nfill; ++o) {
        for (int i = 0; i < nb; ++i) {
            const Cplx ci = 2.0 * C(i, o);
            for (int j = 0; j < nb; ++j)
                P(i, j) += ci * std::conj(C(j, o));
        }
    }
    return P;
}

} // namespace

TEST(CudaGpwKpointHResidency, IngestResidualMatchesHostDownload) {
    auto atoms = h2_atoms();
    const auto basis = flatten_basis_local(atoms);
    const int nb = static_cast<int>(basis.size());
    ASSERT_GE(nb, 2);

    const double cell[3] = {12.0, 12.0, 12.0};
    const MultiGrid mg = make_multigrid(cell, /*cutoff_ry=*/80.0, /*nlev=*/1,
                                        /*rel_cutoff=*/80.0);
    ASSERT_FALSE(mg.levels.empty());
    const UniformGrid& grid = mg.levels[0];
    const long npts = grid.npts();
    ASSERT_GT(npts, 0);

    std::vector<KPoint> kpts(2);
    kpts[0].frac[0] = 0.25;
    kpts[0].frac[1] = 0.0;
    kpts[0].frac[2] = 0.0;
    kpts[0].weight = 0.5;
    kpts[1].frac[0] = -0.25;
    kpts[1].frac[1] = 0.0;
    kpts[1].frac[2] = 0.0;
    kpts[1].weight = 0.5;
    const int nk = 2;

    const KpointBasisGrid kbg =
        build_kpoint_basis_grid(grid, basis, kpts, /*eps=*/1e-8);
    ASSERT_EQ(kbg.nb, nb);
    ASSERT_EQ(kbg.npts, npts);

    // Synthetic V_eff and H0 (geometry-fixed).
    std::vector<double> V(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g) {
        const double t = static_cast<double>(g) / static_cast<double>(npts);
        V[static_cast<std::size_t>(g)] =
            0.1 + 0.05 * std::sin(6.283185307179586 * t);
    }

    std::vector<CplxMatrix> H0(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> S(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> P(static_cast<std::size_t>(nk));
    std::vector<KOrth> orth;
    orth.reserve(static_cast<std::size_t>(nk));
    int max_m = 0;
    for (int ik = 0; ik < nk; ++ik) {
        H0[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        S[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        for (int i = 0; i < nb; ++i) {
            H0[static_cast<std::size_t>(ik)](i, i) =
                -0.5 - 0.02 * static_cast<double>(i)
                - 0.01 * static_cast<double>(ik);
            S[static_cast<std::size_t>(ik)](i, i) = 1.0;
            if (i + 1 < nb) {
                const Cplx off(0.05 + 0.01 * ik, 0.02 * (1 - 2 * ik));
                H0[static_cast<std::size_t>(ik)](i, i + 1) = off;
                H0[static_cast<std::size_t>(ik)](i + 1, i) = std::conj(off);
                S[static_cast<std::size_t>(ik)](i, i + 1) = Cplx(0.08, 0.01);
                S[static_cast<std::size_t>(ik)](i + 1, i) =
                    std::conj(S[static_cast<std::size_t>(ik)](i, i + 1));
            }
        }
        orth.emplace_back(S[static_cast<std::size_t>(ik)], 1e-10);
        max_m = std::max(max_m,
                         static_cast<int>(orth.back().X.shape(1)));
        auto [eps0, C0] =
            diagonalize_k(H0[static_cast<std::size_t>(ik)], orth.back());
        (void)eps0;
        P[static_cast<std::size_t>(ik)] = density_closed(C0, /*n_occ=*/1, nb);
    }

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "streamCreate");
    double* d_V = nullptr;
    check_cuda(cudaMalloc(&d_V, static_cast<size_t>(npts) * sizeof(double)),
               "malloc d_V");
    check_cuda(cudaMemcpyAsync(d_V, V.data(),
                               static_cast<size_t>(npts) * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "H2D V");

    KpointIntegrateDevice kint;
    kint.build(stream, nb, npts, nk);
    for (int ik = 0; ik < nk; ++ik) {
        kint.upload_chi(ik, kbg.chi[static_cast<std::size_t>(ik)].data());
        kint.upload_H0(ik, H0[static_cast<std::size_t>(ik)].data());
    }

    std::vector<double> weights = {0.5, 0.5};
    KpointDiisEigenDevice kdiis_dev;
    kdiis_dev.build(stream, nk, nb, max_m, /*max_vectors=*/4, /*n_occ=*/1,
                    /*occ_factor=*/2.0, weights);
    KpointDiisEigenDevice kdiis_host;
    kdiis_host.build(stream, nk, nb, max_m, /*max_vectors=*/4, /*n_occ=*/1,
                     /*occ_factor=*/2.0, weights);
    for (int ik = 0; ik < nk; ++ik) {
        const int mk =
            static_cast<int>(orth[static_cast<std::size_t>(ik)].X.shape(1));
        kdiis_dev.upload_SX_k(ik, S[static_cast<std::size_t>(ik)].data(),
                              orth[static_cast<std::size_t>(ik)].X.data(), mk);
        kdiis_dev.upload_P_k(ik, P[static_cast<std::size_t>(ik)].data());
        kdiis_host.upload_SX_k(ik, S[static_cast<std::size_t>(ik)].data(),
                               orth[static_cast<std::size_t>(ik)].X.data(), mk);
        kdiis_host.upload_P_k(ik, P[static_cast<std::size_t>(ik)].data());
    }

    std::vector<CplxMatrix> H_host(static_cast<std::size_t>(nk));
    std::vector<const Cplx*> H_ptrs(static_cast<std::size_t>(nk), nullptr);
    for (int ik = 0; ik < nk; ++ik) {
        kint.integrate(d_V, kbg.weight, ik);
        kint.assemble_H(ik);
        // Device-resident path (product mid-SCF).
        kdiis_dev.ingest_H_k_device_rm(ik, kint.d_H());
        // Host path oracle: download then residual(H_hosts).
        H_host[static_cast<std::size_t>(ik)] = xt::zeros<Cplx>({nb, nb});
        kint.download_H(H_host[static_cast<std::size_t>(ik)].data());
        H_ptrs[static_cast<std::size_t>(ik)] =
            H_host[static_cast<std::size_t>(ik)].data();
    }
    check_cuda(cudaStreamSynchronize(stream), "sync assemble/ingest");

    const double err_dev = kdiis_dev.residual_from_device();
    const double err_host = kdiis_host.residual(H_ptrs);
    EXPECT_NEAR(err_dev, err_host, 1e-10)
        << "device residual vs host residual after download_H |d|="
        << std::abs(err_dev - err_host);
    EXPECT_GT(err_host, 0.0); // sanity: synthetic H not already converged

    cudaFree(d_V);
    cudaStreamDestroy(stream);
}
