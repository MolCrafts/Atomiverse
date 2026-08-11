/// @file tests/cuda/gpw/test_kpoint_integrate.cu
/// @brief Unit: device multi-k V→H_k integrate vs host integrate_potential_k.
///
/// Single-function path only — not a full multi-k SCF solve. Exercises
/// `KpointIntegrateDevice` against the CPU twin on a small H₂-class χ^k
/// + synthetic V_eff grid.
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
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
using ATV::cpu::gpw::internal::KPoint;
using ATV::cpu::gpw::internal::KpointBasisGrid;
using ATV::cpu::gpw::internal::build_kpoint_basis_grid;
using ATV::cpu::gpw::internal::integrate_potential_k;
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

} // namespace

TEST(CudaGpwKpointIntegrate, MatchesHostIntegratePotentialK) {
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

    // Two off-Γ k-points (matches multi-k product mesh without TR fold).
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
    ASSERT_EQ(static_cast<int>(kbg.chi.size()), nk);

    // Smooth synthetic V_eff (not a physical KS potential) — nonzero, real.
    std::vector<double> V(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g) {
        const double t = static_cast<double>(g) / static_cast<double>(npts);
        V[static_cast<std::size_t>(g)] = 0.1 + 0.05 * std::sin(6.283185307179586 * t);
    }

    double* d_V = nullptr;
    check_cuda(cudaMalloc(&d_V, static_cast<size_t>(npts) * sizeof(double)),
               "malloc d_V");
    check_cuda(cudaMemcpy(d_V, V.data(),
                          static_cast<size_t>(npts) * sizeof(double),
                          cudaMemcpyHostToDevice),
               "H2D V");

    KpointIntegrateDevice kint;
    kint.build(/*stream=*/nullptr, nb, npts, nk);
    for (int ik = 0; ik < nk; ++ik)
        kint.upload_chi(ik, kbg.chi[static_cast<std::size_t>(ik)].data());
    check_cuda(cudaDeviceSynchronize(), "sync chi");

    // Host twin tol: tree reduction vs host chunked fold — not bit-identical.
    constexpr double kAbsTol = 5.0e-9;
    constexpr double kRelTol = 1.0e-8;

    for (int ik = 0; ik < nk; ++ik) {
        const CplxMatrix Vhost = integrate_potential_k(kbg, ik, V);
        kint.integrate(d_V, kbg.weight, ik);
        CplxMatrix Vdev = xt::zeros<Cplx>({nb, nb});
        kint.download_Vmat(Vdev.data());

        double max_abs = 0.0;
        double max_rel = 0.0;
        for (int u = 0; u < nb; ++u) {
            for (int t = 0; t < nb; ++t) {
                const Cplx dh = Vhost(u, t);
                const Cplx dd = Vdev(u, t);
                const double a = std::abs(dd - dh);
                max_abs = std::max(max_abs, a);
                const double den = std::max(std::abs(dh), 1.0e-14);
                max_rel = std::max(max_rel, a / den);
            }
        }
        EXPECT_LE(max_abs, kAbsTol)
            << "ik=" << ik << " max|ΔV|=" << max_abs;
        EXPECT_LE(max_rel, kRelTol)
            << "ik=" << ik << " max relative=" << max_rel;
        // Hermitian residual (device should preserve Hermiticity to ~tol).
        for (int u = 0; u < nb; ++u) {
            for (int t = 0; t <= u; ++t) {
                const Cplx a = Vdev(u, t);
                const Cplx b = std::conj(Vdev(t, u));
                EXPECT_NEAR(a.real(), b.real(), 1e-12);
                EXPECT_NEAR(a.imag(), b.imag(), 1e-12);
            }
        }
    }

    cudaFree(d_V);
}

TEST(CudaGpwKpointIntegrate, EmptyReject) {
    KpointIntegrateDevice kint;
    EXPECT_THROW(kint.build(nullptr, 0, 10, 1), std::invalid_argument);
    EXPECT_THROW(kint.build(nullptr, 2, 0, 1), std::invalid_argument);
    EXPECT_THROW(kint.build(nullptr, 2, 10, 0), std::invalid_argument);
}
