/// @file tests/cuda/gpw/test_kpoint_collocate.cu
/// @brief Unit: device multi-k density collocate vs host collocate_density_k.
///
/// Fragment-level only (not full SCF): build χ^k on a small H₂ grid, fixed
/// P_k, compare device ρ to host oracle pointwise. Energy dual-gate lives
/// in regressions/gpw/cuda-gpw-kpoint-device-collocate.
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cuda/gpw/internal/kpoint_collocate.cuh"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::make_multigrid;
using ATV::cpu::gpw::internal::CplxMatrix;
using ATV::cpu::gpw::internal::KPoint;
using ATV::cpu::gpw::internal::KpointBasisGrid;
using ATV::cpu::gpw::internal::build_kpoint_basis_grid;
using ATV::cpu::gpw::internal::collocate_density_k;
using ATV::cuda::gpw::internal::KpointCollocateDevice;
using Cplx = std::complex<double>;

namespace {

std::vector<GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {GpwAtom(rA, basis, pot, 1), GpwAtom(rB, basis, pot, 1)};
}

std::vector<GridBasisFunction> flatten(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> out;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            out.push_back(bf);
    return out;
}

CplxMatrix herm_identity_scaled(int nb, double scale) {
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    for (int i = 0; i < nb; ++i)
        P(i, i) = Cplx(scale, 0.0);
    // off-diagonal Hermitian fill so the path is not pure diagonal.
    if (nb >= 2) {
        P(0, 1) = Cplx(0.05, 0.02);
        P(1, 0) = std::conj(P(0, 1));
    }
    return P;
}

} // namespace

TEST(CudaGpwKpointCollocate, DeviceMatchesHostDensity) {
    auto atoms = h2_atoms();
    const auto basis = flatten(atoms);
    const int nb = static_cast<int>(basis.size());
    ASSERT_GE(nb, 2);

    double cell[3] = {12.0, 12.0, 12.0};
    const auto mg = make_multigrid(cell, /*cutoff_ry=*/40.0, /*nlev=*/1,
                                   /*rel_cutoff=*/40.0);
    ASSERT_FALSE(mg.levels.empty());
    const auto& grid = mg.levels[0];
    const long npts = grid.npts();
    ASSERT_GT(npts, 0);

    std::vector<KPoint> kpts(2);
    kpts[0].frac[0] = 0.25;
    kpts[0].weight = 0.5;
    kpts[1].frac[0] = -0.25;
    kpts[1].weight = 0.5;

    const KpointBasisGrid kbg =
        build_kpoint_basis_grid(grid, basis, kpts, /*eps=*/1e-8);
    ASSERT_EQ(kbg.nb, nb);
    ASSERT_EQ(kbg.npts, npts);
    ASSERT_EQ(static_cast<int>(kbg.chi.size()), 2);

    std::vector<CplxMatrix> P_k(2);
    P_k[0] = herm_identity_scaled(nb, 1.0);
    P_k[1] = herm_identity_scaled(nb, 0.8);

    // Host oracle
    std::vector<double> rho_host(static_cast<std::size_t>(npts), 0.0);
    for (int ik = 0; ik < 2; ++ik)
        collocate_density_k(kbg, ik, P_k[static_cast<std::size_t>(ik)],
                            kpts[static_cast<std::size_t>(ik)].weight,
                            rho_host);

    // Device
    double* d_rho = nullptr;
    ASSERT_EQ(cudaMalloc(&d_rho, static_cast<size_t>(npts) * sizeof(double)),
              cudaSuccess);
    KpointCollocateDevice coll;
    coll.build(/*stream=*/nullptr, /*nk=*/2, nb, npts);
    for (int ik = 0; ik < 2; ++ik)
        coll.upload_chi(ik, kbg.chi[static_cast<std::size_t>(ik)].data());
    coll.zero_rho(d_rho);
    for (int ik = 0; ik < 2; ++ik)
        coll.collocate_k(ik, P_k[static_cast<std::size_t>(ik)].data(),
                         kpts[static_cast<std::size_t>(ik)].weight, d_rho);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> rho_dev(static_cast<std::size_t>(npts));
    ASSERT_EQ(cudaMemcpy(rho_dev.data(), d_rho,
                         static_cast<size_t>(npts) * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_rho);

    double max_abs = 0.0;
    double peak = 0.0;
    for (long g = 0; g < npts; ++g) {
        const double h = rho_host[static_cast<std::size_t>(g)];
        const double d = rho_dev[static_cast<std::size_t>(g)];
        peak = std::max(peak, std::abs(h));
        max_abs = std::max(max_abs, std::abs(d - h));
    }
    // Device twin of host continuous matvec — expect near bit-stable.
    constexpr double kAbsTol = 1e-12;
    constexpr double kRelPeak = 1e-10;
    EXPECT_LE(max_abs, std::max(kAbsTol, kRelPeak * peak))
        << "max|Δρ|=" << max_abs << " peak=" << peak;
}
