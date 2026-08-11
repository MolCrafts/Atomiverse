/// @file tests/cuda/gpw/test_gpw_kpoints.cu
/// @brief Unit: multi-k RKS DIIS product path on `cuda::gpw::GpwDft`.
///
/// - Fail-closed joints (UKS×k, MG×k)
/// - Accept multi-k RKS mesh
/// - H₂ k_mesh={2,1,1} PADE energy dual-gate vs same-settings CPU (device)
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

namespace {

std::vector<ATV::cpu::gpw::GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {ATV::cpu::gpw::GpwAtom(rA, basis, pot, 1),
            ATV::cpu::gpw::GpwAtom(rB, basis, pot, 1)};
}

template <typename Method>
void set_h2_cfg(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.max_iter = 40;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.xc = XcKind::PADE;
    m.nonperiodic = false;
    m.local_pp_on_grid = true;
    m.use_ot = false;
    m.ngrids = 1;
    m.eps_default = 1e-8;
}

} // namespace

TEST(CudaGpwKpoints, FailClosedJoints) {
    auto atoms = h2_atoms();
    {
        // Multi-k + UKS → deeper slug
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg(m);
        m.n_occ = 0;
        m.n_alpha = 2;
        m.n_beta = 0;
        m.k_mesh[0] = 2;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // Multi-k + multigrid → deeper slug
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg(m);
        m.k_mesh[2] = 2;
        m.ngrids = 4;
        m.rel_cutoff = 40.0;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // Explicit k list length mismatch
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg(m);
        m.k_points.push_back({0.0, 0.0, 0.25});
        // weights empty → size mismatch
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
}

TEST(CudaGpwKpoints, AcceptMultiKRks) {
    auto atoms = h2_atoms();
    ATV::cuda::gpw::GpwDft m;
    set_h2_cfg(m);
    m.k_mesh[0] = 2;
    EXPECT_TRUE(m.uses_kpath());
    EXPECT_NO_THROW(m.build(atoms));
}

TEST(CudaGpwKpoints, H2MeshEnergyDualGate) {
    auto atoms = h2_atoms();
    constexpr double kETolHa = 5.0e-6;

    // TR off so MP 2×1×1 keeps both ±1/4 (nk=2) — not a TR-folded single k.
    ATV::cpu::gpw::GpwDft cpu;
    set_h2_cfg(cpu);
    cpu.k_mesh[0] = 2;
    cpu.k_mesh[1] = 1;
    cpu.k_mesh[2] = 1;
    cpu.k_time_reversal = false;
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU multi-k SCF not converged";
    ASSERT_GE(rc.n_kpts, 2) << "expected multi-k mesh without TR fold";

    ATV::cuda::gpw::GpwDft gpu;
    set_h2_cfg(gpu);
    gpu.k_mesh[0] = 2;
    gpu.k_mesh[1] = 1;
    gpu.k_mesh[2] = 1;
    gpu.k_time_reversal = false;
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA multi-k SCF not converged";
    ASSERT_EQ(rg.n_kpts, rc.n_kpts);
    ASSERT_GE(rg.n_kpts, 2);

    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LE(de, kETolHa) << "CPU E=" << rc.e_total << " CUDA E=" << rg.e_total
                           << " |dE|=" << de;
}
