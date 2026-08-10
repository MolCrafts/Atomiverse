/// @file tests/cuda/gpw/test_gpw_full_product.cu
/// @brief Unit: public `cuda::gpw::GpwDft` shape + fail-closed guards + H2 dual-gate.
///
/// First product slice: Γ RKS DIIS PADE energy (and optional forces) vs
/// same-settings CPU `GpwDft`. Device path required for energy (GH200).
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
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

void set_h2_cfg_cpu(ATV::cpu::gpw::GpwDft& m) {
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

void set_h2_cfg_cuda(ATV::cuda::gpw::GpwDft& m) {
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

TEST(CudaGpwFullProduct, PublicShapeMembersAndLastResult) {
    ATV::cuda::gpw::GpwDft m;
    m.n_occ = 1;
    m.cutoff_ry = 100.0;
    EXPECT_FALSE(m.is_uks());
    m.n_alpha = 1;
    m.n_beta = 1;
    EXPECT_TRUE(m.is_uks());
    const auto& pre = m.last_result();
    EXPECT_FALSE(pre.converged);
}

TEST(CudaGpwFullProduct, FailClosedUnsupported) {
    auto atoms = h2_atoms();
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.use_ot = true;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.nonperiodic = true;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // UKS+PBE still fail-closed (uks-pbe-gate deeper slug).
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.n_alpha = 2;
        m.n_beta = 0;
        m.n_occ = 0;
        m.xc = XcKind::PBE;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // UKS PADE is accepted (cuda-gpw-uks-gamma).
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.n_alpha = 2;
        m.n_beta = 0;
        m.n_occ = 0;
        m.xc = XcKind::PADE;
        EXPECT_NO_THROW(m.build(atoms));
    }
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.ngrids = 4;
        m.rel_cutoff = 40.0;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.xc = XcKind::BLYP;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        EXPECT_NO_THROW(m.build(atoms));
    }
}

// Energy dual-gate vs same-settings CPU (device path; needs GPU).
TEST(CudaGpwFullProduct, H2GammaPadeEnergyDualGate) {
    auto atoms = h2_atoms();

    ATV::cpu::gpw::GpwDft cpu;
    set_h2_cfg_cpu(cpu);
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_h2_cfg_cuda(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA SCF failed iters=" << rg.iters;
    ASSERT_TRUE(std::isfinite(rg.e_total));

    // Composite of stage foothold tols + host STVnl/DIIS twin (not bit-exact).
    constexpr double kETolHa = 5.0e-6;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
}

TEST(CudaGpwFullProduct, H2GammaPadeForcesDualGate) {
    auto atoms = h2_atoms();

    ATV::cpu::gpw::GpwDft cpu;
    set_h2_cfg_cpu(cpu);
    cpu.build(atoms);
    const auto Fc = ATV::cpu::gpw::gpw_forces_analytic(atoms, cpu);
    ASSERT_EQ(Fc.size(), 2u);

    ATV::cuda::gpw::GpwDft gpu;
    set_h2_cfg_cuda(gpu);
    gpu.build(atoms);
    const auto Fg = ATV::cuda::gpw::gpw_forces_analytic(atoms, gpu);
    ASSERT_EQ(Fg.size(), 2u);

    constexpr double kFTol = 5.0e-5; // Ha/Bohr
    double maxd = 0.0;
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            maxd = std::max(maxd, std::abs(Fg[a][k] - Fc[a][k]));
    EXPECT_LT(maxd, kFTol) << "max|dF|=" << maxd;
}

// High-spin H₂ Γ UKS PADE energy dual-gate (O₂-class collinear UKS path;
// small cell correctness only — needs GPU).
TEST(CudaGpwFullProduct, H2HighSpinUksPadeEnergyDualGate) {
    auto atoms = h2_atoms();

    auto set_uks = [](auto& m) {
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 80.0;
        m.n_occ = 0;
        m.n_alpha = 2;
        m.n_beta = 0;
        m.max_iter = 60;
        m.tol = 1e-8;
        m.diis_depth = 6;
        m.xc = XcKind::PADE;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.use_ot = false;
        m.ngrids = 1;
        m.eps_default = 1e-8;
    };

    ATV::cpu::gpw::GpwDft cpu;
    set_uks(cpu);
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU UKS SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_uks(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA UKS SCF failed iters=" << rg.iters;
    ASSERT_TRUE(std::isfinite(rg.e_total));
    EXPECT_FALSE(rg.P_beta.size() == 0 && gpu.n_beta > 0);
    // High-spin: Sz=1 → s2 ≈ 2 for pure triplet (Löwdin).
    EXPECT_GT(rg.s2, 1.5);

    constexpr double kETolHa = 5.0e-6;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
}
