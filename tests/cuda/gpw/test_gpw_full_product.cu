/// @file tests/cuda/gpw/test_gpw_full_product.cu
/// @brief Unit: public `cuda::gpw::GpwDft` shape + fail-closed guards + H2 dual-gate.
///
/// Product slice: Γ RKS DIIS PADE/PBE energy (and optional forces), multigrid
/// RKS, and collinear UKS PADE vs same-settings CPU `GpwDft`. Device path
/// required for energy dual-gates (GH200).
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

// PBE product twin: same geometry + compact basis; GTH-PBE-q1 PP.
std::vector<ATV::cpu::gpw::GpwAtom> h2_atoms_pbe() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PBE-q1");
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

template <typename Method>
void set_h2_cfg_pbe(Method& m) {
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 80.0;
    m.n_occ = 1;
    m.max_iter = 40;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.xc = XcKind::PBE;
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
        // Multigrid without REL_CUTOFF is fail-closed (would silently collapse).
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.ngrids = 4;
        m.rel_cutoff = 0.0;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // Multigrid with REL_CUTOFF is accepted on the product energy path.
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.ngrids = 4;
        m.rel_cutoff = 40.0;
        EXPECT_NO_THROW(m.build(atoms));
    }
    {
        // UKS + multigrid not on product path (fail closed).
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.n_alpha = 2;
        m.n_beta = 0;
        m.n_occ = 0;
        m.ngrids = 4;
        m.rel_cutoff = 40.0;
        m.xc = XcKind::PADE;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.xc = XcKind::BLYP;
        EXPECT_THROW(m.build(atoms), std::invalid_argument);
    }
    {
        // Closed-shell RKS PBE is accepted (cuda-gpw-pbe-product-gate).
        ATV::cuda::gpw::GpwDft m;
        set_h2_cfg_cuda(m);
        m.xc = XcKind::PBE;
        EXPECT_NO_THROW(m.build(atoms));
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

// Multigrid product energy: ngrids=4 + rel_cutoff=40 vs same-settings CPU.
TEST(CudaGpwFullProduct, H2GammaPadeMultigridEnergyDualGate) {
    auto atoms = h2_atoms();

    auto set_mg = [](auto& m) {
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 80.0;
        m.ngrids = 4;
        m.rel_cutoff = 40.0;
        m.n_occ = 1;
        m.max_iter = 40;
        m.tol = 1e-8;
        m.diis_depth = 6;
        m.xc = XcKind::PADE;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.use_ot = false;
        m.eps_default = 1e-8;
    };

    ATV::cpu::gpw::GpwDft cpu;
    set_mg(cpu);
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU MG SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_mg(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA MG SCF failed iters=" << rg.iters;
    ASSERT_TRUE(std::isfinite(rg.e_total));

    // Composite of stage MG foothold tols + host STVnl/DIIS (not bit-exact).
    constexpr double kETolHa = 5.0e-6;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
}

// Closed-shell H₂ Γ RKS PBE energy dual-gate (product PBE path gated e2e;
// reuses KsChainDevice PBE stage — needs GPU).
TEST(CudaGpwFullProduct, H2GammaPbeEnergyDualGate) {
    auto atoms = h2_atoms_pbe();

    ATV::cpu::gpw::GpwDft cpu;
    set_h2_cfg_pbe(cpu);
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU PBE SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_h2_cfg_pbe(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA PBE SCF failed iters=" << rg.iters;
    ASSERT_TRUE(std::isfinite(rg.e_total));

    constexpr double kETolHa = 5.0e-6;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
}

TEST(CudaGpwFullProduct, H2GammaPbeForcesDualGate) {
    auto atoms = h2_atoms_pbe();

    ATV::cpu::gpw::GpwDft cpu;
    set_h2_cfg_pbe(cpu);
    cpu.build(atoms);
    const auto Fc = ATV::cpu::gpw::gpw_forces_analytic(atoms, cpu);
    ASSERT_EQ(Fc.size(), 2u);

    ATV::cuda::gpw::GpwDft gpu;
    set_h2_cfg_pbe(gpu);
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

// Water Γ RKS PADE energy dual-gate (multi-species production class after H₂
// foothold; cuda-gpw-product-h2o-gate — needs GPU).
TEST(CudaGpwFullProduct, WaterGammaPadeEnergyDualGate) {
    const auto bas_O = GaussianBasis(molrs::Element::O, "SZV-GTH-PADE");
    const auto pot_O = GthPotential(molrs::Element::O, "GTH-PADE-q6");
    const auto bas_H = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot_H = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    constexpr double kA0 = 0.52917721067;
    constexpr double kCell = 12.0;
    const double dHx = 0.7586 / kA0;
    const double dHz = 0.5043 / kA0;
    const double c = 0.5 * kCell;
    const double rO[3] = {c, c, c};
    const double rH1[3] = {c + dHx, c, c + dHz};
    const double rH2[3] = {c - dHx, c, c + dHz};
    std::vector<ATV::cpu::gpw::GpwAtom> atoms = {
        ATV::cpu::gpw::GpwAtom(rO, bas_O, pot_O, 8),
        ATV::cpu::gpw::GpwAtom(rH1, bas_H, pot_H, 1),
        ATV::cpu::gpw::GpwAtom(rH2, bas_H, pot_H, 1)};

    auto set_h2o = [](auto& m) {
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 4;
        m.max_iter = 80;
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
    set_h2o(cpu);
    cpu.build(atoms);
    const auto& rc = cpu.run(atoms);
    ASSERT_TRUE(rc.converged) << "CPU H2O SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_h2o(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA H2O SCF failed iters=" << rg.iters
                              << " E=" << rg.e_total;
    ASSERT_TRUE(std::isfinite(rg.e_total));

    // Multi-center p-shell class (matches O₂ production energy tol).
    constexpr double kETolHa = 5.0e-5;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
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

// Production O₂ triplet Γ UKS PADE energy dual-gate (p-shell AO depth after
// H₂ pure-s foothold). Needs GPU. Periodic cell (CUDA rejects MT).
TEST(CudaGpwFullProduct, O2UksPadeEnergyDualGate) {
    const auto basis = GaussianBasis(molrs::Element::O, "SZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::O, "GTH-PADE-q6");
    constexpr double kBondBohr = 1.2075 / 0.52917721067;
    constexpr double kCell = 12.0;
    const double half = 0.5 * kBondBohr;
    const double c = 0.5 * kCell;
    const double rA[3] = {c, c, c - half};
    const double rB[3] = {c, c, c + half};
    std::vector<ATV::cpu::gpw::GpwAtom> atoms{
        ATV::cpu::gpw::GpwAtom(rA, basis, pot, 8),
        ATV::cpu::gpw::GpwAtom(rB, basis, pot, 8)};

    auto set_uks = [](auto& m) {
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 0;
        m.n_alpha = 7;
        m.n_beta = 5;
        m.max_iter = 80;
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
    ASSERT_TRUE(rc.converged) << "CPU O2 UKS SCF failed iters=" << rc.iters;
    ASSERT_TRUE(std::isfinite(rc.e_total));

    ATV::cuda::gpw::GpwDft gpu;
    set_uks(gpu);
    gpu.build(atoms);
    const auto& rg = gpu.run(atoms);
    ASSERT_TRUE(rg.converged) << "CUDA O2 UKS SCF failed iters=" << rg.iters;
    ASSERT_TRUE(std::isfinite(rg.e_total));
    EXPECT_FALSE(rg.P_beta.size() == 0);
    EXPECT_FALSE(rg.veff_a.empty());
    EXPECT_FALSE(rg.veff_b.empty());
    // Multiplicity 3: Löwdin <S2> near 2.
    EXPECT_GT(rg.s2, 1.5);

    // p-shell O product composite (cuFFT/FFTW); wider than H2 pure-s 5e-6.
    constexpr double kETolHa = 5.0e-5;
    const double de = std::abs(rg.e_total - rc.e_total);
    EXPECT_LT(de, kETolHa) << "E_cuda=" << rg.e_total << " E_cpu=" << rc.e_total
                           << " |dE|=" << de;
}

