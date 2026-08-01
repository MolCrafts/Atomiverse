/// @file tests/cpu/gpw/test_cpu_gpw_forces.cpp
/// @brief Unit: analytic GPW forces match the FD energy gradient (tiny H2).
///
/// E2E CP2K FORCE_EVAL parity lives in regressions/gpw/gpw-forces-water/.
/// This file only checks that gpw_forces_analytic is the gradient of the same
/// GpwDft energy (including local_pp_on_grid consistency).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"

using namespace ATV::cpu::gpw;

// Analytic forces must reproduce the finite-difference oracle component-by-
// component on a cheap closed-shell system (H2 / PADE, isolated).
TEST(GpwForces, H2AnalyticVsFiniteDifference) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    // Off-symmetry so every Cartesian component is nonzero.
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true; // product default

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);

    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_abs = 0.0;
    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_abs = std::max(max_abs,
                               std::abs(Ffd[static_cast<std::size_t>(a)][k]));
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW F] atom %d axis %d  analytic % .8f  fd % .8f  d % .2e\n",
                a, k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW F] max|F|=%.6f  max|dF|=%.3e Ha/Bohr\n", max_abs,
                max_err);
    EXPECT_LT(max_err, 5e-5) << "analytic vs FD mismatch (Ha/Bohr)";
}

// k-mesh forces: finite, Newton-3 residual small, nonzero bond force.
// Full analytic↔FD at k lives in regressions/gpw/gpw-kpoint-forces/ (FD is
// 6N SCFs × k-path — not unit-suite budget).
TEST(GpwForces, H2KmeshForcesBasic) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0}; // compressed 1.0 Bohr bond
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 100.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 60;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    cfg.eps_default = 1e-8;
    cfg.k_mesh[0] = 1;
    cfg.k_mesh[1] = 1;
    cfg.k_mesh[2] = 2;

    const auto F = gpw_forces_analytic(atoms, cfg);
    ASSERT_EQ(F.size(), 2u);
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            EXPECT_TRUE(std::isfinite(F[static_cast<std::size_t>(a)][k]));
    // Restoring along x (equal-and-opposite within residual).
    EXPECT_LT(F[0][0], 0.0);
    EXPECT_GT(F[1][0], 0.0);
    EXPECT_NEAR(F[0][0], -F[1][0], 2e-3);
    EXPECT_GT(std::abs(F[0][0]), 5e-3);
    const double sum_mag =
        std::sqrt((F[0][0] + F[1][0]) * (F[0][0] + F[1][0])
                  + (F[0][1] + F[1][1]) * (F[0][1] + F[1][1])
                  + (F[0][2] + F[1][2]) * (F[0][2] + F[1][2]));
    EXPECT_LT(sum_mag, 2e-3);
}

// Multi-k RKS free-energy FD identity (full 6N SCF × k-path + smear) lives in
// regressions/gpw/gpw-metallic-forces/ — beyond the unit 10 s hard cap.

TEST(GpwForces, UksH2AnalyticVsFiniteDifference) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    // Off-symmetry so every Cartesian component is nonzero.
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);

    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW UKS F] atom %d axis %d  analytic % .8f  fd % .8f  d % .2e\n",
                a, k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW UKS F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "UKS analytic vs FD mismatch (Ha/Bohr)";
}

// Multi-k UKS forces: high-spin H2 on a 1×1×2 mesh — finite, restoring,
// Newton-3 residual small. Full analytic↔FD lives in the regression
// REFERENCE (6N × k-UKS SCF is not unit-suite budget).
TEST(GpwForces, H2KmeshUksForcesBasic) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0}; // compressed 1.0 Bohr bond
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 100.0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.tol = 1e-8;
    cfg.max_iter = 60;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    cfg.eps_default = 1e-8;
    cfg.k_mesh[0] = 1;
    cfg.k_mesh[1] = 1;
    cfg.k_mesh[2] = 2;

    const auto F = gpw_forces_analytic(atoms, cfg);
    ASSERT_EQ(F.size(), 2u);
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            EXPECT_TRUE(std::isfinite(F[static_cast<std::size_t>(a)][k]));
    EXPECT_LT(F[0][0], 0.0);
    EXPECT_GT(F[1][0], 0.0);
    EXPECT_NEAR(F[0][0], -F[1][0], 2e-3);
    EXPECT_GT(std::abs(F[0][0]), 5e-3);
    const double sum_mag =
        std::sqrt((F[0][0] + F[1][0]) * (F[0][0] + F[1][0])
                  + (F[0][1] + F[1][1]) * (F[0][1] + F[1][1])
                  + (F[0][2] + F[1][2]) * (F[0][2] + F[1][2]));
    EXPECT_LT(sum_mag, 2e-3);
}

// Closed-shell multi-k UKS (nα=nβ=1) must match multi-k RKS forces.
TEST(GpwForces, H2KmeshUksClosedShellMatchesRks) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto base = []() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        return m;
    };

    GpwDft rks = base();
    rks.n_occ = 1;
    GpwDft uks = base();
    uks.n_alpha = 1;
    uks.n_beta = 1;

    const auto Fr = gpw_forces_analytic(atoms, rks);
    const auto Fu = gpw_forces_analytic(atoms, uks);
    ASSERT_EQ(Fr.size(), 2u);
    ASSERT_EQ(Fu.size(), 2u);
    double max_d = 0.0;
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            max_d = std::max(
                max_d, std::abs(Fr[static_cast<std::size_t>(a)][k]
                                - Fu[static_cast<std::size_t>(a)][k]));
    std::printf("[GPW k-UKS vs RKS] max|dF|=%.3e Ha/Bohr\n", max_d);
    EXPECT_LT(max_d, 5e-4);
}

// Explicit-Γ through k-UKS machinery matches real Γ-UKS assemble.
TEST(GpwForces, H2ExplicitGammaUksMatchesGamma) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto base = []() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_alpha = 2;
        m.n_beta = 0;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        return m;
    };

    GpwDft gamma = base();
    GpwDft kG = base();
    kG.k_points = {{0.0, 0.0, 0.0}};
    kG.k_point_weights = {1.0};

    const auto Fg = gpw_forces_analytic(atoms, gamma);
    const auto Fk = gpw_forces_analytic(atoms, kG);
    ASSERT_EQ(Fg.size(), 2u);
    ASSERT_EQ(Fk.size(), 2u);
    double max_d = 0.0;
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            max_d = std::max(
                max_d, std::abs(Fg[static_cast<std::size_t>(a)][k]
                                - Fk[static_cast<std::size_t>(a)][k]));
    std::printf("[GPW kΓ-UKS vs Γ-UKS] max|dF|=%.3e Ha/Bohr\n", max_d);
    EXPECT_LT(max_d, 1e-5);
}

// Compressed H2: bond force is restoring and equal-and-opposite (Newton 3).
TEST(GpwForces, H2RestoringBond) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0}; // 1.0 Bohr (eq ~1.4)
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;

    const auto F = gpw_forces_analytic(atoms, cfg);
    ASSERT_EQ(F.size(), 2u);
    EXPECT_NEAR(F[0][0], -F[1][0], 1e-4);
    EXPECT_LT(F[0][0], 0.0);
    EXPECT_GT(F[1][0], 0.0);
    EXPECT_GT(std::abs(F[0][0]), 5e-3);
}

// Γ RKS + DFT-D3 (zero damping): product analytic forces must match multi-SCF
// energy FD, i.e. the D3 force is the exact gradient of the D3 energy the SCF
// converged (CN-dependent C6 makes this a non-trivial gradient). e2e CP2K pin
// is reg_gpw-d3-forces.
TEST(GpwForces, AnalyticMatchesFdWithD3) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.d3 = true;

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW D3 F] atom %d axis %d  an=% .8f  fd=% .8f  d=% .2e\n", a,
                k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW D3 F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "Γ RKS + D3 zero analytic vs multi-SCF FD";
}

// Γ RKS + DFT-D3(BJ): same gradient identity for rational (Becke-Johnson)
// damping. e2e CP2K pin is reg_gpw-d3bj-forces.
TEST(GpwForces, AnalyticMatchesFdWithD3Bj) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.d3bj = true;

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW D3BJ F] atom %d axis %d  an=% .8f  fd=% .8f  d=% .2e\n",
                a, k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW D3BJ F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "Γ RKS + D3(BJ) analytic vs multi-SCF FD";
}

// Multi-k force assembly must fold geometry-only D3(BJ) (product path
// used by reg_gpw-d3bj-kpoint-forces). H2 1×1×2: one SCF, two assemblies vs
// standalone dftd3_pbe_bj.
TEST(GpwForces, KpointProductForcesFoldD3Bj) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = [](bool d3bj) {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.d3bj = d3bj;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    ASSERT_TRUE(scf.converged);
    EXPECT_GE(scf.n_kpts, 1);
    EXPECT_FALSE(scf.C_occ_k.empty());
    EXPECT_TRUE(std::isfinite(scf.e_disp));
    EXPECT_LT(scf.e_disp, 0.0);

    const auto F_on = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
    ASSERT_EQ(F_on.size(), 2u);
    ASSERT_EQ(F_plain.size(), 2u);

    std::vector<int> Z = {1, 1};
    std::vector<std::array<double, 3>> pos = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell[3] = {12.0, 12.0, 12.0};
    const D3Result geom =
        dftd3_pbe_bj(Z, pos, cell, /*periodic=*/true, D3Params{},
                     kD3Cp2kPairCutoffBohr, /*s6=*/1.0, /*s8=*/0.7875,
                     /*a1=*/0.4289, /*a2=*/4.4407, /*cp2k_cn=*/true,
                     /*calculate_c9=*/false);

    double max_d = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d =
                (F_on[static_cast<std::size_t>(a)][k]
                 - F_plain[static_cast<std::size_t>(a)][k])
                - geom.forces[static_cast<std::size_t>(a)][k];
            max_d = std::max(max_d, std::abs(d));
        }
    }
    std::printf(
        "[D3BJ k-forces] e_disp=%.6e  max|F_on-F_plain-F_geom|=%.2e\n",
        scf.e_disp, max_d);
    EXPECT_LT(max_d, 1e-8)
        << "multi-k assemble must add geometry-only D3(BJ) forces";
    EXPECT_GT(std::abs(geom.forces[0][0]), 1e-10);
}

TEST(GpwForces, AnalyticMatchesFdWithD4) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.d4 = true;

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW D4 F] atom %d axis %d  an=% .8f  fd=% .8f  d=% .2e\n", a,
                k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW D4 F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "Γ RKS + D4 analytic vs multi-SCF FD";
}

// Γ RKS + free-atom XDM: product analytic forces fold geometry-only XDM.
// e2e pin is reg_gpw-xdm-forces (water dimer dual-Python + FD dual-gate).
TEST(GpwForces, AnalyticMatchesFdWithXdm) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 140.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.xdm = true;

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, 1e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW XDM F] atom %d axis %d  an=% .8f  fd=% .8f  d=% .2e\n", a,
                k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW XDM F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "Γ RKS + XDM analytic vs multi-SCF FD";
}

// Multi-k force assembly must fold geometry-only free-atom XDM (product path
// used by reg_gpw-xdm-kpoint-forces). H2 1×1×2: one SCF, two assemblies vs
// standalone xdm_pbe.
TEST(GpwForces, KpointProductForcesFoldXdm) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = [](bool xdm) {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.xdm = xdm;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    ASSERT_TRUE(scf.converged);
    EXPECT_GE(scf.n_kpts, 1);
    EXPECT_FALSE(scf.C_occ_k.empty());
    EXPECT_TRUE(std::isfinite(scf.e_disp));
    EXPECT_LT(scf.e_disp, 0.0);

    const auto F_on = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
    ASSERT_EQ(F_on.size(), 2u);
    ASSERT_EQ(F_plain.size(), 2u);

    std::vector<int> Z = {1, 1};
    std::vector<std::array<double, 3>> pos = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell[3] = {12.0, 12.0, 12.0};
    constexpr double kAng2Bohr = 1.0 / 0.52917720859;
    constexpr double kRcutXdm = 50.0 * kAng2Bohr;
    const XdmResult geom =
        xdm_pbe(Z, pos, cell, /*periodic=*/true, kRcutXdm);

    double max_d = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d =
                (F_on[static_cast<std::size_t>(a)][k]
                 - F_plain[static_cast<std::size_t>(a)][k])
                - geom.forces[static_cast<std::size_t>(a)][k];
            max_d = std::max(max_d, std::abs(d));
        }
    }
    std::printf(
        "[XDM k-forces] e_disp=%.6e  max|F_on-F_plain-F_geom|=%.2e\n",
        scf.e_disp, max_d);
    EXPECT_LT(max_d, 1e-8)
        << "multi-k assemble must add geometry-only XDM forces";
    EXPECT_GT(std::abs(geom.forces[0][0]), 1e-10);
}

// Γ-UKS + Dudarev DFT+U: occupation-matrix Pulay fold (W ← W − M) is the
// gradient of E_U[n(P,S)]. U−J = 2 eV keeps the DZV s-shell gross-Mulliken
// occupation in a single SCF branch (at 4 eV the toy is multi-branch; see
// .claude/notes/gpw-dftu-projector-divergence.md). Product e2e pin:
// reg_gpw-dftu-forces.
TEST(GpwForces, DftUUksGammaAnalyticMatchesFd) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    // Symmetric compressed bond along x (same as reg_gpw-dftu-forces A).
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 100.0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.tol = 1e-8;
    cfg.max_iter = 120;
    cfg.mix = 0.35;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    cfg.eps_default = 1e-8;
    cfg.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_ev=*/2.0});

    const auto Fa = gpw_forces_analytic(atoms, cfg);
    const auto Ffd = gpw_forces_fd(atoms, cfg, /*h=*/1e-3);
    ASSERT_EQ(Fa.size(), 2u);
    ASSERT_EQ(Ffd.size(), 2u);

    double max_err = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            max_err = std::max(max_err,
                               std::abs(Fa[static_cast<std::size_t>(a)][k]
                                        - Ffd[static_cast<std::size_t>(a)][k]));
            std::printf(
                "[GPW DFT+U Γ F] atom %d axis %d  an=% .8f  fd=% .8f  d=% .2e\n",
                a, k, Fa[static_cast<std::size_t>(a)][k],
                Ffd[static_cast<std::size_t>(a)][k],
                Fa[static_cast<std::size_t>(a)][k]
                    - Ffd[static_cast<std::size_t>(a)][k]);
        }
    }
    std::printf("[GPW DFT+U Γ F] max|dF|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-5) << "Γ-UKS + DFT+U analytic vs multi-SCF FD";

    // U term live: forces move when U is switched on (GpwDft is non-copyable).
    GpwDft plain;
    plain.cell[0] = plain.cell[1] = plain.cell[2] = 12.0;
    plain.cutoff_ry = 100.0;
    plain.n_alpha = 2;
    plain.n_beta = 0;
    plain.tol = 1e-8;
    plain.max_iter = 120;
    plain.mix = 0.35;
    plain.nonperiodic = false;
    plain.local_pp_on_grid = true;
    plain.xc = XcKind::PADE;
    plain.eps_default = 1e-8;
    const auto F0 = gpw_forces_analytic(atoms, plain);
    double dU = 0.0;
    for (int a = 0; a < 2; ++a)
        for (int k = 0; k < 3; ++k)
            dU = std::max(dU, std::abs(Fa[static_cast<std::size_t>(a)][k]
                                       - F0[static_cast<std::size_t>(a)][k]));
    EXPECT_GT(dU, 1e-4) << "U activation floor on Γ forces";
}
