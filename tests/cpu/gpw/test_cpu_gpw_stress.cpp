/// @file tests/cpu/gpw/test_cpu_gpw_stress.cpp
/// @brief Unit gates for GPW cell stress (analytic fixed-P + multi-SCF FD oracle).
///
/// E2E CP2K / analytic-vs-FD water twin lives in
/// regressions/gpw/gpw-analytic-stress/ and regressions/gpw/gpw-stress-water/.

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;
using namespace ATV::cpu::gpw;

namespace {

GpwDft cheap_h2_cfg() {
    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0; // Bohr
    cfg.cutoff_ry = 120.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 80;
    cfg.nonperiodic = true;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    return cfg;
}

std::vector<GpwAtom> cheap_h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    return {GpwAtom(rA, basis, pot, 1), GpwAtom(rB, basis, pot, 1)};
}

} // namespace

TEST(GpwStress, AnalyticOffDiagonalZeroOrthorhombic) {
    const auto atoms = cheap_h2_atoms();
    const auto s = gpw_stress(atoms, cheap_h2_cfg(), 3e-3);
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(s[static_cast<std::size_t>(k)])) << k;
    EXPECT_DOUBLE_EQ(s[3], 0.0);
    EXPECT_DOUBLE_EQ(s[4], 0.0);
    EXPECT_DOUBLE_EQ(s[5], 0.0);
    const double max_diag =
        std::max({std::abs(s[0]), std::abs(s[1]), std::abs(s[2])});
    std::printf(
        "[GPW stress analytic] Voigt Ha/Bohr3: %.6e %.6e %.6e | 0 0 0\n", s[0],
        s[1], s[2]);
    EXPECT_GT(max_diag, 1e-8);
}

TEST(GpwStress, AnalyticMatchesFdDiagonal) {
    // Hellmann–Feynman fixed-P product vs multi-SCF strain-FD oracle on the
    // same cheap H₂ fixture. Residual is SCF residual / O(h²); gate is unit-
    // tier loose (e2e water twin uses 1e-5).
    const auto atoms = cheap_h2_atoms();
    const auto cfg = cheap_h2_cfg();
    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
    }
    EXPECT_LT(max_err, 5e-5) << "analytic vs multi-SCF FD diagonal";
}

namespace {

/// Toy single-s-primitive periodic H₂ — keeps the axes-mask gate at unit-tier
/// cost (small grid, seconds) unlike the DZV MT fixture above.
std::vector<GpwAtom> toy_h2_atoms() {
    auto s_prim = [](double alpha, const double pos[3]) {
        GridBasisFunction bf;
        GridPrimitive p;
        p.alpha = alpha;
        p.center[0] = pos[0];
        p.center[1] = pos[1];
        p.center[2] = pos[2];
        p.l[0] = p.l[1] = p.l[2] = 0;
        p.coeff = 1.0;
        bf.push_back(p);
        return bf;
    };
    auto hydrogen = [&](const double pos[3]) {
        GpwAtom a;
        a.z_val = 1.0;
        a.z = 1;
        a.pos[0] = pos[0];
        a.pos[1] = pos[1];
        a.pos[2] = pos[2];
        a.basis = {s_prim(0.5, pos)};
        a.rloc = 0.2;
        a.cloc = {-4.17890044, 0.72446331};
        return a;
    };
    const double rA[3] = {3.3, 4.0, 4.0};
    const double rB[3] = {4.7, 4.1, 3.9};
    return {hydrogen(rA), hydrogen(rB)};
}

GpwDft toy_h2_cfg() {
    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 8.0; // Bohr, periodic
    cfg.cutoff_ry = 80.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 80;
    cfg.xc = XcKind::PADE;
    return cfg;
}

} // namespace

TEST(GpwStress, AxesMaskSkipsFrozenAxesAnalyticAndFd) {
    const auto atoms = toy_h2_atoms();
    const auto cfg = toy_h2_cfg();
    const double h = 3e-3;
    const auto full_an = gpw_stress(atoms, cfg, h);
    const auto x_an = gpw_stress(atoms, cfg, h, /*axes*/ {true, false, false});
    EXPECT_NEAR(x_an[0], full_an[0], 1e-12);
    EXPECT_DOUBLE_EQ(x_an[1], 0.0);
    EXPECT_DOUBLE_EQ(x_an[2], 0.0);

    const auto full_fd = gpw_stress_fd(atoms, cfg, h);
    const auto x_fd =
        gpw_stress_fd(atoms, cfg, h, /*axes*/ {true, false, false});
    EXPECT_NEAR(x_fd[0], full_fd[0], 1e-12);
    EXPECT_DOUBLE_EQ(x_fd[1], 0.0);
    EXPECT_DOUBLE_EQ(x_fd[2], 0.0);

    EXPECT_THROW(gpw_stress(atoms, cfg, h, {false, false, false}),
                 std::invalid_argument);
    EXPECT_THROW(gpw_stress_fd(atoms, cfg, h, {false, false, false}),
                 std::invalid_argument);
}

// State overload (xl-bomd-linearized-stress): the analytic fixed-P stress at
// a supplied stationary payload equals the classic path — same payload
// extraction + FD loop, minus the internal reference SCF. Residual is the
// SCF-tolerance difference between the two independently converged densities.
TEST(GpwStress, StateOverloadMatchesClassicAnalytic) {
    const auto atoms = toy_h2_atoms();
    const double h = 3e-3;
    // GpwDft is move-only: build+run a throwaway solver for the payload, then
    // hand fresh configs into the free-function overloads (each copies fields
    // internally via the move-from-return-value pattern of toy_h2_cfg()).
    GpwDft solver = toy_h2_cfg();
    solver.build(atoms);
    const GpwResult state = solver.run(atoms);
    ASSERT_TRUE(state.converged);
    const auto s_state = gpw_stress(atoms, toy_h2_cfg(), state, h);
    const auto s_classic = gpw_stress(atoms, toy_h2_cfg(), h);
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(s_state[static_cast<std::size_t>(a)],
                    s_classic[static_cast<std::size_t>(a)], 1e-8)
            << a;
    EXPECT_DOUBLE_EQ(s_state[3], 0.0);
    EXPECT_DOUBLE_EQ(s_state[4], 0.0);
    EXPECT_DOUBLE_EQ(s_state[5], 0.0);
}

// State overload: a non-stationary payload (converged=false) is refused —
// no silent stress at an arbitrary density.
TEST(GpwStress, StateOverloadRejectsUnconvergedState) {
    const auto atoms = toy_h2_atoms();
    const auto cfg = toy_h2_cfg();
    GpwResult state; // default: converged = false, empty payload
    EXPECT_THROW(gpw_stress(atoms, cfg, state, 3e-3), std::invalid_argument);
}

TEST(GpwStress, EvaluatorWritesKStressAnalyticDefault) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    constexpr double b2a = 1.0 / 1.8897261245650618;
    frame.span(kAtom_x)[0] = 5.5 * b2a;
    frame.span(kAtom_y)[0] = 6.0 * b2a;
    frame.span(kAtom_z)[0] = 6.0 * b2a;
    frame.span(kAtom_x)[1] = 6.7 * b2a;
    frame.span(kAtom_y)[1] = 6.1 * b2a;
    frame.span(kAtom_z)[1] = 5.9 * b2a;
    frame.box = cpu::Box::cubic(12.0 * b2a);

    const Kind H{molrs::Element::H,
                 GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                 GthPotential(molrs::Element::H, "GTH-PADE-q1")};

    GPWEvaluator eval(/*kinds*/ {H}, /*cutoff_ry*/ 120.0, /*max_iter*/ 80,
                      /*tol*/ 1e-8, /*mix*/ 0.3, /*xc*/ XcKind::PADE,
                      /*nonperiodic*/ true, /*compute_forces*/ false);
    eval.compute_stress = true;
    eval.stress_use_fd = false; // product default
    eval.stress_strain_h = 3e-3;

    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);

    EXPECT_TRUE(eval.last_result().converged);
    EXPECT_TRUE(std::isfinite(frame.meta.get(kEnergy)));
    const auto& st = frame.meta.get(kStress);
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(st[static_cast<std::size_t>(k)])) << k;
    EXPECT_NEAR(st[3], 0.0, 1e-14);
    EXPECT_NEAR(st[4], 0.0, 1e-14);
    EXPECT_NEAR(st[5], 0.0, 1e-14);
    const auto& sh = eval.last_stress_ha_bohr3();
    EXPECT_GT(std::max({std::abs(sh[0]), std::abs(sh[1]), std::abs(sh[2])}),
              1e-8);
}

// ── Multi-k analytic stress ────────────────────────────────────────────────

TEST(GpwStress, KpointAnalyticMatchesFdDiagonal) {
    // Periodic H2, mesh 1x1x2 (TR → 1 k at κ_z = +1/4). Hellmann–Feynman
    // fixed-P_k product vs multi-SCF strain-FD oracle.
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
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

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "k-mesh analytic vs multi-SCF FD diagonal";
}

TEST(GpwStress, KpointNoThrowAndFinite) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};
    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 80.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-7;
    cfg.max_iter = 50;
    cfg.xc = XcKind::PADE;
    cfg.local_pp_on_grid = true;
    cfg.k_mesh[0] = cfg.k_mesh[1] = cfg.k_mesh[2] = 2; // MP 2x2x2 TR
    EXPECT_NO_THROW({
        const auto s = gpw_stress(atoms, cfg, 4e-3);
        for (int k = 0; k < 6; ++k)
            EXPECT_TRUE(std::isfinite(s[static_cast<std::size_t>(k)])) << k;
        EXPECT_DOUBLE_EQ(s[3], 0.0);
        EXPECT_DOUBLE_EQ(s[4], 0.0);
        EXPECT_DOUBLE_EQ(s[5], 0.0);
    });
}

// ── Γ-UKS analytic stress ──────────────────────────────────────────────────

TEST(GpwStress, UksGammaAnalyticMatchesFdDiagonal) {
    // High-spin H₂ (Nα=2, Nβ=0) Γ-UKS: fixed-P spin Lagrangian vs multi-SCF FD.
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.n_occ = 0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.xc = XcKind::PADE;
    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  UKS axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    EXPECT_GT(max_diag, 1e-8);
    EXPECT_LT(max_err, 5e-5) << "UKS analytic vs multi-SCF FD diagonal";
}

TEST(GpwStress, UksKpointAnalyticMatchesFdDiagonal) {
    // High-spin H₂ multi-k UKS: fixed-{P_k^σ} vs multi-SCF FD.
    // Geometry: compressed bond + small 1×1×2 mesh (TR-folded → 1 k).
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};
    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
    cfg.cutoff_ry = 100.0;
    cfg.n_occ = 0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.tol = 1e-8;
    cfg.max_iter = 60;
    cfg.xc = XcKind::PADE;
    cfg.local_pp_on_grid = true;
    cfg.nonperiodic = false;
    cfg.k_mesh[0] = 1;
    cfg.k_mesh[1] = 1;
    cfg.k_mesh[2] = 2;
    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  k-UKS axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    const double max_diag =
        std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
    EXPECT_GT(max_diag, 1e-8);
    EXPECT_LT(max_err, 1e-4) << "k-UKS analytic vs multi-SCF FD diagonal";
}

TEST(GpwStress, ClosedShellKUksMatchesKRks) {
    // Closed-shell multi-k UKS (nα=nβ=1) stress ≡ multi-k RKS (n_occ=1).
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.0, 6.0, 6.0};
    const double rB[3] = {7.0, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};
    auto base = []() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 80.0;
        m.tol = 1e-7;
        m.max_iter = 50;
        m.xc = XcKind::PADE;
        m.local_pp_on_grid = true;
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
    const double h = 4e-3;
    const auto sr = gpw_stress(atoms, rks, h);
    const auto su = gpw_stress(atoms, uks, h);
    double max_d = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double d = std::abs(sr[static_cast<std::size_t>(a)]
                                  - su[static_cast<std::size_t>(a)]);
        max_d = std::max(max_d, d);
        std::printf("  closed k-UKS vs k-RKS axis %d |d|=%.3e\n", a, d);
    }
    EXPECT_LT(max_d, 5e-4) << "closed-shell k-UKS stress ≡ k-RKS";
}

TEST(GpwStress, UksKpointGgaAnalyticMatchesFdDiagonal) {
    // Multi-k UKS pure spin-GGA residual (RevPBE / RPBE / BLYP): fixed-P_k^σ
    // Lagrangian must dual-gate vs multi-SCF FD. PADE/PBE/PBEsol already
    // covered by UksKpointAnalyticMatchesFdDiagonal + Γ GGA UKS stress.
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};
    const XcKind ggas[] = {XcKind::RevPBE, XcKind::RPBE, XcKind::BLYP};
    const char* names[] = {"RevPBE", "RPBE", "BLYP"};
    for (int ig = 0; ig < 3; ++ig) {
        GpwDft cfg;
        cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0;
        cfg.cutoff_ry = 100.0;
        cfg.n_occ = 0;
        cfg.n_alpha = 2;
        cfg.n_beta = 0;
        cfg.tol = 1e-8;
        cfg.max_iter = 60;
        cfg.xc = ggas[ig];
        cfg.local_pp_on_grid = true;
        cfg.nonperiodic = false;
        cfg.k_mesh[0] = 1;
        cfg.k_mesh[1] = 1;
        cfg.k_mesh[2] = 2;
        // z-only axes mask keeps the three-GGA sweep unit-tier (the k axis is
        // the one exercising the k-summed fold; x/y FD coverage lives in the
        // cheaper RKS KpointAnalyticMatchesFdDiagonal).
        const std::array<bool, 3> axes = {false, false, true};
        const double h = 3e-3;
        const auto san = gpw_stress(atoms, cfg, h, axes);
        const auto sfd = gpw_stress_fd(atoms, cfg, h, axes);
        const double err = std::abs(san[2] - sfd[2]);
        std::printf("  k-UKS %s axis z  an=%.6e  fd=%.6e  |d|=%.3e\n",
                    names[ig], san[2], sfd[2], err);
        EXPECT_TRUE(std::isfinite(san[2]));
        EXPECT_TRUE(std::isfinite(sfd[2]));
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_GT(std::abs(san[2]), 1e-8) << names[ig];
        EXPECT_LT(err, 1e-4)
            << "k-UKS " << names[ig] << " analytic vs multi-SCF FD";
    }
}

// ── Γ-UKS + Dudarev DFT+U analytic stress ─────────────────────────────────

TEST(GpwStress, DftUUksAnalyticMatchesFdDiagonal) {
    // High-spin H₂ UKS with PLUS_U on H L=0: unit-tier smoke that the
    // fixed-P Lagrangian includes E_U[n(Pσ,S)] and no longer throws. Dual-gate
    // residual is looser than bare UKS (empty β + U on s shells is a harsh
    // toy); product NiO dual-gate lives in reg_gpw-dftu-stress (~1e-5).
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.n_occ = 0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.xc = XcKind::PADE;
    cfg.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_ev=*/4.0});
    // Variational ½(V̂S+SV̂) scatter (2026-07-31) converges slower on this
    // strong-overlap toy: ~90–150 DIIS iters instead of <80.
    cfg.max_iter = 300;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  DFT+U axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        const double max_diag =
            std::max({std::abs(san[0]), std::abs(san[1]), std::abs(san[2])});
        EXPECT_GT(max_diag, 1e-8);
        // Unit smoke: catch order-of-magnitude bugs; e2e NiO gate is tight.
        EXPECT_LT(max_err, 1e-3)
            << "DFT+U UKS analytic vs multi-SCF FD diagonal";
    });
}

// ── Multi-k + Dudarev DFT+U analytic stress ───────────────────────────────

TEST(GpwStress, DftUKpointUksAnalyticMatchesFdDiagonal) {
    // High-spin H₂ UKS + MP 1×1×2 + PLUS_U on H L=0: unit-tier smoke that
    // energy_fixed_P_k includes BZ-averaged E_U[n({P_k},{S_k})] and dual-gates
    // against multi-SCF FD. Product residual lives in
    // reg_gpw-dftu-kpoint-stress.
    // Symmetric compressed bond (matches reg_gpw-dftu-kpoint-stress): the
    // asymmetric cheap_h2 toy oscillates under the variational +U outer loop
    // at U=4 (multi-branch strong-overlap regime), the symmetric one is a
    // single branch.
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms{GpwAtom(rA, basis, pot, 1),
                                     GpwAtom(rB, basis, pot, 1)};
    GpwDft cfg = cheap_h2_cfg();
    cfg.cutoff_ry = 100.0;
    cfg.n_occ = 0;
    cfg.n_alpha = 2;
    cfg.n_beta = 0;
    cfg.xc = XcKind::PADE;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.k_mesh[0] = 1;
    cfg.k_mesh[1] = 1;
    cfg.k_mesh[2] = 2;
    cfg.max_iter = 300; // variational +U scatter needs a larger DIIS budget
    cfg.mix = 0.35;
    cfg.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_ev=*/4.0});
    const double h = 3e-3;
    // z-only axes mask keeps the +U k-path smoke unit-tier (the k axis is the
    // one exercising the BZ-averaged E_U fold; full-axes product residual
    // lives in reg_gpw-dftu-kpoint-stress).
    const std::array<bool, 3> axes = {false, false, true};
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h, axes);
        const auto sfd = gpw_stress_fd(atoms, cfg, h, axes);
        const double err = std::abs(san[2] - sfd[2]);
        std::printf("  k+DFT+U axis z  an=%.6e  fd=%.6e  |d|=%.3e\n", san[2],
                    sfd[2], err);
        EXPECT_TRUE(std::isfinite(san[2]));
        EXPECT_TRUE(std::isfinite(sfd[2]));
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_GT(std::abs(san[2]), 1e-8);
        // Unit smoke: order-of-magnitude; e2e dual-gate is tighter.
        EXPECT_LT(err, 2e-3)
            << "multi-k DFT+U UKS analytic vs multi-SCF FD diagonal";
    });
}

// ── Γ RKS + DFT-D4 analytic stress ────────────────────────────────────────
TEST(GpwStress, AnalyticMatchesFdWithD4) {
    // Geometry-only D4 folded into fixed-P Lagrangian; dual-gate vs multi-SCF
    // FD on cheap MT H₂. Residual unit-tier loose; e2e is reg_gpw-d4-stress.
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.d4 = true;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  D4 axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_LT(max_err, 5e-5) << "Γ RKS + D4 analytic vs multi-SCF FD";
    });
}

// ── Γ RKS + DFT-D3 (zero damping) analytic stress ─────────────────────────
TEST(GpwStress, AnalyticMatchesFdWithD3) {
    // Geometry-only D3 zero damping folded into the fixed-P Lagrangian;
    // dual-gate vs multi-SCF FD on cheap MT H₂. e2e CP2K pin is
    // regressions/gpw/gpw-d3-stress.
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.d3 = true;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  D3 axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_LT(max_err, 5e-5) << "Γ RKS + D3-zero analytic vs multi-SCF FD";
    });
}

// D3 zero damping must actually reach the stress, not be silently ignored:
// switching it on has to move the diagonal (guards the silently-dropped-E_disp
// failure mode the multi-k MBD path had).
TEST(GpwStress, D3ChangesGammaDiagonal) {
    const auto atoms = cheap_h2_atoms();
    GpwDft plain = cheap_h2_cfg();
    GpwDft with_d3 = cheap_h2_cfg();
    with_d3.d3 = true;
    const auto s0 = gpw_stress(atoms, plain, 3e-3);
    const auto s1 = gpw_stress(atoms, with_d3, 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf("[GPW stress D3] max|sigma(d3)-sigma(plain)|=%.3e Ha/Bohr3\n",
                max_shift);
    EXPECT_GT(max_shift, 1e-12) << "D3 is not entering the stress Lagrangian";
}

// Multi-k fixed-P_k Lagrangian must also fold D3-zero (geometry-only): same
// silently-dropped-E_disp failure mode as the multi-k MBD path, guarded for
// the k-mesh product. e2e CP2K pin is reg_gpw-d3-kpoint-stress.
TEST(GpwStress, D3ChangesKpointDiagonal) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};
    auto make_kcfg = []() {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 80.0;
        m.n_occ = 1;
        m.tol = 1e-7;
        m.max_iter = 50;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        return m;
    };
    GpwDft plain = make_kcfg();
    GpwDft with_d3 = make_kcfg();
    with_d3.d3 = true;
    const auto s0 = gpw_stress(atoms, plain, 3e-3);
    const auto s1 = gpw_stress(atoms, with_d3, 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf(
        "[GPW stress D3 k-mesh] max|sigma(d3)-sigma(plain)|=%.3e Ha/Bohr3\n",
        max_shift);
    EXPECT_GT(max_shift, 1e-12)
        << "D3 is not entering the multi-k stress Lagrangian";
}

// Multi-k RKS + D3-zero: analytic fixed-P_k vs multi-SCF strain-FD dual gate
// on a cheap periodic H₂ mesh (unit-tier residual; e2e pin is the water-dimer
// multi-k CP2K twin).
TEST(GpwStress, AnalyticMatchesFdWithD3Kpoint) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
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
    cfg.d3 = true;
    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  D3 k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5)
        << "multi-k RKS + D3-zero analytic vs multi-SCF FD";
}

// ── Γ RKS + DFT-D3(BJ) analytic stress ────────────────────────────────────
TEST(GpwStress, AnalyticMatchesFdWithD3Bj) {
    // Geometry-only D3(BJ) folded into the fixed-P Lagrangian; dual-gate vs
    // multi-SCF FD on cheap MT H₂. Residual unit-tier loose; e2e CP2K pin is
    // reg_gpw-d3bj-stress.
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.d3bj = true;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  D3BJ axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_LT(max_err, 5e-5) << "Γ RKS + D3(BJ) analytic vs multi-SCF FD";
    });
}

// ── Γ RKS + DFT-D3 zero damping + three-body ATM (C9) analytic stress ─────

// Multi-k + D3(BJ): k-summed fixed-P_k Lagrangian folds E_disp^D3(BJ)
// (rebuilt at each strained cell) — dual-gate analytic vs multi-SCF FD.
// e2e CP2K pin: reg_gpw-d3bj-kpoint-stress.
TEST(GpwStress, D3BjMultiKAnalyticMatchesFd) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
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
    cfg.d3bj = true;

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  D3BJ k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "multi-k + D3(BJ) analytic vs multi-SCF FD";
}

// D3(BJ) must actually reach the multi-k stress Lagrangian.
TEST(GpwStress, D3BjMultiKChangesDiagonal) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};

    auto mk = []() {
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
        return cfg;
    };
    GpwDft plain = mk();
    GpwDft with = mk();
    with.d3bj = true;

    const auto s0 = gpw_stress(atoms, plain, 3e-3);
    const auto s1 = gpw_stress(atoms, with, 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf(
        "[GPW stress D3BJ multi-k] max|sigma(d3bj)-sigma(plain)|=%.3e Ha/Bohr3\n",
        max_shift);
    EXPECT_GT(max_shift, 1e-12)
        << "D3(BJ) is not entering the multi-k stress Lagrangian";
}

TEST(GpwStress, AnalyticMatchesFdWithD3Atm) {
    // Geometry-only D3 + ATM folded into the fixed-P Lagrangian; dual-gate vs
    // multi-SCF FD on a cheap *periodic* H₂ so lattice-image triples (CP2K
    // doabc CASE 2/3) actually engage — an isolated pair has no triple. The
    // short d3_rcut_bohr also exercises the configurable CP2K R_CUTOFF member
    // through both stress paths' config copies. Residual unit-tier loose;
    // e2e CP2K pin is reg_gpw-d3-atm-stress.
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0; // Bohr
    cfg.cutoff_ry = 80.0; // identity is grid-consistent — keep the twin cheap
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 80;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    cfg.d3 = true;
    cfg.d3_atm = true;
    cfg.d3_rcut_bohr = 18.0; // Bohr — ±2 image window keeps triples cheap

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  D3+ATM axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "Γ RKS + D3 + ATM analytic vs multi-SCF FD";
}

// Multi-k + D3 zero + ATM (C9): the k-summed fixed-P_k Lagrangian folds
// E_disp^{D3+ATM} rebuilt at each strained cell — dual-gate analytic vs
// multi-SCF FD on the same periodic H2 mesh-1x1x2 fixture used by the MBD
// multi-k unit twin. e2e CP2K pin is reg_gpw-d3-atm-kpoint-stress.
TEST(GpwStress, D3AtmMultiKAnalyticMatchesFd) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.7, 6.1, 5.9};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};

    GpwDft cfg;
    cfg.cell[0] = cfg.cell[1] = cfg.cell[2] = 12.0; // Bohr
    cfg.cutoff_ry = 80.0;
    cfg.n_occ = 1;
    cfg.tol = 1e-8;
    cfg.max_iter = 80;
    cfg.nonperiodic = false;
    cfg.local_pp_on_grid = true;
    cfg.xc = XcKind::PADE;
    cfg.k_mesh[0] = 1;
    cfg.k_mesh[1] = 1;
    cfg.k_mesh[2] = 2;
    cfg.d3 = true;
    cfg.d3_atm = true;
    cfg.d3_rcut_bohr = 18.0; // Bohr — ±2 image window keeps triples cheap

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  D3+ATM k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "multi-k + D3 + ATM analytic vs multi-SCF FD";
}

// ── Γ RKS + MBD@rsSCS analytic stress ─────────────────────────────────────
// Free-atom (isolated) MBD is geometry-only, so it enters the fixed-P
// Lagrangian exactly like D3/D4: analytic must track the multi-SCF strain-FD.
// e2e pin (periodic MBD, pymbd E_disp) is reg_gpw-mbd-stress.
TEST(GpwStress, AnalyticMatchesFdWithMbd) {
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.mbd = true;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  MBD axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_LT(max_err, 5e-5) << "Γ RKS + MBD analytic vs multi-SCF FD";
    });
}

// MBD must actually reach the stress, not be silently ignored: switching it on
// has to move the diagonal. Guards the failure mode the multi-k path had.
TEST(GpwStress, MbdChangesGammaDiagonal) {
    const auto atoms = cheap_h2_atoms();
    GpwDft plain = cheap_h2_cfg();
    GpwDft with_mbd = cheap_h2_cfg();
    with_mbd.mbd = true;
    const auto s0 = gpw_stress(atoms, plain, 3e-3);
    const auto s1 = gpw_stress(atoms, with_mbd, 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf("[GPW stress MBD] max|sigma(mbd)-sigma(plain)|=%.3e Ha/Bohr3\n",
                max_shift);
    EXPECT_GT(max_shift, 1e-12) << "MBD is not entering the stress Lagrangian";
}

// ── Γ RKS + free-atom XDM analytic stress ─────────────────────────────────
// Geometry-only XDM folded into the fixed-P Lagrangian; dual-gate vs
// multi-SCF FD on cheap MT H₂. Residual unit-tier loose; e2e dual-python
// strain pins are reg_gpw-xdm-stress.
TEST(GpwStress, AnalyticMatchesFdWithXdm) {
    const auto atoms = cheap_h2_atoms();
    GpwDft cfg = cheap_h2_cfg();
    cfg.xdm = true;
    const double h = 3e-3;
    EXPECT_NO_THROW({
        const auto san = gpw_stress(atoms, cfg, h);
        const auto sfd = gpw_stress_fd(atoms, cfg, h);
        double max_err = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double err = std::abs(san[static_cast<std::size_t>(a)]
                                        - sfd[static_cast<std::size_t>(a)]);
            max_err = std::max(max_err, err);
            std::printf("  XDM axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                        san[static_cast<std::size_t>(a)],
                        sfd[static_cast<std::size_t>(a)], err);
            EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
            EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
        }
        EXPECT_DOUBLE_EQ(san[3], 0.0);
        EXPECT_DOUBLE_EQ(san[4], 0.0);
        EXPECT_DOUBLE_EQ(san[5], 0.0);
        EXPECT_LT(max_err, 5e-5) << "Γ RKS + XDM analytic vs multi-SCF FD";
    });
}

// XDM must actually reach the stress, not be silently ignored: switching it
// on has to move the diagonal (same guard shape as MbdChangesGammaDiagonal).
TEST(GpwStress, XdmChangesGammaDiagonal) {
    const auto atoms = cheap_h2_atoms();
    GpwDft plain = cheap_h2_cfg();
    GpwDft with_xdm = cheap_h2_cfg();
    with_xdm.xdm = true;
    const auto s0 = gpw_stress(atoms, plain, 3e-3);
    const auto s1 = gpw_stress(atoms, with_xdm, 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf("[GPW stress XDM] max|sigma(xdm)-sigma(plain)|=%.3e Ha/Bohr3\n",
                max_shift);
    EXPECT_GT(max_shift, 1e-12) << "XDM is not entering the stress Lagrangian";
}

// Multi-k + MBD: the k-summed fixed-P_k Lagrangian folds E_disp^MBD (rebuilt
// at each strained cell) — dual-gate analytic vs multi-SCF strain-FD on the
// same periodic H2 mesh-1x1x2 fixture as KpointAnalyticMatchesFdDiagonal.
TEST(GpwStress, MbdMultiKAnalyticMatchesFd) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
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
    cfg.mbd = true;
    cfg.mbd_k_mesh[0] = cfg.mbd_k_mesh[1] = cfg.mbd_k_mesh[2] = 2;

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  MBD k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "multi-k + MBD analytic vs multi-SCF FD";
}

// Multi-k + free-atom XDM: the k-summed fixed-P_k Lagrangian folds
// E_disp^XDM (rebuilt at each strained cell) — dual-gate analytic vs
// multi-SCF strain-FD on the same periodic H2 mesh-1x1x2 fixture as
// MbdMultiKAnalyticMatchesFd. e2e dual-python strain pins are
// reg_gpw-xdm-kpoint-stress.
TEST(GpwStress, XdmMultiKAnalyticMatchesFd) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
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
    cfg.xdm = true;

    const double h = 3e-3;
    const auto san = gpw_stress(atoms, cfg, h);
    const auto sfd = gpw_stress_fd(atoms, cfg, h);
    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double err = std::abs(san[static_cast<std::size_t>(a)]
                                    - sfd[static_cast<std::size_t>(a)]);
        max_err = std::max(max_err, err);
        std::printf("  XDM k-mesh axis %d  an=%.6e  fd=%.6e  |d|=%.3e\n", a,
                    san[static_cast<std::size_t>(a)],
                    sfd[static_cast<std::size_t>(a)], err);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(a)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(a)]));
    }
    EXPECT_DOUBLE_EQ(san[3], 0.0);
    EXPECT_DOUBLE_EQ(san[4], 0.0);
    EXPECT_DOUBLE_EQ(san[5], 0.0);
    EXPECT_LT(max_err, 5e-5) << "multi-k + XDM analytic vs multi-SCF FD";
}

// Multi-k XDM must actually reach the stress Lagrangian: switching it on
// has to move the diagonal (guards silent skip of the k-path dispersion fold).
TEST(GpwStress, XdmChangesMultiKDiagonal) {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    const std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                        GpwAtom(rB, basis, pot, 1)};

    auto make = [](bool xdm) {
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
        cfg.xdm = xdm;
        return cfg;
    };

    const auto s0 = gpw_stress(atoms, make(false), 3e-3);
    const auto s1 = gpw_stress(atoms, make(true), 3e-3);
    double max_shift = 0.0;
    for (int a = 0; a < 3; ++a)
        max_shift = std::max(max_shift,
                             std::abs(s1[static_cast<std::size_t>(a)]
                                      - s0[static_cast<std::size_t>(a)]));
    std::printf(
        "[GPW stress XDM multi-k] max|sigma(xdm)-sigma(plain)|=%.3e Ha/Bohr3\n",
        max_shift);
    EXPECT_GT(max_shift, 1e-12)
        << "XDM is not entering the multi-k stress Lagrangian";
}
