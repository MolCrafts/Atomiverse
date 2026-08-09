/// @file tests/cpu/planewave/test_pw_forces_fd.cpp
/// @brief Analytic-vs-finite-difference force gate for plane-wave DFT.
///
/// Split out of `test_pw_forces.cpp`: an FD force oracle costs one SCF solve
/// per displacement, so bundling it with the cheap API/symmetry gates pushed
/// a single binary past `ATV_TEST_TIMEOUT_SECONDS`. Heavy-gated either way —
/// the target name matches `pw_forces` in `ATV_TEST_HEAVY_REGEX`.
///
/// The comparison is done on **one atom** rather than through the full
/// `pw_forces_fd` sweep (which solves 6N times): the identity
/// \(F_\alpha = -\partial E/\partial R_\alpha\) is per-component, so three
/// central differences prove exactly as much as 3N of them and keep the
/// binary inside the unit-test budget.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

/// Si in the 2-atom FCC primitive cell, second atom pushed off the ideal
/// site so the forces are genuinely non-zero.
cpu::Frame si_dimer(double a, double dx) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_mass)[i] = 28.0855;
    }
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = a / 4.0 + dx;
    f.span(kAtom_y)[1] = a / 4.0;
    f.span(kAtom_z)[1] = a / 4.0;
    const double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box.set_cell(cell);
    return f;
}

PW si_cfg() {
    PW cfg;
    cfg.gth_name = "GTH-PBE-q4"; // Si: local part *and* KB projectors
    cfg.ecut_ry = 8.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-6;
    cfg.max_iter = 20;
    cfg.mix_beta = 0.5;
    return cfg;
}

/// Total energy [Ry] at a displaced geometry — the FD oracle's primitive.
double energy_ry_at(const cpu::Frame& base, int atom, int dir, double delta) {
    cpu::Frame f = base;
    const ColKey<StorageReal> axis[3] = {kAtom_x, kAtom_y, kAtom_z};
    f.span(axis[static_cast<std::size_t>(dir)])[atom] +=
        static_cast<StorageReal>(delta);
    PW solver = si_cfg();
    solver.build(f);
    return solver.run(f).energy;
}

} // namespace

TEST(PWForcesFd, AnalyticMatchesFdGthSi) {
    // THE analytic-vs-FD self-consistency gate. Potential-independent: it
    // must hold for whatever pseudopotential is configured, which is what
    // makes it the gate that catches a mis-wired local form factor or a
    // silently dropped nonlocal term.
    //
    // Si/GTH is chosen deliberately over H/GTH: GTH-PBE-q1 for hydrogen is
    // purely local, so an H-only gate cannot see the Kleinman–Bylander
    // contribution at all. `pw_forces_analytic` used to select the nonlocal
    // term on `!upf_path.empty()`, so every GTH run skipped it (and used a
    // bare-Coulomb local form factor, and bare nuclear Z in the Ewald sum).
    // Those three defects are invisible to an H test and fatal here.
    const auto frame = si_dimer(5.43, 0.08);

    const auto Fa = pw_forces_analytic(frame, si_cfg());
    ASSERT_EQ(Fa.size(), 2u);

    constexpr double h = 5e-3; // Å
    const double ry_to_ev = qe_units::rydberg_to_ev;

    double max_diff = 0.0;
    double max_mag = 0.0;
    for (int dir = 0; dir < 3; ++dir) {
        const double e_plus = energy_ry_at(frame, 0, dir, +h);
        const double e_minus = energy_ry_at(frame, 0, dir, -h);
        const double f_fd = -(e_plus - e_minus) * ry_to_ev / (2.0 * h);
        const double f_an = Fa[0][static_cast<std::size_t>(dir)];
        max_diff = std::max(max_diff, std::abs(f_an - f_fd));
        max_mag = std::max(max_mag, std::abs(f_fd));
    }

    // The force must actually be resolvable, or "agreement" is vacuous.
    EXPECT_GT(max_mag, 0.1) << " |F_fd|max=" << max_mag;
    // Residual floor is the XC "egg-box": E_xc is a real-space FFT-grid
    // quadrature, so it carries a weak dependence on where the ions sit
    // relative to the grid that the analytic Hellmann–Feynman sum does not
    // model. It shrinks with grid spacing, so this bound belongs to *these*
    // settings. Measured 2.5e-4 eV/Å against forces of order 1 eV/Å. The old
    // H-dimer gate's 0.8 eV/Å bound was a toy-potential artifact — it passed
    // with the nonlocal term switched off entirely.
    EXPECT_LT(max_diff, 5e-3) << " max |F_a - F_fd|=" << max_diff;
}

TEST(PWForcesFd, AnalyticForcesObeyNewtonsThirdLaw) {
    // The analytic forces on a periodic cell must sum to zero. The residual
    // (measured ~9e-6 eV/Å) is the SCF tolerance leaking through the density,
    // not a broken sum rule. Cheapest probe of a structure-factor or
    // phase-sign error in either the local or the nonlocal term.
    const auto frame = si_dimer(5.43, 0.08);
    const auto Fa = pw_forces_analytic(frame, si_cfg());
    ASSERT_EQ(Fa.size(), 2u);
    for (int k = 0; k < 3; ++k) {
        const double s =
            Fa[0][static_cast<std::size_t>(k)] + Fa[1][static_cast<std::size_t>(k)];
        EXPECT_LT(std::abs(s), 1e-4) << " sum F[" << k << "]=" << s;
    }
}
