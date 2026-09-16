/// @file tests/cpu/potentials/test_pair_dpd.cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"

using namespace ATV;

TEST(PairDpd, SoftConservativeEnergy) {
    // rc deliberately != 1: the Groot-Warren conservative energy is
    // U = ½ a r_c (1 − r/r_c)², and the shipped kernel dropped the r_c factor
    // for a long time without any gate noticing, because every DPD test and
    // regression in the tree used r_c = 1 Å, where the bug is invisible.
    // a is a force amplitude [eV/Å], so U must carry one power of length.
    const double a = 25.0, gamma = 0.0, T = 1.0, rc = 2.5, dt = 0.01, r = 1.25;
    // γ=0 → no D/R; w = 1 − r/rc = 0.5 → U = 0.5*25*2.5*0.25 = 7.8125 eV
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;

    cpu::Context ctx;
    cpu::PairDpd dpd(a, gamma, T, rc, dt, 1);
    dpd.set_step(0);
    dpd.build({ctx, f});
    dpd.launch({ctx, f, 0, 0});
    const double w = 1.0 - r / rc;
    EXPECT_NEAR(f.meta.get(kEnergy), 0.5 * a * rc * w * w, 1e-12);
    // Repulsive: F0x < 0
    EXPECT_LT(f.span(kAtom_fx)[0], 0.0);
    EXPECT_NEAR(f.span(kAtom_fx)[0], -f.span(kAtom_fx)[1], 1e-12);
    // The force the same loop applies must be −dU/dr: F = a w. Pinning both
    // together is what makes the missing r_c impossible to reintroduce —
    // energy and force disagreed by exactly r_c while this test passed.
    EXPECT_NEAR(-f.span(kAtom_fx)[0], a * w, 1e-12);
    dpd.destroy({ctx, f});
}

namespace {

// A DPD dimer with an OFF-AXIS separation and non-zero relative velocity, so
// every force component carries both the dissipative and the random term
// (an x-aligned pair at rest would put all the signal in one component).
constexpr double kA = 25.0;        // [eV/Å] soft repulsion amplitude
constexpr double kT = 300.0;       // [K] thermostat temperature
constexpr double kRc = 2.5;        // [Å] cutoff
constexpr double kDt = 0.01;       // [fs] timestep — F^R ∝ 1/√dt
constexpr std::uint64_t kSeed = 7; // stream seed

/// @brief Deterministic two-atom DPD frame; literals only, so two
///        constructions are bitwise identical.
cpu::Frame make_dpd_dimer() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.0;
    f.span(kAtom_y)[1] = 0.6;
    f.span(kAtom_z)[1] = 0.4;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.span(kAtom_vx)[0] = 0.02;
    f.span(kAtom_vy)[0] = -0.01;
    f.span(kAtom_vz)[0] = 0.005;
    f.span(kAtom_vx)[1] = -0.015;
    f.span(kAtom_vy)[1] = 0.03;
    f.span(kAtom_vz)[1] = -0.02;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// @brief Force on atom 0 after ONE launch at the given loop step.
///
/// `set_step` is deliberately NOT called: the step this style samples its
/// random stream at must come from the launch itself (`LaunchContext::step`),
/// which is the only step a component sees when a Driver owns the loop.
///
/// @param gamma Dissipative friction [eV·fs/Å²]; 0 switches both the
///              dissipative and the random term off (σ = √(2γk_BT)).
/// @param step  Loop step handed to `launch` [count].
/// @return `{fx, fy, fz}` on atom 0 [eV/Å].
std::array<double, 3> dpd_force_at_step(double gamma, int step) {
    cpu::Frame f = make_dpd_dimer();
    cpu::Context ctx;
    cpu::PairDpd dpd(kA, gamma, kT, kRc, kDt, kSeed);
    dpd.build({ctx, f});
    dpd.launch({ctx, f, /*stage=*/0, step});
    const std::array<double, 3> force{f.span(kAtom_fx)[0], f.span(kAtom_fy)[0],
                                      f.span(kAtom_fz)[0]};
    dpd.destroy({ctx, f});
    return force;
}

} // namespace

// The DPD thermostat's random force must be redrawn every step: F^R is what
// injects the energy the dissipative term removes, and a stream frozen at one
// draw turns the thermostat into a constant, position-independent kick — the
// system heats or cools monotonically instead of sampling the target
// temperature. `set_step` is the direct/test entry point and nothing calls it
// under a Driver, so the step the stream is keyed on has to be the one the
// launch carries.
TEST(PairDpd, RandomForcesAdvanceWithDriverStep) {
    constexpr double kGamma = 1.0; // [eV·fs/Å²] — σ = √(2γk_BT) ≈ 0.23
    const auto step0 = dpd_force_at_step(kGamma, 0);
    const auto step1 = dpd_force_at_step(kGamma, 1);

    // Identical positions, identical velocities, identical seed: the ONLY
    // difference between the two launches is the step, so every difference
    // below is the random draw and nothing else. The amplitude here is
    // σ·w/√dt ≈ 1 eV/Å, so a real redraw is many orders of magnitude above
    // this floor — the threshold exists to reject a difference that is only
    // rounding.
    EXPECT_GT(std::abs(step0[0] - step1[0]), 1e-9);
    EXPECT_GT(std::abs(step0[1] - step1[1]), 1e-9);
    EXPECT_GT(std::abs(step0[2] - step1[2]), 1e-9);

    // Non-vacuity, and the control that keeps the claim honest: with γ = 0
    // there is no random term at all (σ = 0) and no dissipative one, so the
    // two steps MUST agree exactly. A "fix" that perturbed the force with the
    // step number rather than redrawing the stream would break this.
    const auto quiet0 = dpd_force_at_step(0.0, 0);
    const auto quiet1 = dpd_force_at_step(0.0, 1);
    EXPECT_EQ(quiet0[0], quiet1[0]);
    EXPECT_EQ(quiet0[1], quiet1[1]);
    EXPECT_EQ(quiet0[2], quiet1[2]);
    // …and that control is itself non-vacuous: the conservative force is
    // live, so "equal" is not "both zero".
    EXPECT_GT(std::abs(quiet0[0]), 1e-9);
}
