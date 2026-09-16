/// @file tests/cpu/md/test_pair_morse_smoke.cpp
/// @brief Unit checks for `cpu::PairMorse` energy/force at fixed geometry.
///
/// Multi-step NVE is a regression property (see `regressions/md/lj_argon` for
/// the integrator conservation story). This file only launches one force
/// evaluation — analytic Morse well \(V=D(1-e^{-\alpha(r-r_0)})^2\).

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_morse.h"

namespace {

double morse_v(double D, double alpha, double r0, double r) {
    const double e = std::exp(-alpha * (r - r0));
    return D * (1.0 - e) * (1.0 - e);
}

} // namespace

TEST(PairMorse, EnergyAtEquilibriumIsZero) {
    const double D = 0.5;
    const double alpha = 1.5;
    const double r0 = 2.0;
    const double rcut = 6.0;

    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = r0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    frame.span(ATV::kAtom_mass)[0] = 12.0;
    frame.span(ATV::kAtom_mass)[1] = 12.0;
    frame.box = ATV::cpu::Box::cubic(50.0);

    ATV::cpu::Context ctx;
    ATV::cpu::BuildContext bctx{ctx, frame};
    ATV::cpu::PairMorse morse{D, alpha, r0, rcut};
    morse.build(bctx);

    frame.meta.get(ATV::kEnergy) = 0;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    ATV::cpu::LaunchContext lctx{ctx, frame, 0, 0};
    morse.launch(lctx);

    // Shifted Morse: V(r) = D(1−e^{−α(r−r₀)})² − V(rcut). At r = r₀ the
    // unshifted well is zero, so the reported energy is −V(rcut). Force = 0.
    const double want = -morse_v(D, alpha, r0, rcut);
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy), want, 1e-12);
    EXPECT_NEAR(frame.span(ATV::kAtom_fx)[0], 0.0, 1e-12);
    EXPECT_NEAR(frame.span(ATV::kAtom_fx)[1], 0.0, 1e-12);

    morse.destroy({ctx, frame});
}

TEST(PairMorse, EnergyMatchesAnalyticAtDisplacement) {
    const double D = 0.5;
    const double alpha = 1.5;
    const double r0 = 2.0;
    const double rcut = 6.0;
    const double r = 2.2;

    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = r;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    frame.span(ATV::kAtom_mass)[0] = 12.0;
    frame.span(ATV::kAtom_mass)[1] = 12.0;
    frame.box = ATV::cpu::Box::cubic(50.0);

    ATV::cpu::Context ctx;
    ATV::cpu::BuildContext bctx{ctx, frame};
    ATV::cpu::PairMorse morse{D, alpha, r0, rcut};
    morse.build(bctx);

    frame.meta.get(ATV::kEnergy) = 0;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    ATV::cpu::LaunchContext lctx{ctx, frame, 0, 0};
    morse.launch(lctx);

    // Pair styles may apply a constant shift so V(rcut)=0; relative to r0
    // the unshifted Morse value is the closed form. Check against that with
    // the same shift the implementation uses: V_shift = V(rcut).
    const double v_unshifted = morse_v(D, alpha, r0, r);
    const double v_shift = morse_v(D, alpha, r0, rcut);
    const double want = v_unshifted - v_shift;
    EXPECT_NEAR(frame.meta.get(ATV::kEnergy), want, 1e-12);
    // Attractive when stretched: force on atom 0 points +x (toward atom 1).
    EXPECT_GT(frame.span(ATV::kAtom_fx)[0], 0.0);
    EXPECT_LT(frame.span(ATV::kAtom_fx)[1], 0.0);
    EXPECT_NEAR(frame.span(ATV::kAtom_fx)[0], -frame.span(ATV::kAtom_fx)[1],
                1e-12);

    morse.destroy({ctx, frame});
}
