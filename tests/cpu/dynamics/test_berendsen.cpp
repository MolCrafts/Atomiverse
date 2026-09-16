/// @file tests/cpu/dynamics/test_berendsen.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/berendsen.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Two free particles moving along +x with speed `v`; N_f is set explicitly so
/// the scaling factor can be predicted in closed form.
cpu::Frame two_movers(double mass, double v) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = mass;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = v;
        f.span(kAtom_vy)[i] = 0;
        f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(100.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Instantaneous temperature [K] of `f` over `n_dof` degrees of freedom.
double temperature(const cpu::Frame& f, int n_dof) {
    double ke = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        const double vx = f.span(kAtom_vx)[i];
        ke += f.span(kAtom_mass)[i] * vx * vx;
    }
    ke *= 0.5 * units::mass_v2_to_energy;
    return 2.0 * ke / (n_dof * units::kB_ev_per_k);
}

cpu::Context make_ctx(int n_dof) {
    cpu::Context ctx;
    ctx.md.n_dof_global = n_dof;
    return ctx;
}

} // namespace

TEST(Berendsen, ScalesVelocitiesByTheBerendsenFactor) {
    constexpr double mass = 1.0;  // amu
    constexpr double v = 0.01;    // Å/fs
    constexpr double dt = 1.0;    // fs
    constexpr double tau = 100.0; // fs
    constexpr double T0 = 300.0;  // K
    constexpr int n_dof = 6;

    cpu::Frame f = two_movers(mass, v);
    const double T_before = temperature(f, n_dof);

    cpu::md::Berendsen th(T0, tau, dt);
    cpu::Context ctx = make_ctx(n_dof);
    th.build({ctx, f});
    th.launch({ctx, f, ATV::slot_of(md::Phase::end_of_step), 0});

    // λ = sqrt(1 + (dt/τ)(T₀/T − 1)) — Berendsen 1984 Eq. 11.
    const double lambda = std::sqrt(1.0 + (dt / tau) * (T0 / T_before - 1.0));
    EXPECT_NEAR(f.span(kAtom_vx)[0], v * lambda, 1e-12);
    EXPECT_NEAR(f.span(kAtom_vx)[1], v * lambda, 1e-12);
    EXPECT_EQ(f.span(kAtom_vy)[0], 0.0);
}

TEST(Berendsen, MovesTemperatureTowardTheSetpointFromBothSides) {
    constexpr double dt = 1.0, tau = 50.0, T0 = 300.0;
    constexpr int n_dof = 6;

    // Hot start cools, cold start warms.
    for (double v : {0.05, 0.001}) {
        cpu::Frame f = two_movers(1.0, v);
        const double T_before = temperature(f, n_dof);
        cpu::md::Berendsen th(T0, tau, dt);
        cpu::Context ctx = make_ctx(n_dof);
        th.build({ctx, f});
        th.launch({ctx, f, ATV::slot_of(md::Phase::end_of_step), 0});
        const double T_after = temperature(f, n_dof);
        EXPECT_LT(std::fabs(T_after - T0), std::fabs(T_before - T0));
    }
}

TEST(Berendsen, ZeroVelocitiesAreLeftAloneRatherThanDividedByZero) {
    constexpr int n_dof = 6;
    cpu::Frame f = two_movers(1.0, 0.0);
    cpu::md::Berendsen th(300.0, 50.0, 1.0);
    cpu::Context ctx = make_ctx(n_dof);
    th.build({ctx, f});
    th.launch({ctx, f, ATV::slot_of(md::Phase::end_of_step), 0});
    // T sits in the denominator of the Berendsen factor; with no kinetic
    // energy the rescale is undefined and the step must be skipped, not fudged.
    EXPECT_EQ(f.span(kAtom_vx)[0], 0.0);
    EXPECT_TRUE(std::isfinite(f.span(kAtom_vx)[0]));
}

TEST(Berendsen, RejectsNonPositiveConstructorArguments) {
    EXPECT_THROW(cpu::md::Berendsen(0.0, 50.0, 1.0), std::invalid_argument);
    EXPECT_THROW(cpu::md::Berendsen(300.0, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(cpu::md::Berendsen(300.0, 50.0, 0.0), std::invalid_argument);
}

TEST(Berendsen, RampInterpolatesTheSetpointOverTheRunSpan) {
    cpu::md::Berendsen th(100.0, 50.0, 1.0);
    th.set_T_stop(500.0).set_run_span(0, 10);
    cpu::Frame f = two_movers(1.0, 0.02);
    cpu::Context ctx = make_ctx(6);
    th.build({ctx, f});
    th.launch({ctx, f, ATV::slot_of(md::Phase::end_of_step), 5});
    EXPECT_NEAR(th.T_target(), 300.0, 1e-12); // halfway
}
