/// @file tests/cpu/dynamics/test_fix_oscillating_efield.cpp
/// @brief Unit tests for cpu::md::FixOscillatingEfield (E(t), F=qE, U).
#include <cmath>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_oscillating_efield.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame two_ions() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0.0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
    return f;
}

} // namespace

TEST(FixOscillatingEfield, ScheduleCosine) {
    // T = 1000 fs, dt = 1 fs → period in steps = 1000
    cpu::md::FixOscillatingEfield osc(/*Ex0=*/0.0, /*Ey0=*/0.0, /*Ez0=*/0.2,
                                      /*period=*/1000.0, /*dt=*/1.0);
    EXPECT_NEAR(osc.field_z(0), 0.2, 1e-15);
    EXPECT_NEAR(osc.field_z(250), 0.0, 1e-12);  // T/4
    EXPECT_NEAR(osc.field_z(500), -0.2, 1e-15); // T/2
    EXPECT_NEAR(osc.field_z(750), 0.0, 1e-12);  // 3T/4
    EXPECT_NEAR(osc.field_z(1000), 0.2, 1e-15); // T
    EXPECT_NEAR(osc.field_z(125), 0.2 * std::cos(2.0 * M_PI * 125.0 / 1000.0),
                1e-15);
}

TEST(FixOscillatingEfield, ForceAndEnergyAtStep) {
    auto f = two_ions();
    // Place charges on z so U is non-trivial with Ez only.
    f.span(kAtom_z)[0] = 2.0;
    f.span(kAtom_z)[1] = 4.0;
    const double Ez0 = 0.5;
    const double period = 1000.0;
    const double dt = 1.0;
    const int step = 125;
    const double Ez =
        Ez0 * std::cos(2.0 * M_PI * static_cast<double>(step) * dt / period);

    cpu::Context ctx;
    cpu::md::FixOscillatingEfield osc(0.0, 0.0, Ez0, period, dt);
    osc.launch({ctx, f, /*stage=*/0, step});

    EXPECT_NEAR(osc.last_Ez(), Ez, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 1.0 * Ez, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[1], -1.0 * Ez, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-15);
    // U = -q0*Ez*z0 - q1*Ez*z1 = -1*Ez*2 - (-1)*Ez*4 = -2 Ez + 4 Ez = 2 Ez
    EXPECT_NEAR(f.meta.get(kEnergy), 2.0 * Ez, 1e-15);
    EXPECT_NEAR(osc.last_energy(), 2.0 * Ez, 1e-15);
}

TEST(FixOscillatingEfield, PhaseShift) {
    cpu::md::FixOscillatingEfield osc(0.0, 0.0, 1.0, 1000.0, 1.0,
                                      /*phase=*/M_PI / 2.0);
    // cos(0 + π/2) = 0 at step 0
    EXPECT_NEAR(osc.field_z(0), 0.0, 1e-15);
    // cos(2π*250/1000 + π/2) = cos(π/2 + π/2) = cos(π) = -1
    EXPECT_NEAR(osc.field_z(250), -1.0, 1e-12);
}
