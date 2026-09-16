/// @file tests/cpu/dynamics/test_fix_efield.cpp
/// @brief Unit tests for cpu::md::FixEfield (F=qE, U=-qE·r^u).
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
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

TEST(FixEfield, ForceIsQTimesE) {
    auto f = two_ions();
    const double Ex = 0.25, Ey = -0.1, Ez = 0.5;
    cpu::Context ctx;
    cpu::md::FixEfield field(Ex, Ey, Ez);
    field.launch({ctx, f, /*stage=*/0, /*step=*/0});

    EXPECT_NEAR(f.span(kAtom_fx)[0], 1.0 * Ex, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fy)[0], 1.0 * Ey, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 1.0 * Ez, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -1.0 * Ex, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fy)[1], -1.0 * Ey, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[1], -1.0 * Ez, 1e-15);
}

TEST(FixEfield, EnergyIsMinusQDotE) {
    auto f = two_ions();
    const double Ex = 0.1, Ey = 0.0, Ez = 0.0;
    // U = -q0*E*x0 - q1*E*x1 = -1*0.1*1 - (-1)*0.1*3 = -0.1 + 0.3 = 0.2
    cpu::Context ctx;
    cpu::md::FixEfield field(Ex, Ey, Ez);
    field.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.2, 1e-15);
    EXPECT_NEAR(field.last_energy(), 0.2, 1e-15);
}

TEST(FixEfield, UnwrappedEnergyUsesImage) {
    auto f = two_ions();
    // Atom 0: image_x = +1 → xu = 1 + 20 = 21
    f.span(kAtom_image)(0, 0) = static_cast<std::int8_t>(1);
    const double Ex = 0.1;
    // U = -1*0.1*21 - (-1)*0.1*3 = -2.1 + 0.3 = -1.8
    cpu::Context ctx;
    cpu::md::FixEfield field(Ex, 0.0, 0.0);
    field.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), -1.8, 1e-12);
}

TEST(FixEfield, EnergyOnlySkipsForce) {
    auto f = two_ions();
    cpu::Context ctx;
    cpu::md::FixEfield field(0.2, 0.0, 0.0);
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = false;
    field.launch(lc);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.meta.get(kEnergy), -0.2 * 1.0 + 0.2 * 3.0, 1e-15);
}
