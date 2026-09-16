/// @file tests/cpu/dynamics/test_fix_addforce.cpp
/// @brief Unit tests for cpu::md::FixAddForce (constant F, U=-F·r^u, type filter).
#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame two_atoms() {
    // Atom 0 type=1 (pulled); atom 1 type=2 (background).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 10.0;
    f.span(kAtom_y)[0] = 10.0;
    f.span(kAtom_z)[0] = 5.0;
    f.span(kAtom_x)[1] = 10.0;
    f.span(kAtom_y)[1] = 10.0;
    f.span(kAtom_z)[1] = 15.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 39.948;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = 0.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0.0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
    return f;
}

} // namespace

TEST(FixAddForce, AddsForceOnlyToMatchingType) {
    auto f = two_atoms();
    const double Fz = 0.05;
    cpu::Context ctx;
    cpu::md::FixAddForce pull(0.0, 0.0, Fz, /*type_id=*/1);
    pull.launch({ctx, f, /*stage=*/0, /*step=*/0});

    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fy)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], Fz, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fx)[1], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fy)[1], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[1], 0.0, 1e-15);
}

TEST(FixAddForce, EnergyIsMinusFDotR) {
    auto f = two_atoms();
    // U = -Fz * z_ion = -0.05 * 5.0 = -0.25
    cpu::Context ctx;
    cpu::md::FixAddForce pull(0.0, 0.0, 0.05, 1);
    pull.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), -0.25, 1e-15);
    EXPECT_NEAR(pull.last_energy(), -0.25, 1e-15);
}

TEST(FixAddForce, UnwrappedEnergyUsesImage) {
    auto f = two_atoms();
    // Atom 0: image_z = +1 → zu = 5 + 20 = 25
    f.span(kAtom_image)(0, 2) = static_cast<std::int8_t>(1);
    // U = -0.05 * 25 = -1.25
    cpu::Context ctx;
    cpu::md::FixAddForce pull(0.0, 0.0, 0.05, 1);
    pull.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), -1.25, 1e-12);
}

TEST(FixAddForce, EnergyOnlySkipsForce) {
    auto f = two_atoms();
    cpu::Context ctx;
    cpu::md::FixAddForce pull(0.0, 0.0, 0.05, 1);
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = false;
    pull.launch(lc);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.meta.get(kEnergy), -0.25, 1e-15);
}

TEST(FixAddForce, WrongTypeIsNoOp) {
    auto f = two_atoms();
    cpu::Context ctx;
    cpu::md::FixAddForce pull(0.0, 0.0, 0.05, /*type_id=*/99);
    pull.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[1], 0.0, 1e-15);
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-15);
}
