/// @file tests/cpu/dynamics/test_fix_freeze.cpp
/// @brief Unit: FixFreeze zeros force/velocity and restores positions on a type set.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame wall_fluid_pair() {
    cpu::Frame f(2);
    // atom 0 = wall type 2 at origin; atom 1 = fluid type 1 nearby
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 2.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.0;
    f.span(kAtom_y)[1] = 2.0;
    f.span(kAtom_z)[1] = 3.5;
    f.span(kAtom_mass)[0] = 196.9665;
    f.span(kAtom_mass)[1] = 39.948;
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_vx)[0] = 0.5;
    f.span(kAtom_vy)[0] = -0.3;
    f.span(kAtom_vz)[0] = 0.1;
    f.span(kAtom_vx)[1] = 0.2;
    f.span(kAtom_vy)[1] = 0.0;
    f.span(kAtom_vz)[1] = -0.1;
    f.span(kAtom_fx)[0] = 10.0;
    f.span(kAtom_fy)[0] = 20.0;
    f.span(kAtom_fz)[0] = 30.0;
    f.span(kAtom_fx)[1] = 1.0;
    f.span(kAtom_fy)[1] = 2.0;
    f.span(kAtom_fz)[1] = 3.0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

} // namespace

TEST(FixFreeze, BuildZerosFrozenVelocityAndForce) {
    auto f = wall_fluid_pair();
    cpu::md::FixFreeze freeze(/*type_id=*/2);
    cpu::Context ctx;
    freeze.build({ctx, f});

    EXPECT_DOUBLE_EQ(f.span(kAtom_vx)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_vy)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_vz)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fy)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fz)[0], 0.0);
    // Fluid untouched at build for force; velocity left for InitVelocity path
    // (we only zero frozen type).
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[1], 1.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_vx)[1], 0.2);
}

TEST(FixFreeze, PostForceZerosForceOnlyOnType) {
    auto f = wall_fluid_pair();
    cpu::md::FixFreeze freeze(2);
    cpu::Context ctx;
    freeze.build({ctx, f});
    f.span(kAtom_fx)[0] = 99.0;
    f.span(kAtom_fy)[0] = 88.0;
    f.span(kAtom_fz)[0] = 77.0;
    f.span(kAtom_fx)[1] = 5.0;
    freeze.launch({ctx, f, ATV::slot_of(md::Phase::post_force), 0});
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fy)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fz)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[1], 5.0);
}

TEST(FixFreeze, PostIntegrateRestoresPositionAndZerosVelocity) {
    auto f = wall_fluid_pair();
    cpu::md::FixFreeze freeze(2);
    cpu::Context ctx;
    freeze.build({ctx, f});
    // Simulate integrator drift on the wall atom
    f.span(kAtom_x)[0] = 1.5;
    f.span(kAtom_y)[0] = 2.5;
    f.span(kAtom_z)[0] = 0.2;
    f.span(kAtom_vx)[0] = 1.0;
    f.span(kAtom_x)[1] = 9.0; // fluid moved — must stay
    freeze.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    EXPECT_NEAR(f.span(kAtom_x)[0], 1.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_y)[0], 2.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_z)[0], 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(f.span(kAtom_vx)[0], 0.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_x)[1], 9.0);
}

TEST(FixFreeze, ReportsThreeConstraintsPerFrozenAtom) {
    auto f = wall_fluid_pair();
    cpu::md::FixFreeze freeze(2);
    cpu::Context ctx;
    EXPECT_EQ(freeze.n_constraints(ctx, f), 3);
}
