/// @file tests/cpu/dynamics/test_piston_electrode.cpp
/// @brief Unit: PistonElectrode force balance, pressure identity, rigid motion.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/piston_electrode.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Two upper-wall atoms (type 3) + one fluid atom (type 1) for type selectivity.
cpu::Frame upper_wall_pair() {
    cpu::Frame f(3);
    // Wall atoms at z=8
    f.span(kAtom_x)[0] = 2.0;
    f.span(kAtom_y)[0] = 2.0;
    f.span(kAtom_z)[0] = 8.0;
    f.span(kAtom_x)[1] = 6.0;
    f.span(kAtom_y)[1] = 6.0;
    f.span(kAtom_z)[1] = 8.0;
    f.span(kAtom_mass)[0] = 12.0;
    f.span(kAtom_mass)[1] = 36.0; // unequal mass → check mass-weighted share
    f.span(kAtom_type)[0] = 3;
    f.span(kAtom_type)[1] = 3;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    // Fluid
    f.span(kAtom_x)[2] = 4.0;
    f.span(kAtom_y)[2] = 4.0;
    f.span(kAtom_z)[2] = 4.0;
    f.span(kAtom_mass)[2] = 40.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_tag)[2] = 3;

    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_vx)[i] = 0.0;
        f.span(kAtom_vy)[i] = 0.0;
        f.span(kAtom_vz)[i] = 0.0;
        f.span(kAtom_fx)[i] = 0.0;
        f.span(kAtom_fy)[i] = 0.0;
        f.span(kAtom_fz)[i] = 0.0;
    }
    f.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0); // A = 100 Å²
    return f;
}

} // namespace

TEST(PistonElectrode, PressureIdentityUpper) {
    auto f = upper_wall_pair();
    // Unequal forces: 1.0 + 3.0 = 4.0 eV/Å → P = 4/100 = 0.04 eV/Å³
    f.span(kAtom_fz)[0] = 1.0;
    f.span(kAtom_fz)[1] = 3.0;
    f.span(kAtom_fz)[2] = 99.0; // fluid ignored

    cpu::md::PistonElectrode piston(
        /*type=*/3, /*P=*/0.0, cpu::md::PistonElectrode::Side::Upper);
    cpu::Context ctx;
    piston.build({ctx, f});
    piston.launch({ctx, f, ATV::slot_of(md::Phase::post_force), 0});

    EXPECT_NEAR(piston.last_area(), 100.0, 1e-14);
    EXPECT_NEAR(piston.last_F_z_atoms(), 4.0, 1e-14);
    EXPECT_NEAR(piston.last_P_z(), 0.04, 1e-14);
    // P_target=0 ⇒ F_net = F_atoms
    EXPECT_NEAR(piston.last_F_net(), 4.0, 1e-14);
}

TEST(PistonElectrode, ExternalForceBalance) {
    auto f = upper_wall_pair();
    f.span(kAtom_fz)[0] = 2.0;
    f.span(kAtom_fz)[1] = 2.0;    // F_atoms = 4, A=100 → P_atoms=0.04
    const double P_target = 0.04; // eV/Å³

    cpu::md::PistonElectrode piston(3, P_target,
                                    cpu::md::PistonElectrode::Side::Upper);
    cpu::Context ctx;
    piston.build({ctx, f});
    piston.launch({ctx, f, ATV::slot_of(md::Phase::post_force), 0});

    // F_ext = -P*A = -4; F_net = 4 + (-4) = 0
    EXPECT_NEAR(piston.last_F_net(), 0.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[0]), 0.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[1]), 0.0, 1e-12);
    // Fluid force untouched
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[2]), 0.0, 1e-14);
}

TEST(PistonElectrode, MassWeightedShareAndZeroXY) {
    auto f = upper_wall_pair();
    f.span(kAtom_fz)[0] = 10.0;
    f.span(kAtom_fz)[1] = 0.0;
    f.span(kAtom_fx)[0] = 5.0;
    f.span(kAtom_fy)[1] = 7.0;
    // P_target=0, F_net=10, M=12+36=48 → shares 10*(12/48)=2.5 and 7.5
    cpu::md::PistonElectrode piston(3, 0.0,
                                    cpu::md::PistonElectrode::Side::Upper);
    cpu::Context ctx;
    piston.build({ctx, f});
    piston.launch({ctx, f, ATV::slot_of(md::Phase::post_force), 0});

    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[0]), 2.5, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[1]), 7.5, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fx)[0]), 0.0, 1e-14);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fy)[1]), 0.0, 1e-14);
}

TEST(PistonElectrode, LowerSideSign) {
    auto f = upper_wall_pair();
    // Reuse geometry as "lower" wall with upward atom force F=4
    f.span(kAtom_fz)[0] = 2.0;
    f.span(kAtom_fz)[1] = 2.0;
    // Lower: P_z = -F_atoms/A = -0.04; balance needs P_target so F_ext=+P*A
    // cancels when F_atoms = -P*A ... here F_atoms > 0 so not balanced with +P.
    // For lower wall at equilibrium fluid pushes down (F_atoms < 0).
    f.span(kAtom_fz)[0] = -2.0;
    f.span(kAtom_fz)[1] = -2.0; // F_atoms=-4, A=100 → P_z = -(-4)/100=0.04
    const double P_target = 0.04;
    cpu::md::PistonElectrode piston(3, P_target,
                                    cpu::md::PistonElectrode::Side::Lower);
    cpu::Context ctx;
    piston.build({ctx, f});
    piston.launch({ctx, f, ATV::slot_of(md::Phase::post_force), 0});
    EXPECT_NEAR(piston.last_P_z(), 0.04, 1e-14);
    // F_ext = +P*A = +4; F_net = -4+4 = 0
    EXPECT_NEAR(piston.last_F_net(), 0.0, 1e-12);
}

TEST(PistonElectrode, RigidEqualizeVz) {
    auto f = upper_wall_pair();
    f.span(kAtom_vz)[0] = 0.1;
    f.span(kAtom_vz)[1] = 0.5;
    f.span(kAtom_vx)[0] = 0.2;
    f.span(kAtom_vy)[1] = -0.3;
    // mass-weighted COM vz = (12*0.1 + 36*0.5)/48 = (1.2+18)/48 = 0.4
    cpu::md::PistonElectrode piston(3, 0.0,
                                    cpu::md::PistonElectrode::Side::Upper);
    cpu::Context ctx;
    piston.build({ctx, f});
    piston.launch({ctx, f, ATV::slot_of(md::Phase::final_integrate), 0});

    EXPECT_NEAR(static_cast<double>(f.span(kAtom_vz)[0]), 0.4, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_vz)[1]), 0.4, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_vx)[0]), 0.0, 1e-14);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_vy)[1]), 0.0, 1e-14);
    // Fluid untouched
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_vz)[2]), 0.0, 1e-14);
}
