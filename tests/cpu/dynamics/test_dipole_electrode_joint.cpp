/// @file tests/cpu/dynamics/test_dipole_electrode_joint.cpp
/// @brief Unit tests for cpu::md::DipoleElectrodeJoint (E5 joint Q–μ).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/dipole_electrode_joint.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Minimal parallel-plate + two electrolyte ions (spec geometry, R=0).
cpu::Frame toy_cell() {
    cpu::Frame f(4);
    const double z[4] = {2.0, 18.0, 8.0, 12.0};
    const double q0[4] = {0.0, 0.0, 1.0, -1.0};
    const int typ[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 12.0 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

TEST(DipoleElectrodeJoint, AlphaZeroMatchesConstPotBO) {
    auto f_joint = toy_cell();
    auto f_bo = toy_cell();

    cpu::md::DipoleElectrodeJoint joint(
        /*eta=*/2.0, /*alpha_by_type=*/{0.0, 0.0, 0.0, 0.0, 0.0});
    joint.add_electrode(1, 0.0);
    joint.add_electrode(2, 1.0);
    joint.electroneutrality = true;
    joint.xy_image_range = 0;
    joint.joint_tol = 1e-12;
    joint.pol_tol = 1e-12;

    cpu::md::ConstPotBO bo(2.0);
    bo.add_electrode(1, 0.0);
    bo.add_electrode(2, 1.0);
    bo.electroneutrality = true;
    bo.xy_image_range = 0;

    cpu::Context ctx;
    joint.build({ctx, f_joint});
    joint.launch({ctx, f_joint, 0, 0});
    bo.build({ctx, f_bo});
    bo.launch({ctx, f_bo, 0, 0});

    EXPECT_TRUE(joint.last_joint_converged());
    EXPECT_NEAR(joint.total_charge(0), bo.total_charge(0), 1e-10);
    EXPECT_NEAR(joint.total_charge(1), bo.total_charge(1), 1e-10);
    EXPECT_LE(joint.neutrality_residual(), 1e-10);
    // All dipoles zero
    for (double m : joint.last_mu())
        EXPECT_NEAR(m, 0.0, 1e-14);
}

TEST(DipoleElectrodeJoint, PolarizableCouplesAndConverges) {
    auto f = toy_cell();
    cpu::md::DipoleElectrodeJoint joint(2.0,
                                        /*alpha=*/{0.0, 0.0, 0.0, 1.0, 1.5});
    joint.add_electrode(1, 0.0);
    joint.add_electrode(2, 1.0);
    joint.thole_a = 0.39;
    joint.electroneutrality = true;
    joint.xy_image_range = 0;
    joint.joint_tol = 1e-12;
    joint.pol_tol = 1e-12;
    joint.joint_max_iter = 50;

    cpu::Context ctx;
    joint.build({ctx, f});
    joint.launch({ctx, f, 0, 0});

    EXPECT_TRUE(joint.last_joint_converged());
    EXPECT_GT(joint.last_joint_iters(), 1);
    EXPECT_LE(joint.neutrality_residual(), 1e-10);
    EXPECT_NEAR(joint.total_charge(0) + joint.total_charge(1), 0.0, 1e-10);
    // Frame write-back of electrode Q
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]),
                joint.total_charge(0), 1e-14);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[1]),
                joint.total_charge(1), 1e-14);
    // Electrolyte charges preserved
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[2]), 1.0, 1e-14);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[3]), -1.0, 1e-14);
    // Non-zero μ_z on polarizable ions; transverse ~0 by symmetry
    const auto& mu = joint.last_mu();
    ASSERT_EQ(mu.size(), 12u);
    EXPECT_NEAR(mu[3 * 2 + 0], 0.0, 1e-10);
    EXPECT_NEAR(mu[3 * 2 + 1], 0.0, 1e-10);
    EXPECT_GT(std::fabs(mu[3 * 2 + 2]), 1e-4);
    EXPECT_NEAR(mu[3 * 3 + 0], 0.0, 1e-10);
    EXPECT_NEAR(mu[3 * 3 + 1], 0.0, 1e-10);
    EXPECT_GT(std::fabs(mu[3 * 3 + 2]), 1e-4);
}

TEST(DipoleElectrodeJoint, CouplingShiftsQRelativeToAlphaZero) {
    auto f0 = toy_cell();
    auto f1 = toy_cell();

    cpu::md::DipoleElectrodeJoint j0(2.0, {0.0, 0.0, 0.0, 0.0, 0.0});
    j0.add_electrode(1, 0.0).add_electrode(2, 1.0);
    j0.electroneutrality = true;
    j0.xy_image_range = 0;
    j0.joint_tol = 1e-12;

    cpu::md::DipoleElectrodeJoint j1(2.0, {0.0, 0.0, 0.0, 1.0, 1.5});
    j1.add_electrode(1, 0.0).add_electrode(2, 1.0);
    j1.thole_a = 0.39;
    j1.electroneutrality = true;
    j1.xy_image_range = 0;
    j1.joint_tol = 1e-12;
    j1.pol_tol = 1e-12;

    cpu::Context ctx;
    j0.build({ctx, f0});
    j0.launch({ctx, f0, 0, 0});
    j1.build({ctx, f1});
    j1.launch({ctx, f1, 0, 0});

    const double q0 = j0.total_charge(1);
    const double q1 = j1.total_charge(1);
    ASSERT_GT(std::fabs(q0), 1e-12);
    const double rel = std::fabs(q1 - q0) / std::fabs(q0);
    EXPECT_GE(rel, 1e-6);
}

TEST(DipoleElectrodeJoint, RequiresElectrode) {
    EXPECT_THROW(
        {
            cpu::md::DipoleElectrodeJoint j(2.0, {0.0, 1.0});
            cpu::Frame f = toy_cell();
            cpu::Context ctx;
            j.build({ctx, f});
        },
        std::runtime_error);
}

TEST(DipoleElectrodeJoint, RejectsBadEta) {
    EXPECT_THROW(cpu::md::DipoleElectrodeJoint(-1.0, {0.0}),
                 std::invalid_argument);
}
