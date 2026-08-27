/// @file tests/cpu/dynamics/test_charge_updater_probe.cpp
/// @brief Unit tests for the `Component::updates_owned_charges()` probe.
///
/// Every constant-potential charge solver must declare the mid-step charge
/// rewrite (`updates_owned_charges()`) and post `ChargePacker` after writing
/// owned q so ghost rows see the new charges under MPI DD.

#include <gtest/gtest.h>

#include <vector>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/const_pot_cg.h"
#include "atomiverse/cpu/dynamics/const_pot_maze.h"
#include "atomiverse/cpu/dynamics/const_potential.h"
#include "atomiverse/cpu/dynamics/dipole_electrode_joint.h"
#include "atomiverse/cpu/dynamics/qeq.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

using namespace ATV;

TEST(ChargeUpdaterProbe, ConstPotBODeclares) {
    cpu::md::ConstPotBO c(2.0);
    EXPECT_TRUE(c.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, ConstPotMaZeDeclares) {
    cpu::md::ConstPotMaZe c(2.0);
    EXPECT_TRUE(c.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, ConstPotCGDeclares) {
    cpu::md::ConstPotCG c(2.0);
    EXPECT_TRUE(c.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, ConstPotentialBODeclares) {
    cpu::ConstPotentialBO c({1, 2}, {0.0, 1.0});
    EXPECT_TRUE(c.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, DipoleElectrodeJointDeclares) {
    cpu::md::DipoleElectrodeJoint c(2.0, std::vector<double>{0.0, 0.0, 1.0});
    EXPECT_TRUE(c.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, QEqDeclares) {
    cpu::md::QEq qeq(10.0);
    EXPECT_TRUE(qeq.updates_owned_charges());
}

TEST(ChargeUpdaterProbe, PlainComponentDefaultsFalse) {
    cpu::md::VelocityVerlet vv{1.0};
    EXPECT_FALSE(vv.updates_owned_charges());
}
