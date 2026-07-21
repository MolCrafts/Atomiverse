/// @file tests/cpu/md/test_write_sets.cpp
/// @brief Kind / stress-protocol pins after Capability was deleted.

#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/thermostat.h"
#include "atomiverse/cpu/md/deform.h"
#include "atomiverse/cpu/md/fix_csvr.h"
#include "atomiverse/cpu/md/fix_langevin.h"
#include "atomiverse/cpu/md/fix_rattle.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/md/nve_limit.h"
#include "atomiverse/cpu/md/velocity_verlet.h"

using namespace ATV;

namespace {
cpu::md::NoseHooverNPT::Config npt_cfg() {
    return {/*T_target=*/300.0, /*tau_T=*/100.0, /*P_target=*/0.0,
            /*tau_P=*/1000.0, /*dt=*/1.0};
}
} // namespace

TEST(WriteSets, IntegratorsAreIntegratorKind) {
    static_assert(std::is_base_of_v<cpu::Integrator, cpu::md::VelocityVerlet>);
    cpu::md::VelocityVerlet vv{1.0};
    cpu::md::NVELimit nve{1.0, 0.1};
    cpu::md::NoseHooverNPT npt{npt_cfg()};
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&vv), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&nve), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&npt), nullptr);
    EXPECT_EQ(vv.role(), 0);
}

TEST(WriteSets, ThermostatsAreThermostatKind) {
    cpu::md::NoseHoover nh{300.0, 100.0, 1.0};
    cpu::md::FixCSVR csvr{300.0, 100.0, 1.0, 7u};
    EXPECT_NE(dynamic_cast<const cpu::Thermostat*>(&nh), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Thermostat*>(&csvr), nullptr);
}

TEST(WriteSets, NPTRequiresSystemStress) {
    cpu::md::NoseHooverNPT npt{npt_cfg()};
    EXPECT_TRUE(npt.requires_system_stress());
}

TEST(WriteSets, RattleIsConstraintProvider) {
    cpu::md::FixRattle r{1.0};
    r.constrain_bond(0, 0.9572);
    EXPECT_NE(r.as_constraint(), nullptr);
}

TEST(WriteSets, InitDeformLangevinConstruct) {
    EXPECT_NO_THROW((cpu::md::InitVelocity{300.0, 12345u}));
    EXPECT_NO_THROW((cpu::md::Deform{cpu::md::Deform::Mode::erate, 1e-4, 1.0}));
    EXPECT_NO_THROW((cpu::md::FixLangevin{300.0, 300.0, 100.0, 1.0, 42u}));
}
