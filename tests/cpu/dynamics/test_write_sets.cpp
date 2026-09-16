/// @file tests/cpu/md/test_write_sets.cpp
/// @brief Kind / stress-protocol pins after Capability was deleted.

#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/thermostat.h"
#include "atomiverse/cpu/dynamics/deform.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/nph_integrator.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

using namespace ATV;

TEST(WriteSets, IntegratorsAreIntegratorKind) {
    static_assert(std::is_base_of_v<cpu::Integrator, cpu::md::VelocityVerlet>);
    cpu::md::VelocityVerlet vv{1.0};
    cpu::md::NVELimit nve{1.0, 0.1};
    cpu::md::NoseHooverNPT npt{300.0, 100.0, 0.0, 1000.0, 1.0};
    cpu::md::ParrinelloRahmanNPT pr{300.0, 0.0, 100.0, 1000.0, 1.0};
    cpu::md::NphIntegrator nph{0.0, 1000.0, 1.0};
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&vv), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&nve), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&npt), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&pr), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Integrator*>(&nph), nullptr);
    EXPECT_EQ(vv.role(), 0);
}

TEST(WriteSets, ThermostatsAreThermostatKind) {
    cpu::md::NoseHoover nh{300.0, 100.0, 1.0};
    cpu::md::FixCSVR csvr{300.0, 100.0, 1.0, 7u};
    EXPECT_NE(dynamic_cast<const cpu::Thermostat*>(&nh), nullptr);
    EXPECT_NE(dynamic_cast<const cpu::Thermostat*>(&csvr), nullptr);
}

TEST(WriteSets, NPTRequiresSystemStress) {
    cpu::md::NoseHooverNPT npt{300.0, 100.0, 0.0, 1000.0, 1.0};
    EXPECT_TRUE(npt.requires_system_stress());
    cpu::md::ParrinelloRahmanNPT pr{300.0, 0.0, 100.0, 1000.0, 1.0};
    EXPECT_TRUE(pr.requires_system_stress());
    cpu::md::NphIntegrator nph{0.0, 1000.0, 1.0};
    EXPECT_TRUE(nph.requires_system_stress());
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
