/// @file tests/core/test_unit_system.cpp
/// @brief LAMMPS `Update::set_units` rows. The classical engine is lj.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/unit_system.h"
#include "atomiverse/core/units.h"

using namespace ATV;

TEST(UnitSystem, LjRowIsTheEngineCoupling) {
    const auto lj = units::UnitSystem::lj();
    EXPECT_STREQ(lj.style, "lj");
    EXPECT_DOUBLE_EQ(lj.boltz, 1.0);
    EXPECT_DOUBLE_EQ(lj.mvv2e, 1.0);
    EXPECT_DOUBLE_EQ(lj.ftm2v, 1.0);
    EXPECT_DOUBLE_EQ(lj.qqr2e, 1.0);
    EXPECT_DOUBLE_EQ(lj.nktv2p, 1.0);
    EXPECT_DOUBLE_EQ(lj.angstrom, 1.0);
    EXPECT_DOUBLE_EQ(lj.femtosecond, 1.0);
    EXPECT_DOUBLE_EQ(lj.qelectron, 1.0);
    EXPECT_DOUBLE_EQ(lj.qqrd2e(), 1.0);
    EXPECT_DOUBLE_EQ(lj.two_charge_force(), 1.0);
    EXPECT_DOUBLE_EQ(lj.dt, 0.005);
    EXPECT_DOUBLE_EQ(lj.skin, 0.3);
    EXPECT_DOUBLE_EQ(units::md_qqr2e, lj.qqr2e);
    EXPECT_DOUBLE_EQ(units::md_mvv2e, lj.mvv2e);
    EXPECT_DOUBLE_EQ(units::md_boltz, lj.boltz);
    EXPECT_DOUBLE_EQ(units::md_ftm2v, lj.ftm2v);
}

TEST(UnitSystem, FromStyleMatchesLammpsUpdate) {
    const auto metal = units::from_style("metal");
    EXPECT_DOUBLE_EQ(metal.qqr2e, 14.399645);
    EXPECT_DOUBLE_EQ(metal.boltz, 8.617343e-5);
    EXPECT_DOUBLE_EQ(metal.mvv2e, 1.0364269e-4);
    EXPECT_DOUBLE_EQ(metal.femtosecond, 1.0e-3);
    EXPECT_DOUBLE_EQ(metal.qelectron, 1.0);
    EXPECT_DOUBLE_EQ(metal.angstrom, 1.0);
    EXPECT_DOUBLE_EQ(metal.two_charge_force(), metal.qqr2e);
    EXPECT_DOUBLE_EQ(metal.qqrd2e(), metal.qqr2e);

    const auto real = units::from_style("real");
    EXPECT_DOUBLE_EQ(real.qqr2e, 332.06371);
    EXPECT_DOUBLE_EQ(real.boltz, 0.0019872067);
    EXPECT_DOUBLE_EQ(real.mvv2e, 48.88821291 * 48.88821291);
    EXPECT_DOUBLE_EQ(real.ftm2v, 1.0 / 48.88821291 / 48.88821291);
    EXPECT_DOUBLE_EQ(real.nktv2p, 68568.415);
    EXPECT_DOUBLE_EQ(real.two_charge_force(), real.qqr2e);

    const auto cgs = units::from_style("cgs");
    EXPECT_DOUBLE_EQ(cgs.qelectron, 4.8032044e-10);
    EXPECT_DOUBLE_EQ(cgs.angstrom, 1.0e-8);
    EXPECT_NEAR(cgs.two_charge_force(),
                cgs.qqr2e * cgs.qelectron * cgs.qelectron
                    / (cgs.angstrom * cgs.angstrom),
                0.0);

    const auto si = units::from_style("si");
    EXPECT_DOUBLE_EQ(si.qqr2e, 8.9876e9);
    EXPECT_DOUBLE_EQ(si.qelectron, 1.6021765e-19);
    const auto electron = units::from_style("electron");
    EXPECT_DOUBLE_EQ(electron.angstrom, 1.88972612);
    EXPECT_DOUBLE_EQ(electron.mvv2e, 1.06657236);
    const auto micro = units::from_style("micro");
    EXPECT_DOUBLE_EQ(micro.qqr2e, 8.987556e6);
    const auto nano = units::from_style("nano");
    EXPECT_DOUBLE_EQ(nano.qqr2e, 230.7078669);
    EXPECT_DOUBLE_EQ(nano.dt, 0.00045);

    EXPECT_THROW(units::from_style("nope"), std::invalid_argument);
}

TEST(UnitSystem, SetStyleAcceptsOnlyLj) {
    units::set_style("lj");
    EXPECT_STREQ(units::active().style, "lj");
    EXPECT_DOUBLE_EQ(units::active().qqr2e, 1.0);
    EXPECT_THROW(units::set_style("metal"), std::invalid_argument);
    EXPECT_THROW(units::set_style("nope"), std::invalid_argument);
    EXPECT_STREQ(units::active().style, "lj");
    EXPECT_DOUBLE_EQ(units::coulomb, 14.399645);
}
