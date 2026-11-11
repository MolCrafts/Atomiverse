/// @file tests/core/test_thermo_style.cpp
/// @brief `thermo_style custom` parsing, demand and formatting.

#include <cstdio>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/thermo_style.h"
#include "atomiverse/core/units.h"

using namespace ATV;

TEST(ThermoStyle, DefaultIsLammpsStepTempPeKeEtotal) {
    const ThermoStyle style;
    ASSERT_EQ(style.columns().size(), 5u);
    EXPECT_EQ(style.columns()[0], ThermoKeyword::Step);
    EXPECT_TRUE(style.needs_energy());
    EXPECT_FALSE(style.needs_virial());
    EXPECT_NE(style.header().find("PotEng"), std::string::npos);
}

TEST(ThermoStyle, DemandFollowsKeywords) {
    EXPECT_FALSE(ThermoStyle({"step", "temp", "ke"}).needs_energy());
    EXPECT_FALSE(ThermoStyle({"step", "temp", "ke"}).needs_virial());
    EXPECT_TRUE(ThermoStyle({"step", "econserve"}).needs_energy());
    EXPECT_TRUE(ThermoStyle({"step", "pxy"}).needs_virial());
    EXPECT_FALSE(ThermoStyle({"step", "press"}).needs_energy());
    EXPECT_TRUE(ThermoStyle({"step", "press"}).needs_virial());
}

TEST(ThermoStyle, RejectsUnknownRepeatedAndEmptyKeywords) {
    EXPECT_THROW(ThermoStyle({"step", "epair"}), std::invalid_argument);
    EXPECT_THROW(ThermoStyle({"pe", "pe"}), std::invalid_argument);
    EXPECT_THROW(ThermoStyle(std::vector<std::string>{}),
                 std::invalid_argument);
}

TEST(ThermoStyle, PressureIsVirialPlusKineticOverVolumeTimesNktv2p) {
    // LAMMPS compute pressure: P = (K + W) / V · nktv2p, in the style's own
    // pressure unit (lj: reduced; metal: bar; real: atm).
    const ThermoStyle style({"press", "pxx", "vol"});
    for (const char* name : {"lj", "metal", "real"}) {
        const auto u = units::from_style(name);
        ThermoSample s;
        s.volume = 1000.0;
        s.virial[0] = 1.0; // energy
        s.kinetic[0] = 2.0;
        s.boltz = u.boltz;
        s.nktv2p = u.nktv2p;
        const double pxx = 3.0 / 1000.0 * u.nktv2p;
        const std::string line = style.line(s);
        double press = 0, xx = 0, vol = 0;
        ASSERT_EQ(std::sscanf(line.c_str(), "%lf %lf %lf", &press, &xx, &vol),
                  3);
        EXPECT_NEAR(xx, pxx, 1e-6 * pxx) << name;
        EXPECT_NEAR(press, pxx / 3.0, 1e-6 * pxx) << name;
        EXPECT_DOUBLE_EQ(vol, 1000.0);
    }
}

TEST(ThermoStyle, TemperatureUsesDegreesOfFreedom) {
    const ThermoStyle style({"temp"});
    ThermoSample s;
    const auto real = units::from_style("real");
    s.ke = 0.5 * 3.0 * real.boltz * 300.0;
    s.boltz = real.boltz;
    s.ndof = 3;
    double temp = 0;
    ASSERT_EQ(std::sscanf(style.line(s).c_str(), "%lf", &temp), 1);
    EXPECT_NEAR(temp, 300.0, 1e-4);
}
