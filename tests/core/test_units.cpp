/// @file test_units.cpp
/// @brief Unit tests for new unit conversion constants.

#include <gtest/gtest.h>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"

using namespace ATV;

TEST(Units, AngstromToBohr) {
    const AccReal bohr = units::convert(AccReal(5.43), units::Length::Angstrom,
                                        units::Length::Bohr);
    EXPECT_NEAR(bohr, 5.43 * units::kAngstromInBohr, 1e-12);
}

TEST(Units, RoundtripAngstromBohr) {
    const AccReal a = 5.43;
    const AccReal b =
        units::convert(a, units::Length::Angstrom, units::Length::Bohr);
    const AccReal back =
        units::convert(b, units::Length::Bohr, units::Length::Angstrom);
    EXPECT_NEAR(back, a, 1e-12);
}

TEST(Units, ThermalDeBroglieHydrogenAt300K) {
    // Λ = h/sqrt(2π m kB T): H (1.008 amu) at 300 K → 1.0037 Å
    // (h = 4.135667696e-15 eV·s, kB = 8.617333262e-5 eV/K, CODATA 2018).
    const double lambda = units::lambda_thermal(1.008, 300.0);
    EXPECT_NEAR(lambda, 1.0038, 2e-3);
    // Heavier + hotter shrinks it: Λ ~ 1/sqrt(mT).
    EXPECT_NEAR(units::lambda_thermal(4.032, 300.0), lambda / 2.0, 1e-12);
    EXPECT_NEAR(units::lambda_thermal(1.008, 1200.0), lambda / 2.0, 1e-12);
}
