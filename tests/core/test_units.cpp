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

TEST(Units, ThermalDeBroglieUsesTheLjRow) {
    // lj: h = k = mvv2e = 1, so Λ = 1/sqrt(2π m T).
    const double lambda = units::lambda_thermal(1.008, 300.0);
    const double expect = 1.0 / std::sqrt(2.0 * units::pi * 1.008 * 300.0);
    EXPECT_NEAR(lambda, expect, 1e-12);
    // Heavier + hotter shrinks it: Λ ~ 1/sqrt(mT).
    EXPECT_NEAR(units::lambda_thermal(4.032, 300.0), lambda / 2.0, 1e-12);
    EXPECT_NEAR(units::lambda_thermal(1.008, 1200.0), lambda / 2.0, 1e-12);
}
