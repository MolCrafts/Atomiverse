/// @file tests/longrange/test_gaussian_extent.cpp
/// @brief Unit tests for WJGH Gaussian extent + well-separated predicate.

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/gaussian_extent.h"

using namespace ATV;
using ATV::longrange::GaussianExtent;
using ATV::longrange::detail::erfc_inv;

TEST(GaussianExtent, ErfcInvRoundTrip) {
    for (AccReal y :
         {1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9}) {
        if (y >= AccReal(2) || y <= AccReal(0))
            continue;
        const AccReal x = erfc_inv(y);
        EXPECT_NEAR(std::erfc(x), y, 1e-12) << "y=" << y;
    }
}

TEST(GaussianExtent, RextMatchesHandFormula) {
    // Hand values: r_ext = √(2/ζ) · erfc⁻¹(ε).
    const AccReal zeta = AccReal(0.5);
    const AccReal eps = AccReal(1e-9);
    const AccReal want = std::sqrt(AccReal(2) / zeta) * erfc_inv(eps);
    GaussianExtent g(zeta, eps);
    EXPECT_NEAR(g.r_ext(), want, 1e-12);
    EXPECT_NEAR(g.zeta(), zeta, 0.0);
    EXPECT_NEAR(g.threshold(), eps, 0.0);
}

TEST(GaussianExtent, LargerZetaSmallerExtent) {
    GaussianExtent tight(2.0, 1e-8);
    GaussianExtent diffuse(0.2, 1e-8);
    EXPECT_LT(tight.r_ext(), diffuse.r_ext());
}

TEST(GaussianExtent, RejectsBadArgs) {
    EXPECT_THROW(GaussianExtent(0.0, 1e-8), std::invalid_argument);
    EXPECT_THROW(GaussianExtent(-1.0, 1e-8), std::invalid_argument);
    EXPECT_THROW(GaussianExtent(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianExtent(1.0, -1e-8), std::invalid_argument);
    EXPECT_THROW(GaussianExtent(1.0, 1.5), std::invalid_argument);
}

TEST(GaussianExtent, WellSeparatedTruthTable) {
    // Two equal extents r=1; centres along x.
    GaussianExtent a(2.0, 1e-3); // r_ext known from formula
    GaussianExtent b(2.0, 1e-3);
    const AccReal re = a.r_ext();
    EXPECT_GT(re, AccReal(0));

    // Exactly at contact → NOT well-separated (strict inequality).
    EXPECT_FALSE(GaussianExtent::well_separated(a, Vec3{0, 0, 0}, b,
                                                Vec3{2 * re, 0, 0}));

    // Just inside → near field.
    EXPECT_FALSE(GaussianExtent::well_separated(
        a, Vec3{0, 0, 0}, b, Vec3{2 * re - AccReal(1e-6), 0, 0}));

    // Just outside → far field.
    EXPECT_TRUE(GaussianExtent::well_separated(
        a, Vec3{0, 0, 0}, b, Vec3{2 * re + AccReal(1e-6), 0, 0}));

    // Coincident centres → near.
    EXPECT_FALSE(
        GaussianExtent::well_separated(a, Vec3{1, 2, 3}, b, Vec3{1, 2, 3}));
}
