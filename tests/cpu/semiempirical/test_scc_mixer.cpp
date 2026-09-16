/// @file tests/cpu/semiempirical/test_scc_mixer.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

using namespace ATV;
using ATV::cpu::semiempirical::internal::ChargeMixer;

namespace {

/// A linear charge map with an eigenvalue below −1, i.e. exactly the shape
/// that makes undamped Picard a period-2 limit cycle:
///   g(x) = A x + b,  A = diag(−2, 1/2) in the (x0, x1) basis.
/// Fixed point: x* = (I − A)^-1 b = (b0/3, 2 b1).
std::vector<double> map_g(const std::vector<double>& x) {
    return {-2.0 * x[0] + 0.3, 0.5 * x[1] + 0.4};
}

double residual_norm(const std::vector<double>& a,
                     const std::vector<double>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

} // namespace

TEST(ChargeMixer, FirstIterateIsPlainLinearDamping) {
    ChargeMixer mixer(0.25, 6);
    const std::vector<double> x_in{1.0, -2.0};
    const std::vector<double> x_out{3.0, 2.0};
    const std::vector<double> got = mixer.next(x_in, x_out);
    ASSERT_EQ(got.size(), 2u);
    EXPECT_NEAR(got[0], 1.0 + 0.25 * (3.0 - 1.0), 1e-15);
    EXPECT_NEAR(got[1], -2.0 + 0.25 * (2.0 - (-2.0)), 1e-15);
}

TEST(ChargeMixer, ConvergesAMapPlainPicardCannot) {
    // Undamped Picard on this map oscillates forever (|λ| = 2 > 1); confirm
    // that first, then that the mixer reaches the fixed point anyway.
    std::vector<double> picard{0.0, 0.0};
    for (int it = 0; it < 40; ++it)
        picard = map_g(picard);
    EXPECT_GT(residual_norm(picard, map_g(picard)), 1.0);

    ChargeMixer mixer(0.3, 6);
    std::vector<double> x{0.0, 0.0};
    double residual = 1.0;
    int iterations = 0;
    for (int it = 0; it < 40; ++it) {
        const std::vector<double> out = map_g(x);
        residual = residual_norm(out, x);
        iterations = it + 1;
        if (residual < 1e-13)
            break;
        x = mixer.next(x, out);
    }
    EXPECT_LT(residual, 1e-13);
    EXPECT_LE(iterations, 20);
    EXPECT_NEAR(x[0], 0.3 / 3.0, 1e-12);
    EXPECT_NEAR(x[1], 0.8, 1e-12);
}

TEST(ChargeMixer, RejectsBetaOutsideUnitInterval) {
    EXPECT_THROW(ChargeMixer(0.0, 4), std::invalid_argument);
    EXPECT_THROW(ChargeMixer(-0.1, 4), std::invalid_argument);
    EXPECT_THROW(ChargeMixer(1.5, 4), std::invalid_argument);
}

TEST(ChargeMixer, RejectsNonPositiveDepth) {
    EXPECT_THROW(ChargeMixer(0.3, 0), std::invalid_argument);
}

TEST(ChargeMixer, RejectsMismatchedOrEmptyVectors) {
    ChargeMixer mixer(0.3, 4);
    EXPECT_THROW(mixer.next({1.0, 2.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(mixer.next({}, {}), std::invalid_argument);
}
