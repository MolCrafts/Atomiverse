#include "atomiverse/core/gto.h"
#include "core/gto_cases.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

TEST(GaussianMath, BoysIntegralAcrossCancellationAndRecurrenceBoundary) {
    for (const auto& c : gto_test::cases) {
        EXPECT_NEAR(ATV::boys_fn(c.order, c.argument), c.expected,
                    2e-14 * std::abs(c.expected) + 1e-300)
            << "order=" << c.order << " x=" << c.argument;
    }
}

TEST(GaussianMath, BoysSequenceMatchesIndependentIntegralPins) {
    // Check each pin in sequences ending at that order and at higher orders.
    // In particular F_0 must remain accurate when seeded from a higher order.
    for (const auto& c : gto_test::cases) {
        for (int extra : {0, 1, 6}) {
            std::array<double, 32> values{};
            ASSERT_LT(c.order + extra, values.size());
            ATV::boys_sequence(c.order + extra, c.argument, values.data());
            EXPECT_NEAR(values[c.order], c.expected,
                        2e-14 * std::abs(c.expected) + 1e-300)
                << "order=" << c.order << " x=" << c.argument
                << "max_order=" << c.order + extra;
        }
    }
}

TEST(GaussianMath, BoysSequenceFirstOrderAtTinyNonzeroArgument) {
    std::array<double, 2> values{};
    ATV::boys_sequence(1, 1e-20, values.data());
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 1.0 / 3.0);
}
