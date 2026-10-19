#include "atomiverse/cpu/linalg/pulay.h"
#include <gtest/gtest.h>
#include <limits>

using namespace ATV::cpu::linalg;

TEST(Pulay, MinimizesResidualWithUnitSum) {
    const RealMatrix gram = {{4.0, 0.0}, {0.0, 1.0}};
    const auto c = pulay_coefficients(gram);
    ASSERT_EQ(c.size(), 2u);
    EXPECT_NEAR(c(0), 0.2, 1e-14);
    EXPECT_NEAR(c(1), 0.8, 1e-14);
    EXPECT_NEAR(c(0) + c(1), 1.0, 1e-14);
    const auto single = pulay_coefficients(RealMatrix{{0.0}});
    EXPECT_EQ(single(0), 1.0);
}

TEST(Pulay, RejectsInvalidGramAndSingularHistory) {
    EXPECT_THROW(pulay_coefficients(RealMatrix(xt::zeros<double>({0, 0}))),
                 std::invalid_argument);
    EXPECT_THROW(pulay_coefficients(RealMatrix{{1.0, 2.0}}),
                 std::invalid_argument);
    EXPECT_THROW(pulay_coefficients(
                     RealMatrix{{std::numeric_limits<double>::quiet_NaN()}}),
                 std::invalid_argument);
    EXPECT_THROW(pulay_coefficients(RealMatrix{{1.0, 1.0}, {1.0, 1.0}}),
                 std::runtime_error);
}

TEST(Pulay, CommutesInTheOrthonormalBasis) {
    const RealMatrix F = {{2.0, 1.0}, {1.0, 3.0}};
    const RealMatrix P = {{1.0, 0.0}, {0.0, 0.0}};
    const RealMatrix S = {{1.0, 0.0}, {0.0, 1.0}};
    const RealMatrix X = {{2.0, 0.0}, {0.0, 3.0}};
    const auto error = diis_error(F, P, S, X);
    EXPECT_EQ(error(0, 0), 0.0);
    EXPECT_EQ(error(0, 1), -6.0);
    EXPECT_EQ(error(1, 0), 6.0);
    EXPECT_EQ(error(1, 1), 0.0);
}
