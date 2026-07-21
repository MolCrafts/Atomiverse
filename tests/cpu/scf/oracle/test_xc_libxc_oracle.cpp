/// @file tests/cpu/scf/oracle/test_xc_libxc_oracle.cpp
/// @brief Smoke + textbook-value tests for the libxc test oracle.
///
/// These tests run only when -DATV_BUILD_XC_ORACLE=ON; the entire
/// directory is gated at the CMake level so this TU is otherwise not
/// compiled.

#include "xc_libxc_oracle.h"

#include <gtest/gtest.h>

#include <cmath>

using ATV::test::xc_oracle::evaluate_lda_x;
using ATV::test::xc_oracle::evaluate_lsda;
using ATV::test::xc_oracle::evaluate_pbe;

TEST(XCLibxcOracle, LsdaUnpolarizedFiniteAndNegativeExc) {
    auto out = evaluate_lsda(0.5, 0.5);
    EXPECT_TRUE(std::isfinite(out.exc));
    EXPECT_TRUE(std::isfinite(out.vrho_alpha));
    EXPECT_TRUE(std::isfinite(out.vrho_beta));
    EXPECT_LT(out.exc, 0.0);
    EXPECT_DOUBLE_EQ(out.vsigma_aa, 0.0);
    EXPECT_DOUBLE_EQ(out.vsigma_ab, 0.0);
    EXPECT_DOUBLE_EQ(out.vsigma_bb, 0.0);
}

TEST(XCLibxcOracle, LsdaSpinPolarizedHasDistinctVrho) {
    auto out = evaluate_lsda(0.8, 0.2);
    EXPECT_TRUE(std::isfinite(out.exc));
    EXPECT_NE(out.vrho_alpha, out.vrho_beta);
}

TEST(XCLibxcOracle, LsdaBelowCutoffReturnsZero) {
    auto out = evaluate_lsda(1e-20, 1e-20);
    EXPECT_DOUBLE_EQ(out.exc, 0.0);
    EXPECT_DOUBLE_EQ(out.vrho_alpha, 0.0);
    EXPECT_DOUBLE_EQ(out.vrho_beta, 0.0);
}

TEST(XCLibxcOracle, PbeAllChannelsFinite) {
    auto out = evaluate_pbe(0.5, 0.5, 0.1, 0.1, 0.1);
    EXPECT_TRUE(std::isfinite(out.exc));
    EXPECT_TRUE(std::isfinite(out.vrho_alpha));
    EXPECT_TRUE(std::isfinite(out.vrho_beta));
    EXPECT_TRUE(std::isfinite(out.vsigma_aa));
    EXPECT_TRUE(std::isfinite(out.vsigma_ab));
    EXPECT_TRUE(std::isfinite(out.vsigma_bb));
    EXPECT_LT(out.exc, 0.0);
}

TEST(XCLibxcOracle, PbeBelowCutoffReturnsZero) {
    auto out = evaluate_pbe(1e-20, 1e-20, 0, 0, 0);
    EXPECT_DOUBLE_EQ(out.exc, 0.0);
}

// Domain-validation: Slater LDA exchange textbook value at rho_alpha = rho_beta = 0.5
// (total rho = 1, zeta = 0). The closed-form result is independent of any
// correlation parameterization:
//   eps_x = -C_x * rho^(1/3),  C_x = (3/4)(3/pi)^(1/3) ≈ 0.7385587663820224 Ha.
//
// libxc XC_LDA_X must reproduce this to numerical precision; this is the
// canonical cross-library reference for LDA exchange and confirms the oracle
// is driving libxc with the right convention (XC_POLARIZED, density indices,
// exc-per-electron units).
TEST(XCLibxcOracle, SlaterTextbookValueAtUnitDensity) {
    constexpr double kExpected = -0.7385587663820224;
    auto out = evaluate_lda_x(0.5, 0.5);
    EXPECT_NEAR(out.exc, kExpected, 1e-10);
}
