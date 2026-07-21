/// @file tests/cpu/scf/test_ediis_adiis.cpp
/// @brief Unit tests for the EDIIS / ADIIS far-field Fock mixers
///        (`internal::EDIISMixer`, `internal::ADIISMixer`): two-point
///        closed-form simplex solutions, coefficient constraints, objective
///        values, and degenerate-history behavior. No SCF loop runs here.
///
/// Closed forms (2-point history, c = (1-x, x)):
///   EDIIS  f(x) = (1-x)E1 + xE2 − x(1-x)·b,  b = ⟨D1−D2|F1−F2⟩
///          ⇒ interior minimum at x* = (b − (E2−E1)) / (2b) when b > 0.
///   ADIIS  f(y) = 2ay + by²  with y = c1, a = ⟨D1−D2|F2⟩, b = ⟨D1−D2|F1−F2⟩
///          ⇒ interior minimum at y* = −a/b when b > 0 and 0 < −a/b < 1.

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/internal/ediis_adiis.h"

using namespace ATV::cpu::scf;
using internal::ADIISMixer;
using internal::EDIISMixer;

namespace {

RealMatrix diag2(double a, double b) { return {{a, 0.0}, {0.0, b}}; }

bool all_finite(const RealMatrix& M) {
    for (std::size_t i = 0; i < M.shape()[0]; ++i)
        for (std::size_t j = 0; j < M.shape()[1]; ++j)
            if (!std::isfinite(M(i, j)))
                return false;
    return true;
}

} // namespace

// ── EDIIS ────────────────────────────────────────────────────────────────

TEST(EDIIS, DepthOneHistoryReturnsInputUnchanged) {
    EDIISMixer ediis(6);
    RealMatrix F = diag2(0.5, 0.1), D = diag2(0.2, 0.1);
    RealMatrix out = ediis.step(F, D, -1.0);
    EXPECT_DOUBLE_EQ(out(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(out(1, 1), 0.1);
    ASSERT_EQ(ediis.coefficients().size(), 1u);
    EXPECT_DOUBLE_EQ(ediis.coefficients()[0], 1.0);
}

TEST(EDIIS, TwoPointClosedFormInteriorMinimum) {
    // D1−D2 = diag(0.1, −0.1), F1−F2 = diag(0.5, 0.3)
    //   ⇒ b = 0.1·0.5 − 0.1·0.3 = 0.02
    // E1 = −1.00, E2 = −0.99 ⇒ x* = (0.02 − 0.01)/0.04 = 0.25.
    EDIISMixer ediis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, -0.2), D2 = diag2(0.1, 0.2);
    ediis.step(F1, D1, -1.00);
    RealMatrix out = ediis.step(F2, D2, -0.99);

    const auto& c = ediis.coefficients();
    ASSERT_EQ(c.size(), 2u);
    EXPECT_NEAR(c[0], 0.75, 1e-10);
    EXPECT_NEAR(c[1], 0.25, 1e-10);
    EXPECT_NEAR(c[0] + c[1], 1.0, 1e-12);
    EXPECT_GE(c[0], 0.0);
    EXPECT_GE(c[1], 0.0);

    // F̃ = 0.75·F1 + 0.25·F2.
    EXPECT_NEAR(out(0, 0), 0.75 * 0.5 + 0.25 * 0.0, 1e-10);
    EXPECT_NEAR(out(1, 1), 0.75 * 0.1 + 0.25 * (-0.2), 1e-10);
}

TEST(EDIIS, TwoPointVertexSolutionWhenLowerEnergyDominates) {
    // Same b = 0.02, but E2 − E1 = −1 ⇒ x* = (0.02+1)/0.04 ≫ 1 ⇒ vertex
    // c = (0, 1): all weight on the (much) lower-energy iterate.
    EDIISMixer ediis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, -0.2), D2 = diag2(0.1, 0.2);
    ediis.step(F1, D1, -1.0);
    RealMatrix out = ediis.step(F2, D2, -2.0);

    const auto& c = ediis.coefficients();
    EXPECT_NEAR(c[0], 0.0, 1e-10);
    EXPECT_NEAR(c[1], 1.0, 1e-10);
    EXPECT_NEAR(out(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(out(1, 1), -0.2, 1e-10);
}

TEST(EDIIS, ObjectiveMatchesHandFormula) {
    EDIISMixer ediis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, -0.2), D2 = diag2(0.1, 0.2);
    ediis.step(F1, D1, -1.00);
    ediis.step(F2, D2, -0.99);
    // f(c) = c1·E1 + c2·E2 − c1·c2·b with b = 0.02.
    const double f = ediis.objective({0.5, 0.5});
    EXPECT_NEAR(f, 0.5 * (-1.00) + 0.5 * (-0.99) - 0.25 * 0.02, 1e-14);
}

TEST(EDIIS, DegenerateDuplicateHistoryReturnsLatestNoNaN) {
    EDIISMixer ediis(6);
    RealMatrix F = diag2(0.4, -0.3), D = diag2(0.3, 0.1);
    ediis.step(F, D, -1.0);
    ediis.step(F, D, -1.0);
    RealMatrix out = ediis.step(F, D, -1.0);
    EXPECT_TRUE(all_finite(out));
    EXPECT_DOUBLE_EQ(out(0, 0), 0.4);
    EXPECT_DOUBLE_EQ(out(1, 1), -0.3);
    double sum = 0.0;
    for (double ci : ediis.coefficients()) {
        EXPECT_TRUE(std::isfinite(ci));
        EXPECT_GE(ci, 0.0);
        sum += ci;
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(EDIIS, DualChannelSharedCoefficients) {
    // Unrestricted step: the β channel is combined with the SAME c as α.
    EDIISMixer ediis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, -0.2), D2 = diag2(0.1, 0.2);
    RealMatrix Z = diag2(0.0, 0.0);
    // β channel carries zero matrices: metric reduces to the α-only case.
    ediis.step(F1, D1, Z, Z, -1.00);
    auto [fa, fb] = ediis.step(F2, D2, Z, Z, -0.99);
    const auto& c = ediis.coefficients();
    EXPECT_NEAR(c[0], 0.75, 1e-10);
    EXPECT_NEAR(fa(0, 0), 0.75 * 0.5, 1e-10);
    EXPECT_NEAR(fb(0, 0), 0.0, 1e-14);
}

// ── ADIIS ────────────────────────────────────────────────────────────────

TEST(ADIIS, DepthOneHistoryReturnsInputUnchanged) {
    ADIISMixer adiis(6);
    RealMatrix F = diag2(0.5, 0.1), D = diag2(0.2, 0.1);
    RealMatrix out = adiis.step(F, D);
    EXPECT_DOUBLE_EQ(out(0, 0), 0.5);
    ASSERT_EQ(adiis.coefficients().size(), 1u);
    EXPECT_DOUBLE_EQ(adiis.coefficients()[0], 1.0);
}

TEST(ADIIS, TwoPointClosedFormInteriorMinimum) {
    // D1−D2 = diag(0.1, −0.1), F2 = diag(0.0, 0.2)
    //   ⇒ a = ⟨D1−D2|F2⟩ = −0.02
    // F1−F2 = diag(0.5, −0.1) ⇒ b = 0.05 + 0.01 = 0.06
    //   ⇒ y* = −a/b = 1/3 ⇒ c = (1/3, 2/3).
    ADIISMixer adiis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, 0.2), D2 = diag2(0.1, 0.2);
    adiis.step(F1, D1);
    RealMatrix out = adiis.step(F2, D2);

    const auto& c = adiis.coefficients();
    ASSERT_EQ(c.size(), 2u);
    EXPECT_NEAR(c[0], 1.0 / 3.0, 1e-10);
    EXPECT_NEAR(c[1], 2.0 / 3.0, 1e-10);
    EXPECT_NEAR(c[0] + c[1], 1.0, 1e-12);
    EXPECT_GE(c[0], 0.0);
    EXPECT_GE(c[1], 0.0);

    EXPECT_NEAR(out(0, 0), (1.0 / 3.0) * 0.5, 1e-10);
    EXPECT_NEAR(out(1, 1), (1.0 / 3.0) * 0.1 + (2.0 / 3.0) * 0.2, 1e-10);
}

TEST(ADIIS, ObjectiveValueAtKnownCoefficients) {
    ADIISMixer adiis(6);
    RealMatrix F1 = diag2(0.5, 0.1), D1 = diag2(0.2, 0.1);
    RealMatrix F2 = diag2(0.0, 0.2), D2 = diag2(0.1, 0.2);
    adiis.step(F1, D1);
    adiis.step(F2, D2);
    // f(c) = 2c1·a + c1²·b with a = −0.02, b = 0.06:
    //   f({0.5, 0.5}) = 2·0.5·(−0.02) + 0.25·0.06 = −0.005.
    EXPECT_NEAR(adiis.objective({0.5, 0.5}), -0.005, 1e-14);
    // At the solved optimum the objective is not above the tested point.
    EXPECT_LE(adiis.objective(adiis.coefficients()), -0.005 + 1e-14);
}

TEST(ADIIS, DegenerateDuplicateHistoryReturnsLatestNoNaN) {
    ADIISMixer adiis(6);
    RealMatrix F = diag2(0.4, -0.3), D = diag2(0.3, 0.1);
    adiis.step(F, D);
    adiis.step(F, D);
    RealMatrix out = adiis.step(F, D);
    EXPECT_TRUE(all_finite(out));
    EXPECT_DOUBLE_EQ(out(0, 0), 0.4);
    EXPECT_DOUBLE_EQ(out(1, 1), -0.3);
    double sum = 0.0;
    for (double ci : adiis.coefficients()) {
        EXPECT_TRUE(std::isfinite(ci));
        EXPECT_GE(ci, 0.0);
        sum += ci;
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(ADIIS, HistoryTrimsToMaxVectors) {
    ADIISMixer adiis(3);
    for (int k = 0; k < 6; ++k) {
        RealMatrix F = diag2(0.1 * k, -0.05 * k);
        RealMatrix D = diag2(0.2 + 0.01 * k, 0.1);
        adiis.step(F, D);
    }
    EXPECT_EQ(adiis.size(), 3);
    EXPECT_EQ(adiis.coefficients().size(), 3u);
}

#endif // ATV_HAS_HF
