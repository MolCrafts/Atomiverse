/// @file tests/core/test_lj_type_table.cpp
/// @brief Unit: `LJTypeTable` (LAMMPS init_one completion) and `r4_attraction`.
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/lj_type_table.h"
#include "core/typed_lj_cases.h"

using namespace ATV;
using namespace typed_lj_cases;

namespace {
constexpr double kTol = 1e-12;

double rel(double a, double b) {
    return std::fabs(a - b) / std::max(1.0, std::fabs(b));
}
} // namespace

TEST(LJTypeTable, UnsetPairMixesEpsilonSigmaAndCutoff) {
    LJTypeTable t(diagonal_rows(), kCutoff);
    t.build(2, /*shift=*/true);
    ASSERT_EQ(t.n_types(), 2);
    const LJPairCoeff& c = t(0, 1);
    EXPECT_NEAR(c.epsilon, kEps01, kTol);
    EXPECT_NEAR(c.sigma, kSig01, kTol);
    EXPECT_NEAR(c.cut_lj, kCut01, kTol);
    EXPECT_NEAR(c.cut_ljsq, kCut01 * kCut01, kTol);
    // LAMMPS lj1..lj4 and the offset at the pair's own cutoff.
    EXPECT_LT(rel(c.lj1, 48.0 * kEps01 * std::pow(kSig01, 12)), kTol);
    EXPECT_LT(rel(c.lj2, 24.0 * kEps01 * std::pow(kSig01, 6)), kTol);
    EXPECT_LT(rel(c.lj3, 4.0 * kEps01 * std::pow(kSig01, 12)), kTol);
    EXPECT_LT(rel(c.lj4, 4.0 * kEps01 * std::pow(kSig01, 6)), kTol);
    EXPECT_NEAR(c.offset, lj(kEps01, kSig01, kCut01), kTol);
    // Symmetric, and the diagonals keep their own cutoffs.
    EXPECT_EQ(t(1, 0).cut_lj, c.cut_lj);
    EXPECT_NEAR(t(0, 0).cut_lj, kCut0, kTol);
    EXPECT_NEAR(t(1, 1).cut_lj, kCut1, kTol);
}

TEST(LJTypeTable, DiagonalWithoutCutoffUsesTheGlobalOne) {
    LJTypeTable t({{0, 0, kEps0, kSig0}, {1, 1, kEps1, kSig1, kCut1}}, kCutoff);
    t.build(2, true);
    EXPECT_NEAR(t(0, 0).cut_lj, kCutoff, kTol);
    EXPECT_NEAR(t(0, 1).cut_lj, 0.5 * (kCutoff + kCut1), kTol);
}

TEST(LJTypeTable, ExplicitRowOverridesMixingAndDefaultsToGlobalCutoff) {
    auto rows = diagonal_rows();
    rows.push_back({0, 1, 0.3, 3.2}); // no cutoff: global, not the mix
    LJTypeTable t(rows, kCutoff);
    t.build(2, true);
    EXPECT_NEAR(t(1, 0).epsilon, 0.3, kTol);
    EXPECT_NEAR(t(1, 0).sigma, 3.2, kTol);
    EXPECT_NEAR(t(1, 0).cut_lj, kCutoff, kTol);
    t.set_coeff({1, 0, 0.3, 3.2, 7.5}); // a later row wins
    t.build(2, true);
    EXPECT_NEAR(t(0, 1).cut_lj, 7.5, kTol);
    EXPECT_NEAR(t(0, 1).offset, lj(0.3, 3.2, 7.5), kTol);
}

TEST(LJTypeTable, TypeWithoutDiagonalHasNoLJSite) {
    LJTypeTable t({{0, 0, kEps0, kSig0}}, kCutoff);
    t.build(3, true); // type 2 is in the Frame but named by no row
    ASSERT_EQ(t.n_types(), 3);
    for (int b = 0; b < 3; ++b) {
        EXPECT_EQ(t(2, b).epsilon, 0.0);
        EXPECT_EQ(t(2, b).cut_ljsq, 0.0);
        EXPECT_EQ(t(2, b).lj1, 0.0);
    }
}

TEST(LJTypeTable, ShiftOffLeavesNoOffset) {
    LJTypeTable t(diagonal_rows(), kCutoff);
    t.build(2, /*shift=*/false);
    EXPECT_EQ(t(0, 1).offset, 0.0);
    EXPECT_EQ(t(1, 1).offset, 0.0);
}

TEST(LJTypeTable, MaxCutoffCoversLJAndR4Rows) {
    LJTypeTable t({{0, 0, kEps0, kSig0, kCut0}}, kCutoff);
    EXPECT_NEAR(t.max_cutoff(), kCut0, kTol); // below the global: honest
    t.set_r4({{0, 1, kS, kSigR4, 11.0}});
    EXPECT_NEAR(t.max_cutoff(), 11.0, kTol);
    EXPECT_EQ(t.named_types(), 2);
}

TEST(LJTypeTable, R4ConstantsLandOnTheirPairOnly) {
    LJTypeTable t(diagonal_rows(), kCutoff);
    t.set_r4({{1, 0, kS, kSigR4, kCutR4}});
    t.build(2, true);
    const double a = kS * std::pow(kSigR4, 4);
    EXPECT_LT(rel(t(0, 1).r4_a, a), kTol);
    EXPECT_NEAR(t(0, 1).r4_cutsq, kCutR4 * kCutR4, kTol);
    EXPECT_LT(rel(t(0, 1).r4_rc5inv, std::pow(kCutR4, -5)), kTol);
    EXPECT_LT(rel(t(0, 1).r4_e0, 5.0 * a / std::pow(kCutR4, 4)), kTol);
    EXPECT_EQ(t(1, 0).r4_a, t(0, 1).r4_a);
    EXPECT_EQ(t(0, 0).r4_a, 0.0);
    EXPECT_EQ(t(1, 1).r4_cutsq, 0.0);
}

TEST(LJTypeTable, LorentzBerthelotUniformDetection) {
    LJTypeTable lb({{0, 0, kEps0, kSig0}, {1, 1, kEps1, kSig1}}, kCutoff);
    lb.build(2, true);
    EXPECT_TRUE(lb.lorentz_berthelot_uniform());
    // Explicit LB-valued cross term at the global cutoff: still uniform.
    lb.set_coeff({0, 1, kEps01, kSig01});
    lb.build(2, true);
    EXPECT_TRUE(lb.lorentz_berthelot_uniform());

    LJTypeTable cut(diagonal_rows(), kCutoff);
    cut.build(2, true);
    EXPECT_FALSE(cut.lorentz_berthelot_uniform()); // per-pair cutoffs

    LJTypeTable cross(
        {{0, 0, kEps0, kSig0}, {1, 1, kEps1, kSig1}, {0, 1, 0.3, 3.2}},
        kCutoff);
    cross.build(2, true);
    EXPECT_FALSE(cross.lorentz_berthelot_uniform());

    LJTypeTable r4({{0, 0, kEps0, kSig0}}, kCutoff);
    r4.set_r4({{0, 0, kS, kSigR4, kCutR4}});
    r4.build(1, true);
    EXPECT_FALSE(r4.lorentz_berthelot_uniform());
}

TEST(LJTypeTable, RejectsBadRows) {
    EXPECT_THROW(LJTypeTable({}, kCutoff), std::invalid_argument);
    EXPECT_THROW(LJTypeTable({{0, 0, kEps0, kSig0}}, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(LJTypeTable({{-1, 0, kEps0, kSig0}}, kCutoff),
                 std::invalid_argument);
    EXPECT_THROW(LJTypeTable({{0, 0, -1.0, kSig0}}, kCutoff),
                 std::invalid_argument);
    EXPECT_THROW(LJTypeTable({{0, 0, kEps0, 0.0}}, kCutoff),
                 std::invalid_argument);
    EXPECT_THROW(LJTypeTable({{0, 0, kEps0, kSig0, -2.0}}, kCutoff),
                 std::invalid_argument);
    LJTypeTable t({{0, 0, kEps0, kSig0}}, kCutoff);
    EXPECT_THROW(t.set_r4({{0, 0, kS, 0.0, kCutR4}}), std::invalid_argument);
    EXPECT_THROW(t.set_r4({{0, 0, kS, kSigR4, 0.0}}), std::invalid_argument);
    EXPECT_THROW(t.set_r4({{0, 0, NAN, kSigR4, kCutR4}}),
                 std::invalid_argument);
}

TEST(R4Attraction, MatchesTheDefiningFormula) {
    const double a = kS * std::pow(kSigR4, 4);
    for (double r : {2.0, 3.0, 4.5, 5.9}) {
        const R4Term<double> t = r4_attraction<double>(
            r * r, a, std::pow(kCutR4, -5), 5.0 * a / std::pow(kCutR4, 4));
        EXPECT_NEAR(t.energy, r4(kS, kSigR4, kCutR4, r), 1e-14) << r;
        // fpair = F / r.
        EXPECT_NEAR(t.fpair * r, r4_force(kS, kSigR4, kCutR4, r), 1e-14) << r;
        EXPECT_LT(t.energy, 0.0); // S > 0 attracts
    }
}

TEST(R4Attraction, ForceIsMinusTheEnergyDerivative) {
    const double a = kS * std::pow(kSigR4, 4);
    const double k5 = std::pow(kCutR4, -5), e0 = 5.0 * a / std::pow(kCutR4, 4);
    const double r = 3.7, h = 1e-5;
    auto u = [&](double x) {
        return r4_attraction<double>(x * x, a, k5, e0).energy;
    };
    const double fd = -(u(r + h) - u(r - h)) / (2 * h);
    EXPECT_NEAR(r4_attraction<double>(r * r, a, k5, e0).fpair * r, fd, 1e-9);
}

TEST(R4Attraction, EnergyAndForceVanishAtTheCutoff) {
    const double a = kS * std::pow(kSigR4, 4);
    const double k5 = std::pow(kCutR4, -5), e0 = 5.0 * a / std::pow(kCutR4, 4);
    const R4Term<double> at = r4_attraction<double>(kCutR4 * kCutR4, a, k5, e0);
    EXPECT_NEAR(at.energy, 0.0, 1e-15);
    EXPECT_NEAR(at.fpair, 0.0, 1e-15);
    // Just inside: U = O(δ²), F = O(δ).
    const double r = kCutR4 - 1e-4;
    const R4Term<double> in = r4_attraction<double>(r * r, a, k5, e0);
    EXPECT_LT(std::fabs(in.energy), 1e-9);
    EXPECT_LT(std::fabs(in.fpair * r), 1e-5);
}

TEST(LJTail, PairIsTheIntegralBeyondItsCutoff) {
    const LJTail t = lj_tail_pair(kEps0, kSig0, kCut0, 2.0, 3.0);
    EXPECT_LT(rel(t.energy, lj_tail_energy(kEps0, kSig0, kCut0, 2, 3)), kTol);
    EXPECT_LT(rel(t.virial, lj_tail_virial(kEps0, kSig0, kCut0, 2, 3)), kTol);
    // No LJ site (zero cutoff or zero epsilon): no tail.
    EXPECT_EQ(lj_tail_pair(kEps0, kSig0, 0.0, 2, 3).energy, 0.0);
    EXPECT_EQ(lj_tail_pair(0.0, kSig0, kCut0, 2, 3).virial, 0.0);
}

TEST(LJTail, TableSumsOrderedPairsAtTheirOwnCutoffs) {
    LJTypeTable t(diagonal_rows(), kCutoff);
    t.set_r4({{0, 1, kS, kSigR4, kCutR4}}); // the r^-4 term has no tail
    t.build(2, /*shift=*/true);
    const LJTail got = t.tail({2.0, 1.0});
    const double e = lj_tail_energy(kEps0, kSig0, kCut0, 2, 2)
                     + 2.0 * lj_tail_energy(kEps01, kSig01, kCut01, 2, 1)
                     + lj_tail_energy(kEps1, kSig1, kCut1, 1, 1);
    const double w = lj_tail_virial(kEps0, kSig0, kCut0, 2, 2)
                     + 2.0 * lj_tail_virial(kEps01, kSig01, kCut01, 2, 1)
                     + lj_tail_virial(kEps1, kSig1, kCut1, 1, 1);
    EXPECT_LT(rel(got.energy, e), kTol);
    EXPECT_LT(rel(got.virial, w), kTol);
    EXPECT_THROW(t.tail({1.0}), std::invalid_argument);
}
