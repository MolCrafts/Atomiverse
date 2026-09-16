/// @file tests/cpu/longrange/test_ewald_sum.cpp
/// @brief Unit tests for ATV::longrange::EwaldSum pure math.
///
/// Expected values are pinned to the pre-refactor reciprocal-space
/// implementation (bit-identical formulas moved into the facade).
/// No Frame/Component/MPI.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/longrange/ewald_sum.h"

using namespace ATV;
using ATV::longrange::EwaldSum;

namespace {

/// Cubic box of edge L [Å] → inv_cell = diag(1/L).
Mat3 inv_cubic(double L) {
    Mat3 ic;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ic(i, j) = (i == j) ? 1.0 / L : 0.0;
    return ic;
}

} // namespace

TEST(EwaldSumMath, RejectsNonPositiveAlpha) {
    EXPECT_THROW(EwaldSum(0.0, 1e-5), std::invalid_argument);
    EXPECT_THROW(EwaldSum(-0.1, 1e-5), std::invalid_argument);
}

TEST(EwaldSumMath, RejectsAccuracyOutsideUnitInterval) {
    EXPECT_THROW(EwaldSum(0.3, 0.0), std::invalid_argument);
    EXPECT_THROW(EwaldSum(0.3, 1.0), std::invalid_argument);
    EXPECT_THROW(EwaldSum(0.3, -1e-5), std::invalid_argument);
}

TEST(EwaldSumMath, RecommendAlphaPinnedToLammpsHeuristic) {
    // Pin: (1.35 − 0.15·ln(1e-5)) / 10  from pre-refactor KSpaceEwald.
    const double expected = (1.35 - 0.15 * std::log(1e-5)) / 10.0;
    EXPECT_DOUBLE_EQ(EwaldSum::recommend_alpha(10.0, 1e-5), expected);
    EXPECT_GT(expected, 0.3);
    EXPECT_LT(expected, 0.32);
}

TEST(EwaldSumMath, TwoIonCubicKTableAndReciprocalEnergy) {
    // Two opposite charges in a 10 Å cubic box. Pin k-count and reciprocal
    // energy to the pre-refactor direct-Ewald kernel (same gcut/α/accuracy).
    const double L = 10.0;
    const double alpha = 0.3;
    const double accuracy = 1e-5;
    EwaldSum sum(alpha, accuracy);
    sum.rebuild(inv_cubic(L));

    // Half-sphere pin: gcut = 2α√(−ln acc), positive-hemisphere filter
    // (kz>0 ∨ (kz=0∧ky>0) ∨ (kz=ky=0∧kx>0)). For L=10, α=0.3, acc=1e-5 → 73
    // retained k-vectors (was 146 full-sphere before half-sphere optimize).
    EXPECT_EQ(sum.size(), 73u);

    const double x[2] = {2.0, 7.0};
    const double y[2] = {3.0, 6.0};
    const double z[2] = {4.0, 5.0};
    const double q[2] = {1.0, -1.0};
    std::vector<double> sr(sum.size()), si(sum.size());
    sum.structure_factor(x, y, z, q, 2, sr.data(), si.data());

    double e_recip = 0;
    double stress[6];
    sum.reciprocal_energy_virial(sr.data(), si.data(), L * L * L, e_recip,
                                 stress);

    // Pin from pre-refactor implementation output (same formula, same α/acc).
    EXPECT_NEAR(e_recip, 1.5075250344529976, 1e-12);

    const double e_self = sum.self_energy(q[0] * q[0] + q[1] * q[1]);
    // −k_C · α/√π · Σq² with Σq² = 2.
    const double e_self_pin =
        -ATV::units::coulomb * alpha / std::sqrt(ATV::units::pi) * 2.0;
    EXPECT_DOUBLE_EQ(e_self, e_self_pin);

    // Neutral cell → net-charge term is exactly zero.
    EXPECT_DOUBLE_EQ(sum.net_charge_energy(0.0, L * L * L), 0.0);
}

TEST(EwaldSumMath, ForcesOpposeOnOppositeCharges) {
    const double L = 10.0;
    EwaldSum sum(0.3, 1e-5);
    sum.rebuild(inv_cubic(L));
    const double x[2] = {2.0, 7.0};
    const double y[2] = {3.0, 6.0};
    const double z[2] = {4.0, 5.0};
    const double q[2] = {1.0, -1.0};
    std::vector<double> sr(sum.size()), si(sum.size());
    sum.structure_factor(x, y, z, q, 2, sr.data(), si.data());
    double fx[2] = {0, 0}, fy[2] = {0, 0}, fz[2] = {0, 0};
    sum.add_forces(x, y, z, q, 2, L * L * L, sr.data(), si.data(), fx, fy, fz);
    // Newton's third law on a neutral pair in a cubic cell (reciprocal only).
    EXPECT_NEAR(fx[0] + fx[1], 0.0, 1e-10);
    EXPECT_NEAR(fy[0] + fy[1], 0.0, 1e-10);
    EXPECT_NEAR(fz[0] + fz[1], 0.0, 1e-10);
    EXPECT_GT(std::fabs(fx[0]) + std::fabs(fy[0]) + std::fabs(fz[0]), 1e-6);
}
