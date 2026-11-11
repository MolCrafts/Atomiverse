/// @file tests/kokkos/potentials/test_pair_lj_cut.cpp
/// @brief Kokkos Serial LJ kernel against the CPU PairLJCut analytic value.
///
/// The two-atom r = σ energy is the quantity asserted by
/// `tests/cpu/dynamics/test_pair_lj_cut.cpp` (`EnergyMatchesAnalyticFormula`):
/// V = −V_shift. That CPU test allows 1e-6 because one Verlet step moves
/// the atoms. This call does not integrate, so the same number is matched
/// much tighter.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/kokkos/potentials/pair_lj_cut.h"

namespace {

double v_shift(double eps, double sigma, double rcut) {
    const double sr = sigma / rcut;
    const double sr2 = sr * sr;
    const double sr4 = sr2 * sr2;
    const double sr6 = sr4 * sr2;
    return 4.0 * eps * (sr6 * sr6 - sr6);
}

/// CPU pair force multiplier, pair_lj_cut.h: (24ε/r²)[2(σ/r)¹² − (σ/r)⁶].
double fpair_cpu(double eps, double sigma, double r) {
    const double inv_r = 1.0 / r;
    const double sr = sigma * inv_r;
    const double sr2 = sr * sr;
    const double sr4 = sr2 * sr2;
    const double sr6 = sr4 * sr2;
    const double r2inv = inv_r * inv_r;
    return 24.0 * eps * r2inv * (2.0 * sr6 * sr6 - sr6);
}

} // namespace

TEST(KokkosPairLJCut, RejectsNonPositiveSize) {
    EXPECT_THROW(ATV::kokkos::PairLJCut(0.5, 0.0, 8.0), std::invalid_argument);
    EXPECT_THROW(ATV::kokkos::PairLJCut(0.5, 3.0, 0.0), std::invalid_argument);
}

TEST(KokkosPairLJCut, EnergyAtSigmaMatchesCpuAnalytic) {
    // Same geometry as test_cpu_md_pair_lj_cut EnergyMatchesAnalyticFormula.
    const double eps = 0.5;
    const double sigma = 3.0;
    const double rcut = 8.0;
    ATV::kokkos::PairLJCut lj(eps, sigma, rcut);

    std::vector<double> x{0.0, sigma};
    std::vector<double> y{0.0, 0.0};
    std::vector<double> z{0.0, 0.0};
    std::vector<int> counts{1, 0};
    std::vector<int> offsets{0, 1};
    std::vector<int> neighbors{1};
    std::vector<double> fx(2), fy(2), fz(2);

    ATV::kokkos::HalfNeighborList neigh{counts, offsets, neighbors};
    const double energy = lj.evaluate(x, y, z, neigh, fx, fy, fz);

    const double expected = -v_shift(eps, sigma, rcut);
    EXPECT_NEAR(energy, expected, 1e-12);

    // delx = x0 − x1 = −σ, fpair > 0 on the repulsive wall → fx0 < 0.
    const double fpair = fpair_cpu(eps, sigma, sigma);
    const double delx = -sigma;
    EXPECT_NEAR(fx[0], delx * fpair, 1e-12);
    EXPECT_NEAR(fx[1], -delx * fpair, 1e-12);
    EXPECT_NEAR(fy[0], 0.0, 1e-12);
    EXPECT_NEAR(fz[1], 0.0, 1e-12);
    EXPECT_NEAR(std::fabs(fx[0]), 24.0 * eps / sigma, 1e-12);
}

TEST(KokkosPairLJCut, HalfListChainsTwoPairs) {
    const double eps = 0.5;
    const double sigma = 3.0;
    const double rcut = 8.0;
    ATV::kokkos::PairLJCut lj(eps, sigma, rcut);

    // 0—1 and 1—2, each separated by σ. Pair stored once on the lower index.
    std::vector<double> x{0.0, sigma, 2.0 * sigma};
    std::vector<double> y{0.0, 0.0, 0.0};
    std::vector<double> z{0.0, 0.0, 0.0};
    std::vector<int> counts{1, 1, 0};
    std::vector<int> offsets{0, 1, 2};
    std::vector<int> neighbors{1, 2};
    std::vector<double> fx(3), fy(3), fz(3);

    ATV::kokkos::HalfNeighborList neigh{counts, offsets, neighbors};
    const double energy = lj.evaluate(x, y, z, neigh, fx, fy, fz);

    const double one = -v_shift(eps, sigma, rcut);
    EXPECT_NEAR(energy, 2.0 * one, 1e-12);

    const double f = 24.0 * eps / sigma;
    EXPECT_NEAR(fx[0], -f, 1e-12);
    EXPECT_NEAR(fx[1], 0.0, 1e-12); // +f from pair 0-1, −f from pair 1-2
    EXPECT_NEAR(fx[2], f, 1e-12);
}

TEST(KokkosPairLJCut, OutsideCutoffContributesNothing) {
    const double eps = 0.5;
    const double sigma = 3.0;
    const double rcut = 8.0;
    ATV::kokkos::PairLJCut lj(eps, sigma, rcut);

    std::vector<double> x{0.0, rcut + 1.0};
    std::vector<double> y{0.0, 0.0};
    std::vector<double> z{0.0, 0.0};
    std::vector<int> counts{1, 0};
    std::vector<int> offsets{0, 1};
    std::vector<int> neighbors{1};
    std::vector<double> fx{1.0, 1.0};
    std::vector<double> fy{1.0, 1.0};
    std::vector<double> fz{1.0, 1.0};

    ATV::kokkos::HalfNeighborList neigh{counts, offsets, neighbors};
    const double energy = lj.evaluate(x, y, z, neigh, fx, fy, fz);
    EXPECT_DOUBLE_EQ(energy, 0.0);
    EXPECT_DOUBLE_EQ(fx[0], 0.0);
    EXPECT_DOUBLE_EQ(fx[1], 0.0);
    EXPECT_DOUBLE_EQ(fy[0], 0.0);
    EXPECT_DOUBLE_EQ(fz[1], 0.0);
}

TEST(KokkosPairLJCut, RejectsSelfPair) {
    ATV::kokkos::PairLJCut lj(0.5, 3.0, 8.0);
    std::vector<double> x{0.0};
    std::vector<double> y{0.0};
    std::vector<double> z{0.0};
    std::vector<int> counts{1};
    std::vector<int> offsets{0};
    std::vector<int> neighbors{0};
    std::vector<double> fx(1), fy(1), fz(1);
    ATV::kokkos::HalfNeighborList neigh{counts, offsets, neighbors};
    EXPECT_THROW(lj.evaluate(x, y, z, neigh, fx, fy, fz),
                 std::invalid_argument);
}
