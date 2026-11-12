/// @file tests/cpu/longrange/test_ewald_sum.cpp
/// @brief Unit tests for ATV::longrange::EwaldSum pure math.
///
/// Expected values are pinned to the pre-refactor reciprocal-space
/// implementation (bit-identical formulas moved into the facade).
/// No Frame/Component/MPI.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/longrange/ewald_sum.h"

using namespace ATV;
using ATV::longrange::EwaldSum;

namespace {

/// S(k), forces and energy of one charge set.
struct Reciprocal {
    std::vector<double> sr, si, fx, fy, fz;
    double energy = 0;
};

/// The phase-table kernels (`EwaldSum::phases` + `structure_factor` +
/// `add_forces`) with `threads` OpenMP threads available.
Reciprocal tabulated(const EwaldSum& sum, const std::vector<double>& x,
                     const std::vector<double>& y, const std::vector<double>& z,
                     const std::vector<double>& q, double volume, int threads) {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    const int n = static_cast<int>(q.size());
    Reciprocal r;
    r.sr.resize(sum.size());
    r.si.resize(sum.size());
    r.fx.assign(n, 0.0);
    r.fy.assign(n, 0.0);
    r.fz.assign(n, 0.0);
    ATV::longrange::EwaldPhases ph;
    sum.phases(x.data(), y.data(), z.data(), n, ph);
    sum.structure_factor(ph, q.data(), n, r.sr.data(), r.si.data());
    sum.add_forces(ph, q.data(), n, volume, r.sr.data(), r.si.data(),
                   r.fx.data(), r.fy.data(), r.fz.data());
    double stress[6];
    sum.reciprocal_energy_virial(r.sr.data(), r.si.data(), volume, r.energy,
                                 stress);
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif
    return r;
}

/// The textbook sum the tables replace: one cosine and one sine of k·r per
/// (k, atom) pair, S(k) = Σ q e^{ik·r}, F_i = 2K q_i Σ A_k Im(e^{ik·r_i} S̄_k) k.
Reciprocal direct(const EwaldSum& sum, const std::vector<double>& x,
                  const std::vector<double>& y, const std::vector<double>& z,
                  const std::vector<double>& q, double volume) {
    const std::size_t nk = sum.size(), n = q.size();
    const double K = 4.0 * units::pi / volume; // qqrd2e = 1
    Reciprocal r;
    r.sr.assign(nk, 0.0);
    r.si.assign(nk, 0.0);
    r.fx.assign(n, 0.0);
    r.fy.assign(n, 0.0);
    r.fz.assign(n, 0.0);
    for (std::size_t k = 0; k < nk; ++k)
        for (std::size_t i = 0; i < n; ++i) {
            const double phase =
                sum.kx()[k] * x[i] + sum.ky()[k] * y[i] + sum.kz()[k] * z[i];
            r.sr[k] += q[i] * std::cos(phase);
            r.si[k] += q[i] * std::sin(phase);
        }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < nk; ++k) {
            const double phase =
                sum.kx()[k] * x[i] + sum.ky()[k] * y[i] + sum.kz()[k] * z[i];
            const double fac =
                2.0 * K * sum.kw()[k] * q[i]
                * (std::sin(phase) * r.sr[k] - std::cos(phase) * r.si[k]);
            r.fx[i] += fac * sum.kx()[k];
            r.fy[i] += fac * sum.ky()[k];
            r.fz[i] += fac * sum.kz()[k];
        }
    for (std::size_t k = 0; k < nk; ++k)
        r.energy += K * sum.kw()[k] * (r.sr[k] * r.sr[k] + r.si[k] * r.si[k]);
    return r;
}

/// max |a − b| / max |b| over a column.
double rel_max(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0, scale = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
        scale = std::max(scale, std::fabs(b[i]));
    }
    return err / scale;
}

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
    ATV::longrange::EwaldPhases ph;
    sum.phases(x, y, z, 2, ph);
    sum.structure_factor(ph, q, 2, sr.data(), si.data());

    double e_recip = 0;
    double stress[6];
    sum.reciprocal_energy_virial(sr.data(), si.data(), L * L * L, e_recip,
                                 stress);

    // Pin from pre-refactor implementation output (same formula, same α/acc)
    // in lj units (qqrd2e = 1): the metal-unit pin 1.5075250344529976
    // divided by k_e = 14.399645.
    EXPECT_NEAR(e_recip, 0.10469181944784046, 1e-12);

    const double e_self = sum.self_energy(q[0] * q[0] + q[1] * q[1]);
    // −k_C · α/√π · Σq² with Σq² = 2.
    const double e_self_pin = -ATV::units::UnitSystem::lj().qqrd2e() * alpha
                              / std::sqrt(ATV::units::pi) * 2.0;
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
    ATV::longrange::EwaldPhases ph;
    sum.phases(x, y, z, 2, ph);
    sum.structure_factor(ph, q, 2, sr.data(), si.data());
    double fx[2] = {0, 0}, fy[2] = {0, 0}, fz[2] = {0, 0};
    sum.add_forces(ph, q, 2, L * L * L, sr.data(), si.data(), fx, fy, fz);
    // Newton's third law on a neutral pair in a cubic cell (reciprocal only).
    EXPECT_NEAR(fx[0] + fx[1], 0.0, 1e-10);
    EXPECT_NEAR(fy[0] + fy[1], 0.0, 1e-10);
    EXPECT_NEAR(fz[0] + fz[1], 0.0, 1e-10);
    EXPECT_GT(std::fabs(fx[0]) + std::fabs(fy[0]) + std::fabs(fz[0]), 1e-6);
}

TEST(EwaldSumMath, SkewCellReciprocalSphereUsesDualColumnBasis) {
    Mat3 h = Mat3::diagonal(4.0, 3.0, 5.0);
    h(0, 1) = 7.0;
    h(1, 2) = 1.2;
    const double alpha = 0.4, accuracy = 1e-5;
    EwaldSum sum(alpha, accuracy);
    sum.rebuild(h.inverse());
    const Mat3 b = h.inverse().transpose() * AccReal(2 * units::pi);
    const double cutoff2 = 4 * alpha * alpha * -std::log(accuracy);
    std::size_t count = 0;
    for (int i = -12; i <= 12; ++i)
        for (int j = -12; j <= 12; ++j)
            for (int k = -12; k <= 12; ++k) {
                if (k < 0 || (k == 0 && j < 0) || (k == 0 && j == 0 && i <= 0))
                    continue;
                const Vec3 g = b * Vec3{AccReal(i), AccReal(j), AccReal(k)};
                if (g.norm_squared() > cutoff2)
                    continue;
                ++count;
                bool found = false;
                for (std::size_t n = 0; n < sum.size(); ++n)
                    if (std::abs(sum.kx()[n] - g[0]) < 1e-12
                        && std::abs(sum.ky()[n] - g[1]) < 1e-12
                        && std::abs(sum.kz()[n] - g[2]) < 1e-12)
                        found = true;
                EXPECT_TRUE(found) << i << "," << j << "," << k;
            }
    EXPECT_EQ(sum.size(), count);
}

// The phase-table kernels evaluate the same S(k), forces and energy as the
// cosine/sine-per-pair sum, in a general cell (every column sign, both
// in-plane powers negative) with atoms outside the cell and more atoms than
// one kernel block (256), split over two threads. The rounding differs, so the gate is relative: the
// direct sum's own phase k·r carries |k·r|·ε ≈ 1e-14 and the recurrence
// m·ε; measured differences are below 4e-15, so 1e-12 leaves two orders of
// margin while any wrong term (a sign, a missing conjugate) shows at O(1).
TEST(EwaldSumMath, PhaseTablesMatchTheDirectTrigonometricSum) {
    // A general cell (not upper triangular), so every reciprocal basis
    // vector has three nonzero components.
    Mat3 h = Mat3::diagonal(14.0, 12.5, 13.3);
    h(0, 1) = 2.1;
    h(0, 2) = -1.3;
    h(1, 2) = 0.9;
    h(1, 0) = 0.7;
    h(2, 0) = -0.4;
    h(2, 1) = 1.1;
    const double volume = h.det();
    EwaldSum sum(0.4, 1e-9);
    sum.rebuild(h.inverse());
    constexpr int kAtoms = 1200;
    // Enough (k, atom) pairs that 4 available threads open a team of two.
    ASSERT_GT(sum.size() * kAtoms, 2 * EwaldSum::kPairsPerThread);

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> frac(-0.1, 1.1);
    std::uniform_real_distribution<double> charge(0.5, 1.5);
    std::vector<double> x(kAtoms), y(kAtoms), z(kAtoms), q(kAtoms);
    for (int i = 0; i < kAtoms; ++i) {
        const Vec3 r = h * Vec3{frac(rng), frac(rng), frac(rng)};
        x[i] = r[0];
        y[i] = r[1];
        z[i] = r[2];
        q[i] = (i % 2 ? -1.0 : 1.0) * charge(rng);
    }

    const Reciprocal ref = direct(sum, x, y, z, q, volume);
    const Reciprocal got = tabulated(sum, x, y, z, q, volume, 1);
    EXPECT_LT(rel_max(got.sr, ref.sr), 1e-12);
    EXPECT_LT(rel_max(got.si, ref.si), 1e-12);
    EXPECT_LT(rel_max(got.fx, ref.fx), 1e-12);
    EXPECT_LT(rel_max(got.fy, ref.fy), 1e-12);
    EXPECT_LT(rel_max(got.fz, ref.fz), 1e-12);
    EXPECT_LT(std::fabs(got.energy / ref.energy - 1.0), 1e-12);

    // Each S(k) and each atom's force is summed in one order whatever the
    // team: a split team (columns and atom ranges per thread, partial
    // blocks) gives the 1-thread bits.
    const Reciprocal team = tabulated(sum, x, y, z, q, volume, 4);
    EXPECT_EQ(team.sr, got.sr);
    EXPECT_EQ(team.si, got.si);
    EXPECT_EQ(team.fx, got.fx);
    EXPECT_EQ(team.fy, got.fy);
    EXPECT_EQ(team.fz, got.fz);
}
