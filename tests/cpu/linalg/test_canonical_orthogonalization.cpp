/// @file tests/cpu/linalg/test_canonical_orthogonalization.cpp
/// @brief Unit tests for `ATV::cpu::linalg::CanonicalOrth`
///        (spec hf-robust-02-canonical-orthogonalization, acceptance ac-001).
///
/// Contract under test (`atomiverse/cpu/linalg/eigensolve.h`) — direct
/// struct instantiation, no factory function (repo convention):
///   struct CanonicalOrth {
///       CanonicalOrth(const RealMatrix& S, double tau);
///       RealMatrix X; int n_dropped; double s_min;
///   };
/// The constructor eigendecomposes S (ascending), KEEPs eigenvalues
/// s_i >= tau, drops the rest,
/// X = U_kept * diag(1/sqrt(s_kept)) (n x m), n_dropped = columns dropped,
/// s_min = raw smallest eigenvalue of S (before filtering). tau <= 0 keeps all
/// n columns. If every column would be dropped, throw std::runtime_error.
///
/// Domain validation: X^T S X = I_m is the Roothaan-Hall well-conditioning
/// invariant (spec "Domain basis") — it is what makes F' = X^T F X an ordinary
/// symmetric eigenproblem with C = X C' spanning the non-degenerate subspace.
///
/// Fixtures are synthetic S = Q D Q^T with a hand-written orthogonal Q built
/// from Pythagorean-triple Givens rotations (self-checked to 1e-14), plus an
/// exactly diagonal S for assertions where LAPACK dsyevd is bitwise exact.
///
/// Tolerance calibration (numpy/LAPACK dsyevd on the exact fixture below):
///  - kept-column residual max|X^T S X - I_3| ~ 7e-16  -> 1e-12 bound is safe.
///  - reconstructed 1e-12 eigenvalue carries Weyl noise |ds| <= eps*||S||
///    ~ 3e-16 absolute (~3e-4 relative!), so s_min on the *rotated* fixture is
///    pinned absolutely (1e-14); the relative 1e-6 pin lives on the *diagonal*
///    fixture where dsyevd reproduces the entry exactly.
///  - tau = 0 on the near-singular fixture scales the near-null column by
///    1/sqrt(1e-12) = 1e6, amplifying the eigh residual to ~5e-5 in X^T S X;
///    the tight 1e-12 identity bound for the keep-all case therefore uses a
///    well-conditioned D through the same Q.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/linalg/eigensolve.h"

using ATV::cpu::linalg::CanonicalOrth;
using ATV::cpu::linalg::RealMatrix;

namespace {

/// Injected near-null eigenvalue (dimensionless overlap eigenvalue).
constexpr double kTinyEig = 1e-12;
/// Default linear-dependence threshold from the spec (method-member default).
constexpr double kTau = 1e-8;

/// 4x4 Givens rotation in the (i,j) plane with literal cosine/sine.
RealMatrix givens4(int i, int j, double c, double s) {
    RealMatrix g = xt::eye<double>(4);
    g(i, i) = c;
    g(j, j) = c;
    g(i, j) = -s;
    g(j, i) = s;
    return g;
}

/// Fixed hand-written orthogonal Q: product of three Pythagorean-triple
/// Givens rotations (3-4-5 twice, 7-24-25 once), so every literal is a
/// short decimal and Q^T Q = I holds to machine precision.
RealMatrix make_q() {
    RealMatrix q01 = givens4(0, 1, 0.8, 0.6);
    RealMatrix q12 = givens4(1, 2, 0.6, 0.8);
    RealMatrix q23 = givens4(2, 3, 0.28, 0.96);
    RealMatrix q01_12(xt::linalg::dot(q01, q12));
    return RealMatrix(xt::linalg::dot(q01_12, q23));
}

/// Symmetric positive-semi-definite S = Q diag(d) Q^T, explicitly
/// symmetrized so the eigensolver input carries no last-bit asymmetry.
RealMatrix make_spd(const std::array<double, 4>& d) {
    RealMatrix q = make_q();
    RealMatrix qd = q;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            qd(i, j) = q(i, j) * d[j];
    RealMatrix qt(xt::transpose(q));
    RealMatrix s(xt::linalg::dot(qd, qt));
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i + 1; j < 4; ++j) {
            const double v = 0.5 * (s(i, j) + s(j, i));
            s(i, j) = v;
            s(j, i) = v;
        }
    return s;
}

/// Near-singular spectrum: one direction at kTinyEig, three healthy ones.
RealMatrix make_near_singular_s() {
    return make_spd({1.5, 1.0, 0.5, kTinyEig});
}

/// Well-conditioned spectrum (used where 1/sqrt(s) must not amplify noise).
RealMatrix make_healthy_s() { return make_spd({1.5, 1.0, 0.5, 0.25}); }

/// Exactly diagonal S: LAPACK dsyevd returns the diagonal entries exactly
/// (zero off-diagonals -> tridiagonalization and D&C are exact), so
/// eigenvalue pins on this fixture need no reconstruction-noise allowance.
RealMatrix make_diagonal_s() {
    RealMatrix s = xt::zeros<double>({4, 4});
    s(0, 0) = 1.5;
    s(1, 1) = 1.0;
    s(2, 2) = 0.5;
    s(3, 3) = kTinyEig;
    return s;
}

/// X^T S X, materialized step by step (no lazy views into BLAS calls).
RealMatrix xt_s_x(const RealMatrix& s, const RealMatrix& x) {
    RealMatrix sx(xt::linalg::dot(s, x));
    RealMatrix xt(xt::transpose(x));
    return RealMatrix(xt::linalg::dot(xt, sx));
}

/// max_ij |M_ij - delta_ij|.
double max_abs_dev_from_identity(const RealMatrix& m) {
    double dev = 0.0;
    for (std::size_t i = 0; i < m.shape(0); ++i)
        for (std::size_t j = 0; j < m.shape(1); ++j) {
            const double target = (i == j) ? 1.0 : 0.0;
            dev = std::max(dev, std::abs(m(i, j) - target));
        }
    return dev;
}

} // namespace

// ── Fixture self-check ───────────────────────────────────────────────

// The synthetic fixture must be self-checking: Q^T Q = I to 1e-14, so any
// downstream assertion failure indicts the implementation, not the fixture.
TEST(CanonicalOrthogonalization, FixtureQIsOrthogonal) {
    RealMatrix q = make_q();
    RealMatrix qt(xt::transpose(q));
    RealMatrix qtq(xt::linalg::dot(qt, q));
    EXPECT_LT(max_abs_dev_from_identity(qtq), 1e-14)
        << "Givens-product fixture Q lost orthogonality";
}

// ── ac-001: synthetic near-singular S, tau = 1e-8 ───────────────────

// [ac-001] Basics: one ~1e-12 eigen-direction below tau=1e-8 -> X is n x (n-1).
TEST(CanonicalOrthogonalization, NearSingularXHasShapeNByNMinusOne) {
    CanonicalOrth r(make_near_singular_s(), kTau);
    ASSERT_EQ(r.X.shape(0), 4u);
    EXPECT_EQ(r.X.shape(1), 3u);
}

// [ac-001] Domain validation: X^T S X = I_m — the Roothaan-Hall
// well-conditioning invariant that makes F' = X^T F X an ordinary
// symmetric eigenproblem on the kept subspace.
TEST(CanonicalOrthogonalization, NearSingularXtSXIsIdentityTo1em12) {
    RealMatrix s = make_near_singular_s();
    CanonicalOrth r(s, kTau);
    ASSERT_EQ(r.X.shape(1), 3u);
    EXPECT_LT(max_abs_dev_from_identity(xt_s_x(s, r.X)), 1e-12)
        << "kept columns must S-orthonormalize the subspace";
}

// [ac-001] Basics: exactly the one sub-threshold direction is dropped.
TEST(CanonicalOrthogonalization, NearSingularDropsExactlyOneColumn) {
    CanonicalOrth r(make_near_singular_s(), kTau);
    EXPECT_EQ(r.n_dropped, 1);
}

// [ac-001] Basics: s_min reports the raw smallest eigenvalue of S (before
// filtering). Absolute pin: eigh on the rotated fixture carries Weyl noise
// |ds| <= eps * ||S||_2 ~ 3e-16 (measured), so 1e-14 gives ~30x margin while
// still proving s_min is the injected 1e-12 direction and not 0.5. The
// tighter relative-1e-6 pin lives on the exact diagonal fixture below.
TEST(CanonicalOrthogonalization, NearSingularSminReportsRawSmallestEigenvalue) {
    CanonicalOrth r(make_near_singular_s(), kTau);
    EXPECT_NEAR(r.s_min, kTinyEig, 1e-14)
        << "s_min must be the raw pre-filter smallest eigenvalue";
}

// [ac-001] Basics: on an exactly diagonal S, dsyevd reproduces the injected
// eigenvalue exactly, so s_min must match 1e-12 to relative 1e-6 — and it is
// still reported even though that direction is dropped (raw, pre-filter).
TEST(CanonicalOrthogonalization,
     DiagonalSminMatchesInjectedTinyEigenvalueRel1em6) {
    CanonicalOrth r(make_diagonal_s(), kTau);
    EXPECT_EQ(r.n_dropped, 1);
    EXPECT_NEAR(r.s_min, kTinyEig, 1e-6 * kTinyEig);
}

// ── tau <= 0 disables filtering ──────────────────────────────────────

// Edge case: tau = 0 keeps all n columns on a well-conditioned S, and the
// full n x n X still satisfies X^T S X = I_n to 1e-12. (The near-singular
// fixture cannot carry this tight bound: 1/sqrt(1e-12) amplifies the eigh
// residual to ~5e-5 — see file header — hence the healthy spectrum here.)
TEST(CanonicalOrthogonalization, TauZeroKeepsAllColumnsWellConditioned) {
    RealMatrix s = make_healthy_s();
    CanonicalOrth r(s, 0.0);
    ASSERT_EQ(r.X.shape(0), 4u);
    ASSERT_EQ(r.X.shape(1), 4u);
    EXPECT_EQ(r.n_dropped, 0);
    EXPECT_LT(max_abs_dev_from_identity(xt_s_x(s, r.X)), 1e-12);
}

// Edge case: tau = 0 keeps even the direction that the default threshold
// would drop — filtering is disabled, not merely thresholded at zero.
TEST(CanonicalOrthogonalization, TauZeroKeepsSubThresholdColumnsToo) {
    CanonicalOrth r(make_near_singular_s(), 0.0);
    EXPECT_EQ(r.X.shape(1), 4u);
    EXPECT_EQ(r.n_dropped, 0);
}

// ── all columns dropped ──────────────────────────────────────────────

// [ac-001] Edge case: tau above the largest eigenvalue (1.5) drops every
// column -> std::runtime_error (repo "no fallback values" convention).
TEST(CanonicalOrthogonalization, TauDroppingEveryColumnThrows) {
    RealMatrix s = make_near_singular_s();
    EXPECT_THROW(CanonicalOrth(s, 10.0), std::runtime_error);
}

// ── boundary: s_i == tau ─────────────────────────────────────────────

// Edge case: keep rule is s_i >= tau, so tau exactly equal to an eigenvalue
// KEEPS that column. Diagonal fixture: dsyevd returns 0.5 exactly, so with
// tau = 0.5 the comparison is exact — only the 1e-12 direction is dropped.
// (A strict '>' bug would drop 2 columns and yield a 4x2 X.)
TEST(CanonicalOrthogonalization, TauEqualToEigenvalueKeepsThatColumn) {
    CanonicalOrth r(make_diagonal_s(), 0.5);
    EXPECT_EQ(r.X.shape(1), 3u);
    EXPECT_EQ(r.n_dropped, 1);
}

// ── lowdin member (single-eigh optimization) ─────────────────────────

// Healthy basis: the lowdin member must be BITWISE identical to
// build_orthogonalization_matrix(S) — same eigendecomposition, same
// operations — so the SCF loop can pay one eigh per run() instead of two
// without moving any pre-existing result.
TEST(CanonicalOrthogonalization, LowdinBitwiseMatchesLegacyBuilder) {
    RealMatrix s = make_healthy_s();
    CanonicalOrth r(s, kTau);
    ASSERT_EQ(r.n_dropped, 0);
    RealMatrix legacy = ATV::cpu::linalg::build_orthogonalization_matrix(s);
    ASSERT_EQ(r.lowdin.shape(0), 4u);
    ASSERT_EQ(r.lowdin.shape(1), 4u);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(r.lowdin(i, j), legacy(i, j))
                << "lowdin differs from build_orthogonalization_matrix at ("
                << i << "," << j << ")";
}

// Filtered basis: no meaningful full-rank S^{-1/2} — lowdin stays empty.
TEST(CanonicalOrthogonalization, LowdinEmptyWhenColumnsDropped) {
    CanonicalOrth r(make_near_singular_s(), kTau);
    ASSERT_EQ(r.n_dropped, 1);
    EXPECT_EQ(r.lowdin.size(), 0u);
}

// ── purity / immutability ────────────────────────────────────────────

// Immutability: the input overlap matrix is read-only — bitwise unchanged
// after the call (the eigensolver must work on a copy, not in place).
TEST(CanonicalOrthogonalization, InputOverlapMatrixIsUnchanged) {
    RealMatrix s = make_near_singular_s();
    RealMatrix before = s;
    (void)CanonicalOrth(s, kTau);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(s(i, j), before(i, j))
                << "S mutated at (" << i << "," << j << ")";
}
