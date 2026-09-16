/// @file tests/cpu/linalg/test_eigensolve.cpp
/// @brief Unit tests for the two NEW shared dense-linalg primitives added to
///        `atomiverse/cpu/linalg/eigensolve.h` by spec scf-review-01-linalg:
///        `ATV::cpu::linalg::sym_power` and `ATV::cpu::linalg::eig_nonsymmetric`.
///
/// Pins the current contract of both primitives (written RED-first for spec
/// scf-review-01-linalg; the contracts below are the delivered API).
///
/// ── Contract under test ──────────────────────────────────────────────────
///
///   /// Symmetric matrix power M^p = U λ^p Uᵀ for real symmetric M = U λ Uᵀ.
///   /// `floor` clamps each eigenvalue from below (λ_eff = max(λ, floor))
///   /// before raising to `p`, so p < 0 cannot amplify a near-null direction.
///   /// `label` is folded into any diagnostic message.
///   RealMatrix sym_power(const RealMatrix& M, double p, double floor = 0.0,
///                        std::string_view label = "sym_power");
///
///   /// Nonsymmetric real eigenproblem (LAPACK dgeev), explicit-workspace /
///   /// throw-on-info, no LWORK=-1 query — the same pattern dsygvd_ uses in
///   /// this file. Eigenvalues land in (wr, wi) [both resized to n]. When
///   /// `vr != nullptr`, right eigenvectors land column-wise in *vr (n×n,
///   /// column-major): for a real eigenvalue λ_j, column j is its eigenvector;
///   /// for a conjugate pair, columns j / j+1 hold the LAPACK real/imag parts.
///   /// Passing nullptr requests JOBVR='N' (eigenvalues only). `label` feeds
///   /// the throw message. Input A is read-only (dgeev works on a copy).
///   void eig_nonsymmetric(const RealMatrix& A, RealVector& wr, RealVector& wi,
///                         RealMatrix* vr = nullptr,
///                         std::string_view label = "eig_nonsymmetric");
///
/// Fixtures are exact hard-coded literals (SPD 2×2 from a 3-4-5 Givens rotation,
/// upper-triangular and skew 2×2 for the nonsymmetric spectra). Tolerances:
/// sym_power identities ≤1e-12 (LAPACK dsyevd is bitwise-exact on these 2×2s);
/// eig eigenvalue / eigenvector residuals ≤1e-12.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "atomiverse/cpu/linalg/eigensolve.h"

namespace linalg = ATV::cpu::linalg;
using linalg::RealMatrix;
using linalg::RealVector;

namespace {

/// 2×2 matrix from row-major literals (0,0)=a (0,1)=b (1,0)=c (1,1)=d.
RealMatrix mat2(double a, double b, double c, double d) {
    RealMatrix m = xt::zeros<double>({std::size_t(2), std::size_t(2)});
    m(0, 0) = a;
    m(0, 1) = b;
    m(1, 0) = c;
    m(1, 1) = d;
    return m;
}

/// max_ij |A_ij - B_ij| over equal-shape 2×2 matrices.
double max_abs_diff(const RealMatrix& a, const RealMatrix& b) {
    double dev = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            dev = std::max(dev, std::abs(a(i, j) - b(i, j)));
    return dev;
}

/// max_ij |M_ij - δ_ij| for a 2×2.
double max_dev_from_identity(const RealMatrix& m) {
    double dev = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            dev = std::max(dev, std::abs(m(i, j) - (i == j ? 1.0 : 0.0)));
    return dev;
}

/// SPD S = U diag(4,9) Uᵀ built from a hard-coded 3-4-5 Givens rotation
/// (c=0.8, s=0.6 ⇒ c²+s²=1). Eigenvalues are exactly {4,9}; trace 13, det 36.
RealMatrix rotated_spd_4_9() {
    RealMatrix u = mat2(0.8, -0.6, 0.6, 0.8);
    RealMatrix d = mat2(4.0, 0.0, 0.0, 9.0);
    RealMatrix ut(xt::transpose(u));
    RealMatrix ud(xt::linalg::dot(u, d));
    RealMatrix s(xt::linalg::dot(ud, ut));
    // Symmetrize away any last-bit asymmetry before the eigensolver sees it.
    const double off = 0.5 * (s(0, 1) + s(1, 0));
    s(0, 1) = off;
    s(1, 0) = off;
    return s;
}

} // namespace

// ── TestEigensolve: sym_power ────────────────────────────────────────────

// [Basics] Diagonal SPD S=diag(4,9): sym_power(S,0.5) squared reproduces S.
TEST(TestEigensolve, SymPowerHalfSquaredEqualsDiagonalS) {
    RealMatrix s = mat2(4.0, 0.0, 0.0, 9.0);
    RealMatrix root = linalg::sym_power(s, 0.5);
    RealMatrix sq(xt::linalg::dot(root, root));
    EXPECT_LT(max_abs_diff(sq, s), 1e-12)
        << "sym_power(S,0.5)^2 must reconstruct S";
}

// [Basics] Diagonal SPD S=diag(4,9): sym_power(S,-0.5) == diag(1/2, 1/3).
TEST(TestEigensolve, SymPowerNegHalfEqualsInverseRootDiagonal) {
    RealMatrix s = mat2(4.0, 0.0, 0.0, 9.0);
    RealMatrix inv_root = linalg::sym_power(s, -0.5);
    RealMatrix expected = mat2(0.5, 0.0, 0.0, 1.0 / 3.0);
    EXPECT_LT(max_abs_diff(inv_root, expected), 1e-12)
        << "sym_power(diag(4,9),-0.5) must be diag(1/2, 1/3)";
}

// [Basics] Rotated SPD (same spectrum, off-diagonal): sym_power(S,0.5)^2 == S.
TEST(TestEigensolve, SymPowerHalfSquaredEqualsRotatedS) {
    RealMatrix s = rotated_spd_4_9();
    RealMatrix root = linalg::sym_power(s, 0.5);
    RealMatrix sq(xt::linalg::dot(root, root));
    EXPECT_LT(max_abs_diff(sq, s), 1e-12)
        << "sym_power(S,0.5)^2 must reconstruct the rotated S";
}

// [Basics] Rotated SPD: sym_power(S,-0.5)·sym_power(S,0.5) == I (the
// inverse-square-root identity, in operator form so no S^{-1/2} literal needed).
TEST(TestEigensolve, SymPowerNegHalfTimesHalfIsIdentityRotated) {
    RealMatrix s = rotated_spd_4_9();
    RealMatrix root = linalg::sym_power(s, 0.5);
    RealMatrix inv_root = linalg::sym_power(s, -0.5);
    RealMatrix prod(xt::linalg::dot(inv_root, root));
    EXPECT_LT(max_dev_from_identity(prod), 1e-12)
        << "sym_power(S,-0.5)·sym_power(S,0.5) must be the identity";
}

// [Edge] Ill-conditioned S=diag(1, 1e-14) with p=-0.5 and floor=1e-10: the
// floor clamps the near-null direction so every entry stays finite (no
// inf/nan), and the healthy 1.0 direction is untouched (1.0^-0.5 == 1.0).
TEST(TestEigensolve, SymPowerFloorKeepsAllEntriesFinite) {
    RealMatrix s = mat2(1.0, 0.0, 0.0, 1e-14);
    RealMatrix out = linalg::sym_power(s, -0.5, 1e-10);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_TRUE(std::isfinite(out(i, j)))
                << "sym_power with floor produced non-finite entry at (" << i
                << "," << j << ")";
    EXPECT_NEAR(out(0, 0), 1.0, 1e-12)
        << "healthy eigenvalue 1.0 must be unaffected by the floor";
}

// ── TestEigensolve: eig_nonsymmetric ─────────────────────────────────────

// [Basics] Upper-triangular A=[[2,1],[0,3]] → real eigenvalues {2,3}, zero imag.
TEST(TestEigensolve, EigNonsymmetricUpperTriangularRealSpectrum) {
    RealMatrix a = mat2(2.0, 1.0, 0.0, 3.0);
    RealVector wr;
    RealVector wi;
    linalg::eig_nonsymmetric(a, wr, wi);
    ASSERT_EQ(wr.size(), std::size_t(2));
    ASSERT_EQ(wi.size(), std::size_t(2));
    EXPECT_NEAR(std::min(wr(0), wr(1)), 2.0, 1e-12);
    EXPECT_NEAR(std::max(wr(0), wr(1)), 3.0, 1e-12);
    EXPECT_NEAR(wi(0), 0.0, 1e-12);
    EXPECT_NEAR(wi(1), 0.0, 1e-12);
}

// [Edge] Skew A=[[0,1],[-1,0]] → conjugate pair ±i (real parts 0, imag ±1).
TEST(TestEigensolve, EigNonsymmetricSkewConjugatePair) {
    RealMatrix a = mat2(0.0, 1.0, -1.0, 0.0);
    RealVector wr;
    RealVector wi;
    linalg::eig_nonsymmetric(a, wr, wi);
    ASSERT_EQ(wr.size(), std::size_t(2));
    ASSERT_EQ(wi.size(), std::size_t(2));
    EXPECT_NEAR(wr(0), 0.0, 1e-12);
    EXPECT_NEAR(wr(1), 0.0, 1e-12);
    EXPECT_NEAR(std::min(wi(0), wi(1)), -1.0, 1e-12);
    EXPECT_NEAR(std::max(wi(0), wi(1)), 1.0, 1e-12);
}

// [Basics] Right eigenvectors: for the real case A·v_j == λ_j·v_j per column.
TEST(TestEigensolve, EigNonsymmetricRightEigenvectorResidual) {
    RealMatrix a = mat2(2.0, 1.0, 0.0, 3.0);
    RealVector wr;
    RealVector wi;
    RealMatrix vr;
    linalg::eig_nonsymmetric(a, wr, wi, &vr);
    ASSERT_EQ(vr.shape(0), std::size_t(2));
    ASSERT_EQ(vr.shape(1), std::size_t(2));
    for (std::size_t j = 0; j < 2; ++j) {
        ASSERT_NEAR(wi(j), 0.0, 1e-12) << "expected real eigenvalue at " << j;
        const double v0 = vr(0, j);
        const double v1 = vr(1, j);
        // A·v (row-major A literals) vs λ·v.
        const double av0 = 2.0 * v0 + 1.0 * v1;
        const double av1 = 0.0 * v0 + 3.0 * v1;
        EXPECT_NEAR(av0, wr(j) * v0, 1e-12) << "A·v != λ·v (row 0), col " << j;
        EXPECT_NEAR(av1, wr(j) * v1, 1e-12) << "A·v != λ·v (row 1), col " << j;
    }
}

// [Immutability] dgeev destroys its matrix argument; the wrapper must copy, so
// the caller's A is bitwise unchanged after the call.
TEST(TestEigensolve, EigNonsymmetricLeavesInputUnchanged) {
    RealMatrix a = mat2(2.0, 1.0, 0.0, 3.0);
    RealMatrix before = a;
    RealVector wr;
    RealVector wi;
    linalg::eig_nonsymmetric(a, wr, wi);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_EQ(a(i, j), before(i, j))
                << "A mutated at (" << i << "," << j << ")";
}
