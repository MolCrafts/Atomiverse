/// @file tests/cpu/linalg/test_linear_solve.cpp
/// @brief Unit tests for the NEW shared preconditioned conjugate-gradient
///        solver `ATV::cpu::linalg::solve_pcg`, added in the NEW header
///        `atomiverse/cpu/linalg/linear_solve.h` by spec scf-review-01-linalg.
///
/// Pins the current contract of the solver (written RED-first for spec
/// scf-review-01-linalg; the contract below is the delivered API).
///
/// ── Contract under test ──────────────────────────────────────────────────
///
///   /// Solve A x = b for symmetric-positive-definite A given only its action
///   /// (`matvec`: callable RealVector(const RealVector&) returning A·p) and a
///   /// Jacobi preconditioner `diag` (the diagonal of A; applied as the
///   /// element-wise division r/diag). Iterates from x = 0 to relative
///   /// residual `tol`, at most `max_iter` steps. Throws std::runtime_error
///   /// when it does not converge within `max_iter` (the non-convergence
///   /// contract inherited from CphfPack::solve — "no fallback").
///   template <typename MatVec>
///   RealVector solve_pcg(MatVec&& matvec, const RealVector& diag,
///                        const RealVector& b, double tol, int max_iter);
///
/// Fixtures are exact hard-coded literals. SPD A=[[4,1],[1,3]] (det 11), b=[1,2]
/// ⇒ x = A⁻¹b = [ (3·1−1·2)/11, (4·2−1·1)/11 ] = [1/11, 7/11]. Tolerance ≤1e-10
/// on the solution; CG on a 2×2 SPD reaches machine precision in 2 iterations.

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

#include "atomiverse/cpu/linalg/linear_solve.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace linalg = ATV::cpu::linalg;
using linalg::RealVector;

namespace {

/// Action of the SPD test matrix A=[[4,1],[1,3]] on a 2-vector p.
RealVector apply_a(const RealVector& p) {
    RealVector out = {4.0 * p(0) + 1.0 * p(1), 1.0 * p(0) + 3.0 * p(1)};
    return out;
}

} // namespace

// ── TestLinearSolve: solve_pcg ───────────────────────────────────────────

// [Basics] SPD A=[[4,1],[1,3]], b=[1,2] converges within max_iter=50 to the
// exact hard-coded solution x=[1/11, 7/11] to ≤1e-10.
TEST(TestLinearSolve, SolvePcgMatchesExactSpdSolution) {
    RealVector diag = {4.0, 3.0}; // Jacobi preconditioner = diagonal of A.
    RealVector b = {1.0, 2.0};
    RealVector x = linalg::solve_pcg(apply_a, diag, b, /*tol=*/1e-12,
                                     /*max_iter=*/50);
    ASSERT_EQ(x.size(), std::size_t(2));
    EXPECT_NEAR(x(0), 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(x(1), 7.0 / 11.0, 1e-10);
}

// [Edge] max_iter too small (1 step on a system that needs 2) must throw
// std::runtime_error — the CphfPack non-convergence contract, no fallback.
TEST(TestLinearSolve, SolvePcgThrowsWhenMaxIterTooSmall) {
    RealVector diag = {4.0, 3.0};
    RealVector b = {1.0, 2.0};
    EXPECT_THROW((void)linalg::solve_pcg(apply_a, diag, b, /*tol=*/1e-12,
                                         /*max_iter=*/1),
                 std::runtime_error);
}
