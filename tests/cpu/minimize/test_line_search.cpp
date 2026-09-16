/// @file tests/cpu/minimizer/test_line_search.cpp
/// @brief Unit tests for the backend-agnostic line-search numerics
///        (cpu/minimizer/line_search.h) on an analytic convex quadratic.
///
/// f(x) = 1/2 x^T A x - b^T x,  grad f(x) = A x - b.  A is SPD, so f is
/// strictly convex with a unique minimizer x* = A^{-1} b. Along a descent
/// direction p the 1-D restriction phi(alpha) = f(x0 + alpha p) is a convex
/// parabola, the cleanest setting to verify the Armijo (N&W 3.6a) and
/// strong-Wolfe (N&W 3.7a/3.7b) conditions on the returned step.

#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/minimize/line_search.h"

using namespace ATV::cpu::minimizer;

namespace {

using Vec = std::array<double, 3>;

// SPD matrix with a spread of eigenvalues (well-conditioned but not trivial).
constexpr std::array<std::array<double, 3>, 3> kA = {{
    {{4.0, 1.0, 0.5}},
    {{1.0, 3.0, 0.3}},
    {{0.5, 0.3, 2.0}},
}};
constexpr Vec kB = {{1.0, -2.0, 0.5}};

Vec matvec(const Vec& x) {
    Vec r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i] += kA[i][j] * x[j];
    return r;
}

double dot(const Vec& a, const Vec& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double energy(const Vec& x) {
    const Vec Ax = matvec(x);
    return 0.5 * dot(x, Ax) - dot(kB, x);
}

Vec grad(const Vec& x) {
    const Vec Ax = matvec(x);
    return {{Ax[0] - kB[0], Ax[1] - kB[1], Ax[2] - kB[2]}};
}

// Build the phi functor for objective f along x0 + alpha*p.
struct Phi {
    Vec x0, p;
    LsSample operator()(double alpha, bool /*need_slope*/) const {
        const Vec xt = {
            {x0[0] + alpha * p[0], x0[1] + alpha * p[1], x0[2] + alpha * p[2]}};
        LsSample s;
        s.f = energy(xt);
        s.dphi = dot(grad(xt), p);
        return s;
    }
};

} // namespace

// ac-001: strong-Wolfe returns alpha satisfying BOTH 3.7a and 3.7b.
TEST(LineSearch, StrongWolfeSatisfiesBothConditions) {
    const Vec x0 = {{3.0, 3.0, 3.0}};
    const Vec g0 = grad(x0);
    const Vec p = {{-g0[0], -g0[1], -g0[2]}}; // steepest descent
    const double phi0 = energy(x0);
    const double dphi0 = dot(g0, p); // < 0
    ASSERT_LT(dphi0, 0.0);

    const double c1 = 1e-4, c2 = 0.9;
    auto r = strong_wolfe(Phi{x0, p}, phi0, dphi0, /*alpha_init=*/1.0, c1, c2,
                          /*max_evals=*/30, /*alpha_max=*/10.0);
    ASSERT_TRUE(r.ok);
    ASSERT_GT(r.alpha, 0.0);

    // Recompute phi and phi' at the accepted step and assert both conditions.
    const Phi phi{x0, p};
    const LsSample at = phi(r.alpha, true);
    EXPECT_LE(at.f, phi0 + c1 * r.alpha * dphi0);              // 3.7a
    EXPECT_LE(std::abs(at.dphi), c2 * std::abs(dphi0) + 1e-9); // 3.7b
    EXPECT_GT(r.n_grad_evals, 0);
}

// ac-008 numerics: strong-Wolfe with the CG constant c2 = 0.1 also holds.
TEST(LineSearch, StrongWolfeCGConstant) {
    const Vec x0 = {{-2.0, 5.0, 1.0}};
    const Vec g0 = grad(x0);
    const Vec p = {{-g0[0], -g0[1], -g0[2]}};
    const double phi0 = energy(x0);
    const double dphi0 = dot(g0, p);
    ASSERT_LT(dphi0, 0.0);

    const double c1 = 1e-4, c2 = 0.1;
    auto r = strong_wolfe(Phi{x0, p}, phi0, dphi0, 1.0, c1, c2, 40, 10.0);
    ASSERT_TRUE(r.ok);
    const Phi phi{x0, p};
    const LsSample at = phi(r.alpha, true);
    EXPECT_LE(at.f, phi0 + c1 * r.alpha * dphi0);
    EXPECT_LE(std::abs(at.dphi), c2 * std::abs(dphi0) + 1e-9);
}

// ac-004: a strong-Wolfe step yields a positive curvature pair yᵀs > 0,
// which is what keeps the L-BFGS inverse-Hessian update positive definite.
TEST(LineSearch, StrongWolfeGivesPositiveCurvaturePair) {
    const Vec x0 = {{4.0, -3.0, 2.0}};
    const Vec g0 = grad(x0);
    const Vec p = {{-g0[0], -g0[1], -g0[2]}};
    const double phi0 = energy(x0);
    const double dphi0 = dot(g0, p);

    auto r = strong_wolfe(Phi{x0, p}, phi0, dphi0, 1.0, 1e-4, 0.9, 30, 10.0);
    ASSERT_TRUE(r.ok);

    // s = alpha*p, y = grad(x0 + s) - grad(x0); assert yᵀs > 0.
    const Vec s = {{r.alpha * p[0], r.alpha * p[1], r.alpha * p[2]}};
    const Vec x1 = {{x0[0] + s[0], x0[1] + s[1], x0[2] + s[2]}};
    const Vec g1 = grad(x1);
    const Vec y = {{g1[0] - g0[0], g1[1] - g0[1], g1[2] - g0[2]}};
    EXPECT_GT(dot(y, s), 0.0);
}

// ac-002: Armijo backtracking returns alpha satisfying 3.6a.
TEST(LineSearch, ArmijoSatisfiesSufficientDecrease) {
    const Vec x0 = {{3.0, 3.0, 3.0}};
    const Vec g0 = grad(x0);
    const Vec p = {{-g0[0], -g0[1], -g0[2]}};
    const double phi0 = energy(x0);
    const double dphi0 = dot(g0, p);

    const double c1 = 1e-4;
    auto r = armijo_backtrack(Phi{x0, p}, phi0, dphi0, /*alpha_init=*/1.0, c1,
                              /*rho=*/0.5, /*max_evals=*/30,
                              /*alpha_min=*/1e-8);
    ASSERT_TRUE(r.ok);
    const Phi phi{x0, p};
    const LsSample at = phi(r.alpha, false);
    EXPECT_LE(at.f, phi0 + c1 * r.alpha * dphi0); // 3.6a
    EXPECT_EQ(r.n_grad_evals, 0);                 // Armijo is energy-only
}

// Non-descent direction is rejected (ok=false) so the caller can reset p=-g.
TEST(LineSearch, RejectsNonDescentDirection) {
    const Vec x0 = {{3.0, 3.0, 3.0}};
    const Vec g0 = grad(x0);
    const Vec p = {{g0[0], g0[1], g0[2]}}; // uphill: dphi0 > 0
    const double phi0 = energy(x0);
    const double dphi0 = dot(g0, p);
    ASSERT_GT(dphi0, 0.0);

    auto rsw = strong_wolfe(Phi{x0, p}, phi0, dphi0, 1.0, 1e-4, 0.9, 30, 10.0);
    EXPECT_FALSE(rsw.ok);
    auto rar =
        armijo_backtrack(Phi{x0, p}, phi0, dphi0, 1.0, 1e-4, 0.5, 30, 1e-8);
    EXPECT_FALSE(rar.ok);
}

// Full minimization driven by repeated strong-Wolfe steps reaches x* = A^{-1}b.
TEST(LineSearch, SteepestDescentWithWolfeConverges) {
    Vec x = {{5.0, -5.0, 5.0}};
    for (int it = 0; it < 200; ++it) {
        const Vec g = grad(x);
        const double gnorm2 = dot(g, g);
        if (std::sqrt(gnorm2) < 1e-6)
            break;
        const Vec p = {{-g[0], -g[1], -g[2]}};
        const double phi0 = energy(x);
        const double dphi0 = -gnorm2;
        auto r = strong_wolfe(Phi{x, p}, phi0, dphi0, 1.0, 1e-4, 0.9, 30, 10.0);
        // ok=false only when the objective is flat to f64 precision (gradient
        // already negligible) — a graceful "no further progress" stop, which
        // a real optimizer reaches well after its force-tolerance convergence.
        if (!r.ok)
            break;
        x = {{x[0] + r.alpha * p[0], x[1] + r.alpha * p[1],
              x[2] + r.alpha * p[2]}};
    }
    const Vec g = grad(x);
    EXPECT_LT(std::sqrt(dot(g, g)), 1e-6); // converged to the unique minimizer
}
