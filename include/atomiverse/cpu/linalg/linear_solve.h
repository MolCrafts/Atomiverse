/// @file cpu/linalg/linear_solve.h
/// @brief Shared dense iterative linear solvers for AO-basis electronic
///        structure: the Jacobi-preconditioned conjugate gradient used by the
///        CPHF / CP-KS (coupled-perturbed) linear-response equations.
///
/// Method-agnostic — a symmetric-positive-definite solver that takes only the
/// action of A (a matvec closure) plus its diagonal preconditioner, so CPHF
/// and UCPHF share one implementation instead of each carrying a private copy.
/// Header-inline free math with no owning type, written in the explicit /
/// throw-on-failure style of the sibling `eigensolve.h`.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace linalg {

/// @brief Solve A x = b for symmetric-positive-definite A by Jacobi-
///        preconditioned conjugate gradient, given only the action of A.
///
/// Starts from x = 0 (so the initial residual is exactly b), applies the
/// diagonal (Jacobi) preconditioner z = r / diag element-wise, and iterates
/// the standard PCG recurrence until the residual 2-norm drops below
/// tol · ||b||₂ (relative convergence) or `max_iter` iterations elapse. There
/// is no fallback: failing to converge throws (the non-convergence contract
/// inherited from CphfPack::solve).
///
/// @tparam MatVec Callable `RealVector(const RealVector&)` returning A·p.
/// @param matvec   The action of A on a vector (matrix-dependent units).
/// @param diag     Diagonal of A used as the Jacobi preconditioner
///                 (matrix-dependent units); same length as b, no zero entries.
/// @param b        Right-hand side vector (matrix-dependent units).
/// @param tol      Relative convergence threshold on ||r||₂ / ||b||₂
///                 (dimensionless).
/// @param max_iter Maximum CG iterations (dimensionless count).
/// @return Solution x (matrix-dependent units), length b.size().
/// @throws std::runtime_error if not converged within max_iter — no fallback.
template <typename MatVec>
RealVector solve_pcg(MatVec&& matvec, const RealVector& diag,
                     const RealVector& b, double tol, int max_iter) {
    const std::size_t n = b.size();
    RealVector x = xt::zeros<double>({n});
    RealVector r = b; // r0 = b - A·0 = b
    RealVector z = xt::zeros<double>({n});
    RealVector p = xt::zeros<double>({n});

    double rz = 0.0;
    double b_norm_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        z(i) = r(i) / diag(i);
        p(i) = z(i);
        rz += r(i) * z(i);
        b_norm_sq += b(i) * b(i);
    }
    const double b_norm = std::sqrt(b_norm_sq);
    const double threshold = tol * (b_norm > 0.0 ? b_norm : 1.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        RealVector Ap = matvec(p);

        double pAp = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            pAp += p(i) * Ap(i);
        const double alpha = rz / pAp;

        double r_norm_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x(i) += alpha * p(i);
            r(i) -= alpha * Ap(i);
            r_norm_sq += r(i) * r(i);
        }
        if (std::sqrt(r_norm_sq) <= threshold)
            return x;

        double rz_new = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            z(i) = r(i) / diag(i);
            rz_new += r(i) * z(i);
        }
        const double beta = rz_new / rz;
        rz = rz_new;

        for (std::size_t i = 0; i < n; ++i)
            p(i) = z(i) + beta * p(i);
    }

    throw std::runtime_error(
        "solve_pcg: preconditioned CG did not converge within max_iter="
        + std::to_string(max_iter) + " iterations — no fallback");
}

} // namespace linalg
} // namespace cpu
} // namespace ATV
