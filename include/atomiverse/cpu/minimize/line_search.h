/// @file cpu/minimize/line_search.h
/// @brief Backend-agnostic line-search numerics for the CPU optimizers.
///
/// Pure 1-D line-search routines shared by `cpu::minimizer::LBFGS` and
/// `cpu::minimizer::CG`. They are templated over a caller-supplied `phi`
/// functor that evaluates the 1-D restriction
///
///     phi(alpha) = f(x_k + alpha * p_k)
///
/// along a descent direction `p_k`, returning the objective value and
/// (on request) the directional derivative `phi'(alpha) = g(x_k + alpha
/// p_k) . p_k`. The functor owns everything backend-specific: writing the
/// trial geometry into the frame, triggering a force re-evaluation, the
/// owned-row partials, and — under MPI — the global reduction that makes
/// `phi`/`phi'` identical on every rank. The numerics here are therefore
/// domain-decomposition-correct by construction as long as the functor
/// reduces: this file performs no I/O, no MPI, and no allocation.
///
/// Conditions and algorithms follow Nocedal & Wright, *Numerical
/// Optimization* 2e (N&W): Armijo eq. 3.6a, strong-Wolfe eqs. 3.7a/3.7b,
/// backtracking Alg. 3.1, bracketing Alg. 3.5 + zoom Alg. 3.6.
#pragma once

#include <cmath>

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Which line-search policy an optimizer uses for its step.
enum class LineSearchMode {
    Fixed,      ///< Legacy fixed/SD-scaled step (no trial re-evaluation).
    Armijo,     ///< Backtracking that enforces sufficient decrease only.
    StrongWolfe ///< Bracketing + zoom enforcing the strong-Wolfe conditions.
};

/// @brief One sample of the 1-D line function `phi`.
struct LsSample {
    double f = 0;    ///< phi(alpha) = f(x_k + alpha p_k) [eV].
    double dphi = 0; ///< phi'(alpha) = g(x_k+alpha p_k).p_k [eV/Å·(dir units)];
                     ///< valid only when the sample was requested with slope.
};

/// @brief Outcome of a line search.
struct LineSearchResult {
    double alpha = 0;      ///< Accepted (or best) step length.
    bool ok = false;       ///< True when the target condition was satisfied.
    int n_value_evals = 0; ///< phi evaluations (energy).
    int n_grad_evals = 0;  ///< phi' evaluations (gradient/force).
};

/// @brief Backtracking line search enforcing the Armijo condition (N&W 3.6a).
///
/// Requires a descent direction: `dphi0 = phi'(0) < 0`. Starting from
/// `alpha_init`, contracts by `rho` until
/// `phi(alpha) <= phi(0) + c1*alpha*phi'(0)` or the trial floor `alpha_min`
/// / evaluation cap `max_evals` is reached. Each trial costs one energy
/// evaluation only (no gradient).
///
/// @param phi        Functor: `LsSample phi(double alpha, bool need_slope)`.
/// @param phi0       phi(0) = current energy [eV].
/// @param dphi0      phi'(0) = g_k . p_k (< 0 for a descent direction).
/// @param alpha_init First trial step length.
/// @param c1         Armijo constant in (0,1), typically 1e-4.
/// @param rho        Contraction factor in (0,1), typically 0.5.
/// @param max_evals  Maximum number of trial energy evaluations.
/// @param alpha_min  Smallest step length to try before giving up.
/// @return Accepted alpha with `ok=true`, or the last trial with `ok=false`.
template <class Phi>
LineSearchResult armijo_backtrack(Phi&& phi, double phi0, double dphi0,
                                  double alpha_init, double c1, double rho,
                                  int max_evals, double alpha_min) {
    LineSearchResult r;
    r.alpha = alpha_init;
    if (!(dphi0 < 0))
        return r; // not a descent direction — caller must reset p = -g
    double alpha = alpha_init;
    for (int k = 0; k < max_evals; ++k) {
        const LsSample s = phi(alpha, false);
        ++r.n_value_evals;
        if (s.f <= phi0 + c1 * alpha * dphi0) {
            r.alpha = alpha;
            r.ok = true;
            return r;
        }
        if (alpha <= alpha_min) {
            r.alpha = alpha;
            return r;
        }
        alpha *= rho;
        if (alpha < alpha_min)
            alpha = alpha_min;
    }
    r.alpha = alpha;
    return r;
}

namespace detail {

/// @brief N&W Algorithm 3.6 — zoom between a bracket [lo, hi] that is known
/// to contain a point satisfying the strong-Wolfe conditions. `f_lo` is
/// phi(alpha_lo); `alpha_lo` is the best sufficient-decrease point so far.
/// Uses bisection (a robust, derivative-free choice for the interpolation
/// step N&W leaves open). `budget` caps the remaining evaluations.
template <class Phi>
LineSearchResult zoom(Phi&& phi, double phi0, double dphi0, double c1,
                      double c2, double alpha_lo, double f_lo, double alpha_hi,
                      int budget, LineSearchResult r) {
    for (int k = 0; k < budget; ++k) {
        const double alpha_j = 0.5 * (alpha_lo + alpha_hi);
        const LsSample s = phi(alpha_j, true);
        ++r.n_value_evals;
        ++r.n_grad_evals;
        if (s.f > phi0 + c1 * alpha_j * dphi0 || s.f >= f_lo) {
            alpha_hi =
                alpha_j; // 3.6a violated or no improvement → shrink right
        } else {
            if (std::abs(s.dphi) <= -c2 * dphi0) { // strong-Wolfe curvature met
                r.alpha = alpha_j;
                r.ok = true;
                return r;
            }
            if (s.dphi * (alpha_hi - alpha_lo) >= 0)
                alpha_hi = alpha_lo;
            alpha_lo = alpha_j;
            f_lo = s.f;
        }
        if (std::abs(alpha_hi - alpha_lo) < 1e-14)
            break;
    }
    // Budget/precision exhausted: return the best sufficient-decrease point.
    r.alpha = alpha_lo;
    return r;
}

} // namespace detail

/// @brief Strong-Wolfe line search (N&W Algorithm 3.5 bracketing + 3.6 zoom).
///
/// Returns alpha satisfying both the sufficient-decrease condition (3.7a)
/// `phi(alpha) <= phi(0) + c1*alpha*phi'(0)` and the curvature condition
/// (3.7b) `|phi'(alpha)| <= c2*|phi'(0)|`, with `0 < c1 < c2 < 1`. Each
/// trial costs one energy + one gradient evaluation. For quasi-Newton
/// directions use `c2 = 0.9`; for nonlinear-CG use `c2 = 0.1`.
///
/// Requires a descent direction `dphi0 < 0`; returns `ok=false` otherwise so
/// the caller can reset `p = -g`.
///
/// @param phi        Functor: `LsSample phi(double alpha, bool need_slope)`.
/// @param phi0       phi(0) [eV].
/// @param dphi0      phi'(0) = g_k . p_k (< 0).
/// @param alpha_init First trial (1.0 for L-BFGS on p=-H g).
/// @param c1,c2      Wolfe constants, 0 < c1 < c2 < 1.
/// @param max_evals  Cap on total energy/gradient trials.
/// @param alpha_max  Upper bound on the step length during bracketing.
/// @return Accepted alpha with `ok=true`, or best-so-far with `ok=false`.
template <class Phi>
LineSearchResult strong_wolfe(Phi&& phi, double phi0, double dphi0,
                              double alpha_init, double c1, double c2,
                              int max_evals, double alpha_max) {
    LineSearchResult r;
    r.alpha = alpha_init;
    if (!(dphi0 < 0))
        return r; // not a descent direction — caller must reset p = -g

    double alpha_prev = 0.0;
    double f_prev = phi0;
    double alpha = (alpha_init < alpha_max) ? alpha_init : 0.5 * alpha_max;
    bool first = true;

    while (r.n_value_evals < max_evals) {
        const LsSample s = phi(alpha, true);
        ++r.n_value_evals;
        ++r.n_grad_evals;

        // 3.7a fails, or phi did not decrease vs the previous bracket end →
        // the minimizer is between alpha_prev and alpha.
        if (s.f > phi0 + c1 * alpha * dphi0 || (!first && s.f >= f_prev)) {
            return detail::zoom(phi, phi0, dphi0, c1, c2, alpha_prev, f_prev,
                                alpha, max_evals - r.n_value_evals, r);
        }
        // 3.7b curvature satisfied → done.
        if (std::abs(s.dphi) <= -c2 * dphi0) {
            r.alpha = alpha;
            r.ok = true;
            return r;
        }
        // Slope turned non-negative → minimizer is between alpha and alpha_prev.
        if (s.dphi >= 0) {
            return detail::zoom(phi, phi0, dphi0, c1, c2, alpha, s.f,
                                alpha_prev, max_evals - r.n_value_evals, r);
        }
        // Still descending strongly: expand the step toward alpha_max.
        alpha_prev = alpha;
        f_prev = s.f;
        first = false;
        if (alpha >= alpha_max)
            break;
        alpha = std::min(2.0 * alpha, alpha_max);
    }
    // Bracketing budget exhausted; return the last admissible step.
    r.alpha = alpha_prev > 0 ? alpha_prev : alpha;
    return r;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
