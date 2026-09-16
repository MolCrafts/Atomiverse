/// @file cpu/minimize/cg.h
/// @brief CG — Polak-Ribière conjugate-gradient Optimizer component.
///
/// Pluggable alternative to `cpu::minimizer::LBFGS` inside
/// `cpu::minimizer::Minimize`. Same outer contract: binds
/// `search_direction` (computes the step) and `convergence_check`
/// (publishes force / displacement / energy metrics onto `ConvergenceMetrics`,
/// where a `Criterion` — by default `cpu::ForceConverged` — judges them).
/// The component measures; it never declares convergence.
///
/// Algorithm (Polak-Ribière+):
///   d_0 = -g_0
///   For k ≥ 1:
///     β = max(0, (g_k · (g_k − g_{k−1})) / (g_{k−1} · g_{k−1}))
///     d_k = -g_k + β · d_{k−1}
///   x_{k+1} = x_k + α · d_k
///
/// Restart: when β = 0 (numerator non-positive), the direction collapses
/// back to steepest descent. No explicit periodic restart in this
/// implementation — the Polak-Ribière+ projection onto non-negative β
/// already handles ill-conditioned regions.
///
/// Convergence is not this class's business: `CG` and `LBFGS` publish the
/// identical `ConvergenceMetrics`, and one `Criterion` (`cpu::ForceConverged`)
/// judges both. That is what makes the optimizer hot-swap test
/// (`tests/cpu/test_minimize_driver_swap_optimizer.cpp`) meaningful — the two
/// classes are judged by the same criterion component, not by two copies of the
/// same five tolerances.
///
/// References:
///   Polak & Ribière, ESAIM Math. Modelling Numer. Anal. 3, 35 (1969)
///   Nocedal & Wright, Numerical Optimization, Ch. 5
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/minimize/lbfgs.h" // for OptRecord

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Polak-Ribière conjugate-gradient Optimizer.
class CG : public cpu::Optimizer {
  public:
    /// @brief Construct CG optimizer.
    ///
    /// Convergence tolerances are **not** arguments here — they belong to the
    /// `cpu::ForceConverged` criterion (`opt.add_criterion(...)`).
    /// @param alpha0    Step size [Å²/eV] (default 0.01).
    explicit CG(double alpha0 = 0.01);

    /// @brief Select the line-search policy used at the `line_search` phase.
    ///
    /// `StrongWolfe` (default) enforces N&W 3.7a+3.7b along the PR+ direction;
    /// a small `c2` (0.1) keeps the conjugate direction a descent direction
    /// (N&W §5.2). `Armijo` enforces only sufficient decrease; `Fixed`
    /// reproduces the legacy `x += alpha0 * d` step. Must be called before
    /// `opt.build()`.
    /// @param mode      Fixed / Armijo / StrongWolfe (default StrongWolfe).
    /// @param c1        Sufficient-decrease constant (default 1e-4).
    /// @param c2        Wolfe curvature constant (default 0.1 for CG).
    /// @param max_step  Max trial displacement [Å]; caps alpha (default 0.2).
    /// @param min_alpha Alpha floor (default 1e-8).
    /// @return Reference to self for chaining.
    CG& line_search(LineSearchMode mode = LineSearchMode::StrongWolfe,
                    double c1 = 1e-4, double c2 = 0.1, double max_step = 0.2,
                    double min_alpha = 1e-8);

    std::string name() const override { return "cg"; }
    /* role via Optimizer base */

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::line_search, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    double max_force() const { return max_force_; }
    const std::vector<OptRecord>& trace() const { return trace_; }

  private:
    double alpha0_;

    // Line search. Default StrongWolfe (c2=0.1 for CG); `line_search()` overrides.
    LineSearchMode ls_mode_ = LineSearchMode::StrongWolfe;
    double ls_c1_ = 1e-4;        // Sufficient-decrease constant.
    double ls_c2_ = 0.1;         // Wolfe curvature constant (small for CG).
    double ls_contract_ = 0.5;   // Armijo backtracking factor.
    double ls_max_step_ = 0.2;   // Max trial displacement [Å] (caps alpha).
    double ls_min_alpha_ = 1e-8; // Alpha floor.
    // Carry-over from search_direction → line_search within one step (the
    // current direction lives in d_prev_, the origin in x_prev_):
    double ls_phi0_ = 0;    // phi(0) = energy at x_prev_ [eV].
    double ls_gd0_ = 0;     // phi'(0) = g . d (< 0 on descent).
    double ls_dir_max_ = 0; // global max|d| component, for the cap.

    // Previous-step state.
    std::vector<double> x_prev_;
    std::vector<double> g_prev_;
    std::vector<double> d_prev_; ///< Previous CG search direction.
    bool has_prev_ = false;
    bool x_prev_valid_ = false; ///< False after migration reshuffles rows.
    double max_force_ = 0;
    double energy_prev_ = 0;
    int step_ = 0;

    // Per-step workspaces, sized once in build() — no launch() allocation.
    std::vector<double> x_ws_;
    std::vector<double> g_ws_;
    std::vector<double> d_ws_;

    /// @brief Last observed `dd->neighbor_generation`; a bump means atom
    /// migration reshuffled local rows → previous-step state is discarded.
    std::uint64_t last_generation_ = ~std::uint64_t{0};

    std::vector<OptRecord> trace_;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
