/// @file cuda/minimizer/cg.cuh
/// @brief cuda::minimizer::CG — Polak-Ribière+ conjugate-gradient
///        Optimizer component.
///
/// CUDA mirror of `cpu::minimizer::CG`: a hot-swappable alternative to
/// `cuda::minimizer::LBFGS` / `SteepestDescent` inside
/// `cuda::minimizer::Minimize`. Same outer contract as the CPU class —
/// binds `search_direction` (computes and applies the step) and
/// `convergence_check` (PUBLISHES its metrics into `ctx.runtime.loop.metrics`;
/// a `cuda::LoopHandler` judges them — the component decides nothing).
///
/// Algorithm (Polak-Ribière+):
///   d_0 = -g_0
///   For k ≥ 1:
///     β = max(0, (g_k · (g_k − g_{k−1})) / (g_{k−1} · g_{k−1}))
///     d_k = -g_k + β · d_{k−1}
///   x_{k+1} = x_k + α · d_k
///
/// Execution-model classification: the `search_direction` phase is
/// **GPU-pure compute** — β is produced and consumed entirely on the
/// device (a fused dot-product reduction feeds a single-thread β kernel
/// whose result the step kernel reads by pointer, the same
/// device-scalar chaining as `cuda::md::NoseHoover`'s scale factor).
/// The `convergence_check` phase is the shared `ConvCheck`
/// host-dependent bridge (one pinned D2H + sync per iteration).
///
/// **Tolerances are not here.** They belong to the termination policy —
/// `cuda::ForceConverged`, which `cuda::minimizer::Minimize` attaches by default and
/// which the user replaces with `min.add_handler(cuda::ForceConverged{ftol, …})`. This
/// component measures; it does not decide. That is why it no longer carries five of its
/// six constructor arguments, and why `CG::converged()` is gone: ask the loop
/// (`min.converged()`), which is the thing that actually knows.
///
/// @note Like its CPU mirror, CG applies a FIXED step `x += α₀·d` with
/// no per-atom displacement cap (`LBFGS` caps at 0.1 Å, CG does not).
/// α₀ must therefore be sized to the stiffness of the landscape:
/// `α₀ · |F|_max` should stay well below the shortest interaction
/// length scale, or the trajectory diverges. For stiff starts prefer
/// `LBFGS` or shrink α₀.
///
/// References:
///   Polak & Ribière, ESAIM Math. Modelling Numer. Anal. 3, 35 (1969)
///   Nocedal & Wright, Numerical Optimization, Ch. 5
///
/// @see cuda::minimizer::Minimize for the driver this component plugs into
/// @see cpu::minimizer::CG for the CPU mirror
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/minimizer/conv_check.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief Polak-Ribière+ conjugate-gradient Optimizer.
///
/// Optimizer state (previous positions / gradient / direction) lives in
/// planar 3N `work_t` device buffers allocated in `build()` and released
/// in `destroy()` — no allocation in `launch()`. Buffers are zeroed at
/// build, which reproduces the CPU step-0 behaviour exactly: forces are
/// zero at step 0, `g_prev · g_prev = 0` forces β = 0, and the first
/// moving step is plain steepest descent.
class CG : public Component {
  public:
    /// @brief Construct CG optimizer.
    /// @param alpha0    Step size [Angstrom^2/eV] (default 0.01).
    explicit CG(double alpha0 = 0.01) : alpha0_(alpha0) {}

    /// @brief Component name.
    /// @return "cg".
    std::string name() const override { return "cg"; }
    Role role() const override { return 0; }

    /// @brief Structural role of this component.
    /// @return ComponentRole::Optimizer.
    /* role via Optimizer base */

    /// @brief Default bindings: search_direction + convergence_check.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    /// @brief Allocate + zero the optimizer state buffers.
    void build(const BuildContext& ctx) override;

    /// @brief Dispatch the bound phase (search_direction / convergence_check).
    void launch(const LaunchContext& ctx) override;

    /// @brief Release all device buffers.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Max force magnitude at the latest convergence check.
    double max_force() const { return conv_.max_force(); }

    /// @brief Per-step optimization trace.
    const std::vector<OptRecord>& trace() const { return trace_; }

  private:
    double alpha0_; ///< Step size [Angstrom^2/eV].

    int n_ = 0;    ///< Atom count.
    int step_ = 0; ///< Optimizer step counter.

    // Planar [x:0..n, y:n..2n, z:2n..3n] device state, work_t precision
    // (position/gradient differences of nearly-equal values would lose
    // too much precision in real_t storage).
    work_t* d_xprev_ = nullptr; ///< Pre-step positions (3N).
    work_t* d_gprev_ = nullptr; ///< Previous gradient (3N).
    work_t* d_dprev_ = nullptr; ///< Previous search direction (3N).
    work_t* d_pr_ = nullptr;    ///< Device {num, den, beta} scalars.

    ConvCheck conv_;
    std::vector<OptRecord> trace_;
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
