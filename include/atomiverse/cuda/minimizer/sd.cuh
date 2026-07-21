/// @file cuda/minimizer/sd.cuh
/// @brief cuda::minimizer::SteepestDescent — steepest-descent Optimizer
///        component.
///
/// `SteepestDescent` is an `Optimizer`-role component that moves atoms
/// along the force (negative gradient) direction each iteration:
/// `x += alpha * f / m * kFtm2v`. It is hot-swappable: it plugs into the
/// single `cuda::minimizer::Minimize` driver exactly as the CPU
/// `cpu::minimizer::LBFGS` / `CG` Optimizer components plug into
/// `cpu::minimizer::Minimize`.
///
/// Phase sequence per iteration (driven by `cuda::minimizer::Minimize`):
///   search_direction (SD step) -> line_search -> force ->
///   convergence_check -> end_of_step
///
/// At step 0 forces are zero (from initialization), so no movement
/// occurs. From step 1 onward, atoms move downhill using forces from the
/// previous iteration.
///
/// ## It measures, it does not decide
///
/// At `convergence_check` it runs the shared `ConvCheck` reduction and PUBLISHES the
/// force + energy metrics into `ctx.runtime.loop.metrics`, where the loop's
/// `cuda::LoopHandler`s judge them. It keeps no previous geometry (the step is applied
/// in place, with nothing to roll back to), so it measures **no displacement** and says
/// so — `has_displacement = false` — and `cuda::ForceConverged` then simply leaves the
/// two displacement criteria out of its test rather than guessing.
///
/// A component that published nothing would be worse than useless here: the handler
/// would read an all-zero `LoopMetrics` and declare an untouched system converged.
///
/// @see cuda::minimizer::Minimize for the driver that owns the loop
/// @see cuda/driver/loop_handler.cuh for the handler that judges the metrics
/// @see minimizer::Phase for the minimizer phase sequence
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

/// @brief Steepest-descent Optimizer component: `x += alpha * f / m * kFtm2v`.
///
/// Binds `search_direction` (the step) and `convergence_check` (the metric publish).
/// The `alpha` parameter controls the step size (a pseudo-timestep in fs) — same unit
/// conversion as `VelocityVerlet`. Tagged `role()==Optimizer` so
/// `cuda::minimizer::Minimize` build-time validation accepts it as the
/// required optimization step.
///
/// @see cuda::minimizer::Minimize for the driver this component plugs into
class SteepestDescent : public Component {
  public:
    /// @brief Construct a steepest-descent step.
    /// @param alpha Step size [fs] (pseudo-timestep, same units as
    ///        VelocityVerlet dt).
    explicit SteepestDescent(double alpha) : alpha_(alpha) {}

    /// @brief Component name.
    /// @return "steepest_descent".
    std::string name() const override { return "steepest_descent"; }
    Role role() const override { return 0; }

    /// @brief Structural role of this component.
    /// @return ComponentRole::Optimizer.
    /* role via Optimizer base */

    /// @brief Default bindings: search_direction + convergence_check, both at order 0.
    /// @return The two bindings.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    /// @brief Allocate the ConvCheck metric accumulators.
    /// @param ctx Build context (frame, stream, runtime).
    void build(const BuildContext& ctx) override;

    /// @brief Dispatch the bound phase (search_direction step / metric publish).
    /// @param ctx Launch context (frame, stream, runtime).
    void launch(const LaunchContext& ctx) override;

    /// @brief Release the ConvCheck buffers.
    /// @param ctx Destroy context.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Step-size parameter.
    /// @return alpha [fs].
    double alpha() const { return alpha_; }

    /// @brief Max force magnitude at the latest convergence check [eV/Angstrom].
    double max_force() const { return conv_.max_force(); }

    /// @brief Per-step optimization trace.
    const std::vector<OptRecord>& trace() const { return trace_; }

  private:
    double alpha_; ///< Step size (pseudo-timestep) [fs].
    int step_ = 0; ///< Optimizer step counter.

    ConvCheck conv_; ///< Metric reduction (publishes; never decides).
    std::vector<OptRecord> trace_; ///< Per-step diagnostics.
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
