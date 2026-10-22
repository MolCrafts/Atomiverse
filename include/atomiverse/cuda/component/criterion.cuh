/// @file cuda/component/criterion.cuh
/// @brief CUDA termination criterion participating in Component lifecycle.
#pragma once
#include "atomiverse/core/verdict.h"
#include "atomiverse/cuda/component/component.cuh"
namespace ATV::cuda {
/// @brief Everything a handler is allowed to look at.
struct LoopState {
    BigInt iteration; ///< 0-based index of the iteration just completed.
    const cuda::Frame& frame; ///< Device-resident state as it now stands.
    const cuda::Context&
        runtime; ///< Execution context (streams, service slices).
    const ConvergenceMetrics&
        metrics; ///< What this loop's components measured.
                 ///< @see facade/loop.h
};

/// @brief A termination policy, asked once after every iteration of the loop it is on.
class Criterion : public Component {
  public:
    virtual ~Criterion() = default;

    /// @brief Whether this policy is required for successful convergence.
    /// @return False for stop-only guards; convergence criteria override this.
    virtual bool contributes_to_convergence() const { return false; }

    /// @brief Identifier, used in the stop reason the driver reports.
    /// @brief Verdict-only policies have no scheduled measurement.
    void launch(const LaunchContext&) override {}

    /// @brief Judge the iteration just completed.
    /// @param state Iteration index, frame, context, metrics.
    /// @return `Continue`, `Converged` or `Stop`. @see ATV::Verdict
    virtual Verdict verdict(const LoopState& state) = 0;
};

} // namespace ATV::cuda
