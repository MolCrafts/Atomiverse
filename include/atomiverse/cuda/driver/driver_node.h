/// @file cuda/driver/driver_node.h
/// @brief cuda::DriverNode — non-template base for a nestable CUDA loop.
///
/// Mirrors `cpu::DriverNode`. Type erasure is at the node boundary so an
/// outer Evaluate / MD Plan can hold a `cuda::scf::RHF` without agreeing
/// on phase enums. The child runs on the parent's device frame and stream
/// (`run_nested_erased`); it does not allocate a second device Frame.
#pragma once
#include "atomiverse/core/input_change.h"

namespace ATV {
namespace cuda {

class Context;
class Frame;

/// @brief Erased base for a nestable CUDA loop node.
class DriverNode {
  public:
    virtual ~DriverNode() = default;

    /// @brief Prepare this loop's Plan and workspace using the parent execution.
    /// Geometry-dependent values are refreshed when the loop runs.
    /// @param ctx Parent CUDA context (streams).
    /// @param frame Parent device frame.
    virtual void build_on(Context& ctx, Frame& frame) = 0;

    /// @brief Run this loop on the parent's context and device frame.
    /// @param ctx Parent context. `ctx.loop` is re-pointed at this loop
    ///        for the call and restored before return.
    /// @param frame Parent device frame (energy / forces written here).
    /// @param budget Iteration budget; `< 0` uses `default_budget()`.
    /// @return Iterations actually executed (≤ budget).
    virtual int run_nested_erased(Context& ctx, Frame& frame, int budget) = 0;

    /// @brief Release this loop's attachment to the parent execution.
    /// Method-owned reusable workspace may survive this detach.
    /// @param ctx Parent context.
    /// @param frame Parent device frame.
    virtual void destroy_on(Context& ctx, Frame& frame) = 0;

    /// @brief Nested budget when the caller passes `< 0`.
    /// @return Preset default (0 unless overridden).
    virtual int default_budget() const { return 0; }

    virtual bool supports_input_changes(InputChange changes) const {
        return !has_input_change(changes, InputChange::Identity);
    }
    virtual void invalidate_inputs(InputChange, Context&, Frame&) {}
};

} // namespace cuda
} // namespace ATV
