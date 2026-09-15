/// @file cpu/asynchronous/async_component_base.h
/// @brief `AsyncComponentBase` -- driver-agnostic base for off-hot-path
/// analysis components.
///
/// Role-neutral. Both observers (which compute derived quantities) and
/// recorders (which write I/O) inherit directly. The Observer/Recorder
/// distinction is preserved by subclass responsibility, not by an
/// intermediate base class.
///
/// This base class declares **no default bindings**. The phase enum
/// (`md::Phase`, `minimizer::Phase`, `evaluate::Phase`) is driver-specific, so
/// hardcoding one here would silently misbind when reused across drivers.
/// Users either:
///   - Bind explicitly: `sim.on(my_phase_enum::end_of_step).add_ref(obs, 20);`
///   - Inherit from a driver-specific wrapper that provides the binding:
///     `class MyRDF : public cpu::md::AsyncObserver { ... };`
///     (see `cpu/md/async_observer.h`, `cpu/minimizer/async_observer.h`).
///
/// Subclass contract:
///   - `capture_mask()` -- which frame fields the snapshot should carry.
///   - `on_snapshot(FrameSnapshot&)` -- runs on an analysis worker thread.
///     May call MPI collectives on `executor()->analysis_comm()`.
///   - `finalize()` -- called exactly once by the executor during shutdown.
///
/// Lifecycle and delivery:
///   - `build()` resolves `ctx.runtime.analysis.executor`, registers as a subscriber,
///     pre-sizes ring slots for this observer's capture mask.
///   - `launch()` is a `final` thin dispatcher: skip on the `frequency()`
///     stride, else submit (this, current_slot) to the executor in ~30 ns.
///     No physics.
///   - `destroy()` is a `final` no-op; `finalize()` is invoked by the
///     executor at `shutdown()` exactly once.
///
/// @see cpu/md/async_observer.h            driver-specific default binding
/// @see cpu/minimizer/async_observer.h     driver-specific default binding
/// @see .claude/specs/async-analysis-snapshot-ring-01-uniformity.md
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

class AnalysisExecutor;

/// @brief Driver-agnostic base for components whose work runs off the main hot path.
class AsyncComponentBase : public cpu::Component {
  public:
    /// @brief No default binding. Users bind explicitly or subclass one of
    /// the driver-specific wrappers in `cpu/<driver>/async_observer.h`.
    std::vector<DefaultBinding> default_bindings() const override { return {}; }

    /// @brief Mark as deferrable so architecture-level tooling can verify
    /// that compute does not land on the main stream.
    Scheduling scheduling() const override { return Scheduling::deferrable; }

    /// @brief Which frame fields the snapshot should carry for this subscriber.
    virtual CaptureMask capture_mask() const = 0;

    /// @brief Register with the executor resolved from `ctx.runtime.analysis.executor`.
    void build(const BuildContext& ctx) override;

    /// @brief Thin dispatcher: skip or submit. `final` to prevent subclasses
    /// from doing physics here.
    void launch(const LaunchContext& ctx) final;

    /// @brief No-op; `finalize()` is called by the executor during shutdown.
    void destroy(const DestroyContext& ctx) final;

    /// @brief Worker-thread entry point. Runs on the analysis executor thread.
    /// @param snap Snapshot holding fields selected by `capture_mask()`.
    virtual void on_snapshot(const FrameSnapshot& snap) = 0;

    /// @brief Optional end-of-run hook. Called once by `AnalysisExecutor::shutdown()`.
    virtual void finalize() {}

    /// @brief Accessor for the bound executor (e.g. for MPI sub-communicator).
    AnalysisExecutor* executor() const noexcept { return exec_; }

  private:
    AnalysisExecutor* exec_ = nullptr;
};

} // namespace cpu
} // namespace ATV
