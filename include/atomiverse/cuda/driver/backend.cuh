/// @file cuda/driver/backend.cuh
/// @brief Backend -- GPU execution engine and component lifecycle manager.
///
/// Owns Runtime (CUDA streams), manages DeviceFrame lifecycle,
/// and dispatches the build / launch / destroy lifecycle on components.
/// CUDA-dependent, internal header -- only included by .cu files.
///
/// @see Driver for the user-facing facade that delegates to Backend
#pragma once

#include "atomiverse/core/precision.h"
#include <cuda_runtime.h>

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/frame/input_snapshot.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/timer.h"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {

namespace cpu {
class Frame;
}

namespace cuda {

/// @brief GPU execution engine -- lifecycle manager + dispatcher.
///
/// Manages the full simulation lifecycle:
/// 1. init() -- create CUDA streams
/// 2. materialize() -- transfer Frame to device (DeviceFrame)
/// 3. build_components() -- one-time resource allocation
/// 4. run() / step() -- per-step kernel dispatch
/// 5. destroy_components() -- release build-time resources (reverse order)
/// 6. retrieve() -- transfer results back to host
/// 7. shutdown() -- free all device memory and streams
///
/// @see Driver::run() which orchestrates this full sequence
class Backend {
  public:
    /// @brief Create CUDA streams and populate Runtime.
    void init();

    /// @brief Attach an external Timer for GPU profiling.
    /// @param t Non-owning pointer to a Timer (lifetime managed by Driver).
    void set_timer(Timer* t) { timer_ = t; }

    /// @brief Bind a compiled Plan for execution.
    /// @param plan Plan to execute (non-owning reference, must outlive Backend usage).
    void load(const Plan<Component>& plan);

    /// @brief Materialize DeviceFrame from a host Frame (calls cpu::to_cuda).
    /// @param frame Host-side system state to transfer to device.
    void materialize(const cpu::Frame& frame);

    /// @brief Build all components in Plan order (calls Component::build).
    void build_components();

    /// @brief Build only the components in `plan` (mid-run remount).
    /// @param plan Temporary one-node (or subset) plan; does not replace
    ///        the loaded schedule.
    void build_plan(const Plan<Component>& plan);

    /// @brief Destroy only the components in `plan` (mid-run unfix).
    /// @param plan Temporary one-node (or subset) plan; does not replace
    ///        the loaded schedule.
    void destroy_plan(const Plan<Component>& plan);

    /// @brief Execute a half-open range of scheduled phases without counting a step.
    /// Frequencies, stream selection and profiling match the full-step dispatch.
    /// @param step_idx Zero-based simulation step passed to components.
    /// @param first_phase First included phase index [dimensionless].
    /// @param end_phase First excluded phase index [dimensionless].
    /// @throws std::invalid_argument If the range is outside the loaded plan.
    void execute_phases(BigInt step_idx, int first_phase, int end_phase);
    /// Dispatch a child schedule using the same lifecycle, requests and streams.
    /// Nested dispatch is included in its parent's timer entry, not timed twice.
    void execute_phases(const Plan<Component>& plan, BigInt step_idx,
                        int first_phase, int end_phase);

    /// Stable mutation lock borrowed by mounted child drivers.
    const bool* mutation_lock() const { return &dispatching_; }

    /// Establish force/scalar readiness without advancing the trajectory.
    void prepare_run(BigInt step_idx = 0);

    /// @brief Execute one simulation step (launch all components by phase).
    /// @param step_idx Zero-based step index within the simulation.
    void step(BigInt step_idx);

    /// @brief Execute multiple simulation steps.
    /// @param n_steps Number of steps to execute.
    void run(int n_steps);

    /// @brief Destroy all components in reverse Plan order (calls Component::destroy).
    void destroy_components();

    /// @brief Retrieve results back to host Frame (calls cuda::to_cpu).
    /// @param frame Host-side Frame to populate with device results.
    /// @param copy_forces Retrieve forces; false preserves host force columns.
    void retrieve(cpu::Frame& frame, bool copy_forces = true);

    /// Compare with the last write-back and apply caller edits before continuation.
    void refresh_inputs(const cpu::Frame& frame);
    /// Classify and preflight edits before new component build can mutate state.
    InputChange validate_inputs(const cpu::Frame& frame) const;
    /// A failed evaluation cannot reuse partially advanced device inputs/forces.
    void mark_inputs_dirty() noexcept {
        inputs_dirty_ = true;
        forces_ready_ = false;
    }

    /// @brief Synchronize all CUDA streams.
    void sync();

    /// @brief Release all resources (CUDA streams, device memory).
    void shutdown();

    /// @brief Access the CUDA context (streams).
    cuda::Context& context() { return context_; }

    /// @brief Access the cuda::Frame.
    cuda::Frame& frame() { return frame_; }
    EvaluationTransaction& evaluation_transaction() { return evaluation_; }

  private:
    void publish_box_();
    detail::ComponentLifecycle<Component> lifecycle_;
    EvaluationTransaction evaluation_;
    cpu::FrameInputSnapshot input_snapshot_;
    bool inputs_dirty_ = false;
    bool identity_dirty_ = false;
    bool forces_ready_ = false;
    bool dispatching_ = false;
    std::vector<std::pair<int, Component*>> force_bindings_;
    cuda::Context context_{}; ///< CUDA stream handles.
    cuda::Frame frame_{};     ///< Device-resident system state + host metadata.
    const Plan<Component>* plan_ =
        nullptr;             ///< Non-owning pointer to compiled Plan.
    Timer* timer_ = nullptr; ///< Non-owning pointer to optional Timer.
};

} // namespace cuda
} // namespace ATV
