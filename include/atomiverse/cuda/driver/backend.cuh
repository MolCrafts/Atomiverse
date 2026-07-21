/// @file cuda/driver/backend.cuh
/// @brief Backend -- GPU execution engine and component lifecycle manager.
///
/// Owns Runtime (CUDA streams), manages DeviceFrame lifecycle,
/// and dispatches the build / launch / destroy lifecycle on components.
/// CUDA-dependent, internal header -- only included by .cu files.
///
/// @see Driver for the user-facing facade that delegates to Backend
#pragma once

#include <cuda_runtime.h>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/timer.h"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/facade/plan.h"

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
    void load(const Plan<Component>& plan) { plan_ = &plan; }

    /// @brief Materialize DeviceFrame from a host Frame (calls cpu::to_cuda).
    /// @param frame Host-side system state to transfer to device.
    void materialize(const cpu::Frame& frame);

    /// @brief Build all components in Plan order (calls Component::build).
    void build_components();

    /// @brief Execute one simulation step (launch all components by phase).
    /// @param step_idx Zero-based step index within the simulation.
    void step(int step_idx);

    /// @brief Execute multiple simulation steps.
    /// @param n_steps Number of steps to execute.
    void run(int n_steps);

    /// @brief Destroy all components in reverse Plan order (calls Component::destroy).
    void destroy_components();

    /// @brief Retrieve results back to host Frame (calls cuda::to_cpu).
    /// @param frame Host-side Frame to populate with device results.
    void retrieve(cpu::Frame& frame);

    /// @brief Synchronize all CUDA streams.
    void sync();

    /// @brief Release all resources (CUDA streams, device memory).
    void shutdown();

    /// @brief Access the CUDA context (streams).
    cuda::Context& context() { return context_; }

    /// @brief Access the cuda::Frame.
    cuda::Frame& frame() { return frame_; }

  private:
    cuda::Context context_{}; ///< CUDA stream handles.
    cuda::Frame frame_{};     ///< Device-resident system state + host metadata.
    const Plan<Component>* plan_ =
        nullptr;             ///< Non-owning pointer to compiled Plan.
    Timer* timer_ = nullptr; ///< Non-owning pointer to optional Timer.
};

} // namespace cuda
} // namespace ATV
