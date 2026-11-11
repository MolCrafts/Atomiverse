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
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/frame/input_snapshot.h"
#include "atomiverse/cuda/asynchronous/capture_buffer.cuh"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/graph_ring.cuh"
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

    /// @brief Per-run output of the current state (LAMMPS `Output::setup`):
    ///        calls `setup_output` on every component of the MD plan with
    ///        `ctx.step` = `step_idx`, the steps completed so far. Eager,
    ///        device work only (no sync). No-op for other plans.
    void setup_output(BigInt step_idx);

    /// @brief Forced output of a minimization's final state (LAMMPS
    ///        `Min::run` on a stop before MAXITER): calls `final_output` on
    ///        every component of the minimizer plan with `ctx.step` =
    ///        `step_idx` (iterations completed). Eager, device work only.
    void final_output(BigInt step_idx);

    /// @brief Execute one simulation step (launch all components by phase).
    ///
    /// Never waits on the device (iron law 10): the step ends with an async
    /// copy of the fault word into a pinned ring, observed later by
    /// `fault_observed()`.
    /// @param step_idx Zero-based step index within the simulation.
    void step(BigInt step_idx);

    /// @brief True once a completed step reported a device fault.
    ///
    /// Non-blocking: the fault word is copied every `kFaultPollEvery` steps
    /// and only completed copies are read, so a fault is seen late. The device state froze at the
    /// faulting step, so a loop that stops issuing steps here loses nothing;
    /// `retrieve()` then reports the fault.
    bool fault_observed() const noexcept { return fault_observed_; }

    /// @brief Last step of the current MD run: it evaluates the energy (and
    ///        a demanded virial) whatever the cadences say, so `retrieve()`
    ///        returns the final state's scalars. -1: none.
    void set_report_step(BigInt step) noexcept { report_step_ = step; }

    /// @brief Re-evaluate forces, energy and a demanded virial at the
    ///        current state (a run that stopped before its report step).
    void evaluate_current(BigInt step_idx) {
        forces_ready_ = false;
        prepare_run(step_idx);
    }

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

    /// @brief The driver's shared asynchronous capture buffer.
    CaptureBuffer& capture() noexcept { return capture_; }
    const CaptureBuffer& capture() const noexcept { return capture_; }

    /// @brief Access the CUDA context (streams).
    cuda::Context& context() { return context_; }

    /// @brief Access the cuda::Frame.
    cuda::Frame& frame() { return frame_; }
    EvaluationTransaction& evaluation_transaction() { return evaluation_; }

  private:
    void publish_box_();
    /// `Ndof = 3N - 3 - n_constraints` over the loaded plan's constraint
    /// providers, before any component builds (thermostats cache it).
    void publish_dof_(const cpu::Frame& frame);
    void observe_fault_(BigInt step_idx);
    void apply_demand_(LaunchContext& ctx, BigInt step) const;
    void execute_step_graphed_(BigInt step_idx);
    void sync_device_step_(BigInt step_idx);
    /// One `setup_output` (`final = false`) or `final_output` pass over the
    /// MD / minimizer plan's components at its end-of-iteration slot.
    void output_pass_(BigInt step_idx, bool final);
    void advance_device_step_(cudaStream_t stream);
    void forget_graphs_() noexcept;
    long long* device_step_ = nullptr; ///< context_.device_step storage.
    BigInt device_step_next_ = -1;     ///< Value the device copy holds next.
    /// Step graphs keyed by the ordered (component, epoch, phase, flags)
    /// segment.
    std::map<std::vector<std::uintptr_t>, GraphRing> step_graphs_;
    /// One captured launch of a segment.
    struct SegmentEntry {
        Component* component; ///< Null: the device step advance.
        int phase;
        bool forces, energy, stress, box;
    };
    /// The ring a segment ordinal of the step resolved to last time, with
    /// the key it resolved: a steady step re-resolves every segment by one
    /// key comparison, with no allocation and no map lookup.
    struct SegmentMemo {
        std::vector<std::uintptr_t> key;
        GraphRing* ring = nullptr;
    };
    std::vector<SegmentEntry> segment_;       ///< Reused per step.
    std::vector<std::uintptr_t> segment_key_; ///< Reused per segment.
    std::vector<SegmentMemo> segment_memo_;   ///< Indexed by segment ordinal.
    std::vector<EvaluationDemand> demands_;   ///< MD plans only.
    bool gate_evaluation_ = false;
    bool virial_demanded_ = false;
    /// A loaded component `tilts_box()`: kernels must take the general-cell
    /// specialization even while the published cell is still orthorhombic,
    /// because the host never re-reads the cell the device shears.
    bool general_cell_ = false;
    BigInt report_step_ = -1;
    void forget_fault_();
    void release_fault_ring_();
    static constexpr int kFaultRing = 4;
    static constexpr int kFaultPollEvery = 64; ///< Steps between copies.
    unsigned* fault_ring_ = nullptr;           ///< Pinned fault-word copies.
    cudaEvent_t fault_ready_[kFaultRing] = {};
    bool fault_pending_[kFaultRing] = {};
    int fault_slot_ = 0;
    bool fault_observed_ = false;
    ATV::detail::ComponentLifecycle<Component> lifecycle_;
    EvaluationTransaction evaluation_;
    cpu::FrameInputSnapshot input_snapshot_;
    bool inputs_dirty_ = false;
    bool identity_dirty_ = false;
    bool forces_ready_ = false;
    bool dispatching_ = false;
    std::vector<std::pair<int, Component*>> force_bindings_;
    cuda::Context context_{}; ///< CUDA stream handles.
    /// Every asynchronous consumer's frames (published on context_.capture).
    CaptureBuffer capture_;
    cuda::Frame frame_{}; ///< Device-resident system state + host metadata.
    const Plan<Component>* plan_ =
        nullptr;             ///< Non-owning pointer to compiled Plan.
    Timer* timer_ = nullptr; ///< Non-owning pointer to optional Timer.
};

} // namespace cuda
} // namespace ATV
