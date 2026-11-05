/// @file cpu/driver/runtime.h
/// @brief cpu::Runtime -- the CPU loop runtime: Plan dispatch + component
///        lifecycle, composed from communication / device / asynchronous
///        sub-runtimes.
///
/// backend-mirror-05-dissolve replaced the former god-class: domain
/// decomposition now lives in `communication::DomainRuntime`, thread + timing
/// setup in `device::DeviceRuntime`, the asynchronous frame capture in
/// `cpu::CaptureBuffer`. This type owns those three, binds the Frame,
/// and dispatches Plans (one per driver node in the tree). It operates directly
/// on the user's cpu::Frame (zero-copy, zero transfer).
///
/// One Runtime is shared by a whole driver tree: a sub-driver runs its own Plan
/// on its parent's Runtime (published on `Context::exec.runtime`), so nesting
/// does not double the runtime, the decomposition, or the frame binding.
#pragma once

#include <exception>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "atomiverse/core/evaluation_history.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/asynchronous/capture_buffer.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/device/device_runtime.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/input_snapshot.h"

namespace ATV {
namespace cpu {

class NeighborProvider;
class DriverNode;

namespace md {
namespace mpi {
struct DomainDecomposition;
} // namespace mpi
} // namespace md

/// @brief CPU loop runtime -- lifecycle manager + Plan dispatcher.
class Runtime {
  public:
    class ScopedPlan;
    /// Protect a synchronous callback, including criterion/outcome polling.
    class DispatchScope {
      public:
        explicit DispatchScope(Runtime* runtime)
            : runtime_(runtime),
              flag_(runtime ? &runtime->dispatch_active_ : nullptr),
              saved_(flag_ && *flag_), exceptions_(std::uncaught_exceptions()) {
            if (flag_)
                *flag_ = true;
        }
        ~DispatchScope() {
            if (flag_)
                *flag_ = saved_;
            // Nested dispatch first unwinds its borrowed plan/window. Destroy
            // only after the outermost callback has left all component stacks.
            if (runtime_ && runtime_->driver_owned_ && !saved_
                && std::uncaught_exceptions() > exceptions_)
                runtime_->unwind_failed_run();
        }
        DispatchScope(const DispatchScope&) = delete;
        DispatchScope& operator=(const DispatchScope&) = delete;

      private:
        Runtime* runtime_;
        bool* flag_;
        bool saved_;
        int exceptions_;
    };
    /// Shared by the tree: structural edits are only legal between dispatches.
    const bool& dispatch_active() const { return dispatch_active_; }
    EvaluationTransaction& evaluation_transaction() noexcept {
        return evaluation_history_;
    }
    void invalidate_plan() {
        contracts_ready_ = false;
        forces_ready_ = false;
    }

    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    /// @brief Configure threads (device), detect MPI, allocate the DD
    ///        (communication), publish the capture buffer and services.
    void init();

    /// @brief Bind a compiled Plan for execution (non-owning).
    void load(const Plan<cpu::Component>& plan, BigInt step_idx = 0);

    /// @brief Set the Frame to operate on (zero-copy) and bind the DD to it.
    ///
    /// Also decides the fate of the Frame tag index: the loaded Plan is walked
    /// for components (and nested loop nodes) that declared
    /// `Component::requires_tag_index()`, and the answer is handed to
    /// `DomainRuntime::bind_frame`. This is the only place that knows the whole
    /// driver tree *and* still runs before the frame is bound.
    void set_frame(cpu::Frame& frame);
    /// Compare host edits with the previous successful run boundary. Preflight
    /// every participant before invalidating caches; keep the same Runtime.
    InputChange refresh_external_inputs();
    void checkpoint_inputs();

    /// @brief Build all components in Plan order.
    void build_components();

    /// @brief Execute one simulation step.
    void step(BigInt step_idx);
    /// Dispatch a normal phase with cadence and timing, allowing a workflow
    /// to commit/rollback between evaluation and observation. MD must use step().
    void step_stage(int slot, BigInt step_idx);
    /// Establish initial MD forces once, without advancing the clock.
    void prepare_run(BigInt step_idx = 0);
    /// Per-run output of the current state (LAMMPS `Output::setup`): calls
    /// `setup_output` on every component of the MD or minimizer plan with
    /// `ctx.step` = `step_idx`, the steps / iterations completed so far.
    /// No-op for other plans.
    void setup_output(BigInt step_idx);
    /// Forced output of a minimization's final state (LAMMPS `Min::run` on a
    /// stop before MAXITER): calls `final_output` on every component of the
    /// minimizer (or MD) plan with `ctx.step` = `step_idx`.
    void final_output(BigInt step_idx);
    /// Workflow of the plan executing now (the nested plan inside a
    /// `ScopedPlan`); `Evaluate` when none is bound.
    PhaseKind active_kind() const {
        return plan_ ? plan_->kind : PhaseKind::Evaluate;
    }

    // ── Nested plans: a sub-driver runs its own Plan on THIS runtime ──────────
    void build_plan(const Plan<cpu::Component>& plan);
    void step_plan(const Plan<cpu::Component>& plan, BigInt step_idx);
    void destroy_plan(const Plan<cpu::Component>& plan);

    /// @brief Re-run only the bindings of a single phase (line-search trials).
    /// @tparam PhaseEnum The workflow Phase enum this runtime's Plan was compiled from.
    template <typename PhaseEnum>
    void run_stage(PhaseEnum phase, BigInt step_idx) {
        run_slot_(ATV::slot_of(phase), step_idx);
    }

    /// @brief Execute multiple simulation steps.
    void run(int n_steps);

    /// @brief Destroy all components in reverse Plan order.
    void destroy_components();

    /// @brief End of a run: every asynchronous capture is consumed before
    ///        the driver reports success; a consumer failure is thrown here
    ///        (once) if the run has not raised it yet.
    void drain_capture() {
        DispatchScope dispatch(this);
        capture_->drain(/*raise=*/true);
    }

    /// @brief The driver tree's shared asynchronous capture buffer.
    CaptureBuffer& capture() noexcept { return *capture_; }
    const CaptureBuffer& capture() const noexcept { return *capture_; }

    /// @brief Release all resources.
    void shutdown();

    /// Unwind a failed setup/dispatch while borrowed components and Frame live.
    /// Preserves the original exception; the next Driver run rebuilds its plan.
    void unwind_failed_run() noexcept;
    bool failed() const noexcept { return failed_; }
    /// Driver-owned calls unwind on escaping errors; raw Runtime callers retain
    /// their explicit recovery/borrowed-plan contract.
    void enable_driver_unwind() noexcept { driver_owned_ = true; }

    /// @brief Access the CPU context.
    Context& context() { return context_; }
    /// @brief Does the plan being dispatched consume the system stress?
    /// @return True when some component of the active plan reads it.
    bool stress_consumed() const { return plan_ && plan_->stress_consumed; }

    /// @brief Cap the OpenMP thread count. Must be called before `init()`.
    void set_threads(int n) { device_.set_threads(context_, n); }

    /// @brief Access the Frame.
    Frame& frame() { return *frame_; }
    /// @brief Is a Frame bound (`set_frame`)?
    /// @return True once the Runtime holds a Frame.
    bool has_frame() const noexcept { return frame_ != nullptr; }

    /// @brief The per-component timing registry (opt-in), owned by the device runtime.
    TimerRegistry& timing() { return device_.timing(); }
    const TimerRegistry& timing() const { return device_.timing(); }

    /// @brief Per-(unique-)component owned heap bytes for the driver memory summary.
    std::vector<std::pair<std::string, std::size_t>>
    collect_component_memory() const;

    /// @brief Enable/disable per-launch timing (device runtime; preference sticks).
    void enable_timing(bool on) { device_.enable_timing(on); }

    /// @brief Clear accumulated step timing.
    void reset_timing() { device_.reset_timing(); }

    /// @brief Adopt an externally-owned DomainDecomposition (before `init()`).
    /// Out-of-line: the unique_ptr is destroyed where DomainDecomposition is complete.
    void adopt_dd(std::unique_ptr<md::mpi::DomainDecomposition> dd);

    /// @brief Release the owned DomainDecomposition for reuse.
    std::unique_ptr<md::mpi::DomainDecomposition> release_dd();

    /// @brief Run serially on a full-replica frame even under multi-rank MPI.
    void set_serial_on_replica() noexcept {
        context_.exec.serial_on_replica = true;
    }

  private:
    bool failed_ = false;
    bool driver_owned_ = false;
    /// @brief Replay one compiled Plan slot. The erased half of `run_stage`.
    void run_slot_(int slot, BigInt step_idx);

    /// @brief NeighborProvider cache for Verlet / linear `ctx.md.comm` dispatch.
    ///        Halo adapters are not cached; Runtime calls one `Comm*`.
    struct VerletComm {
        cpu::NeighborProvider* neighbor = nullptr;
        cpu::Component* neighbor_comp = nullptr;
        int neighbor_slot = -1;
        int neighbor_op = -1;
        /// The run's ghost contract: every mounted component's
        /// `declare_ghosts` folded onto the engine's own, once at `build()`.
        /// Each communication pass compiles itself off this directly.
        /// See `core/ghost_contract.h`.
        GhostContract ghost;

        /// Widest `NeighborConsumer::halo_cutoff()` [Å] among mounted
        /// components. A consumer may need a coordinate shell wider than
        /// the provider's enumeration reach; honour whichever is larger.
        double consumer_halo_cutoff = 0.0;
        double provider_reach = 0.0;
    };

    void cache_verlet_comm_();
    /// @brief Size `ctx.md.comm` from `NeighborProvider::neighbor_reach()` [Å].
    void apply_ghost_cutoff_();
    /// @brief True when `slot` is Evaluate `evaluate` or Minimize `force`.
    bool is_linear_force_slot_(int slot) const;
    /// @brief True when every non-engine op of the linear force slot is
    ///        `Component::trial_invariant()`.
    bool force_slot_trial_invariant_() const;
    /// @brief Position-only halo refresh (no reneighbor).
    void comm_refresh_(int slot, int step);
    /// @brief Exchange + borders. Flat slot args so Verlet can walk
    ///        `pre_exchange` / `pre_neighbor` in between; no schedule bag.
    void decide_neighbor_();
    void comm_rebuild_(int exchange_slot, int borders_slot, BigInt step);
    /// @brief nflag XOR: forward, or exchange + borders + neighbor build.
    void comm_before_force_(int slot, BigInt step_idx);
    /// @brief Newton-on reverse of ghost forces [eV/Å].
    void comm_after_force_(int slot, BigInt step_idx);
    /// @brief LAMMPS Verlet::setup: `comm->exchange` then `borders` then
    ///        neighbor build (MPI only). Ghost positions in Å.
    void setup_verlet_comm_(BigInt step_idx = 0);
    /// @brief Evaluate/Minimize: Comm around force/evaluate (Å positions,
    ///        reverse of ghost forces [eV/Å]); skip `verlet_engine` adapters.
    void step_linear_(BigInt step_idx);
    /// @brief LAMMPS Verlet::run comm on `ctx.md.comm`. `nflag==0` →
    ///        `forward_comm` (ghost positions [Å]); else exchange + borders +
    ///        neighbor build. `reverse_comm` every step (newton-on, [eV/Å]).
    void step_verlet_(BigInt step_idx);
    void launch_op_(std::size_t slot, std::size_t op, BigInt step_idx);
    void walk_slot_(int phase, BigInt step_idx, bool skip_engine);
    /// One `setup_output` (`final = false`) or `final_output` pass over the
    /// plan's components at its end-of-iteration slot.
    void output_pass_(BigInt step_idx, bool final);
    void launch_engine_(cpu::Component* c, int slot, int op, int phase,
                        BigInt step_idx);
    void release_component_(cpu::Component* component);
    void refresh_plan_(BigInt step_idx = 0);
    bool dispatch_active_ = false;
    int active_slot_ = -1;
    bool physical_pipeline_active_ = false;
    bool forces_ready_ = false;
    std::uint64_t force_box_revision_ = 0;
    std::vector<StageSlot<cpu::Component>> force_bindings_;
    bool components_built_ = false;
    bool contracts_ready_ = false;

    // One reference per Plan containing an object, not per phase binding.
    // Zero marks a build in progress so recursive ownership fails explicitly.
    // `alive` and `node` are captured when the object is taken in, so a
    // borrowed component destroyed before its driver is released without
    // being called into (no destroy(), no nested_node()).
    struct ComponentUse {
        std::size_t users = 0;
        std::shared_ptr<const bool> alive;
        cpu::DriverNode* node = nullptr;
    };
    std::unordered_map<cpu::Component*, ComponentUse> component_users_;
    std::vector<cpu::Component*> component_order_;
    // Distinct stage adapters can borrow the same node. Its subtree belongs
    // to this Runtime once, independently of adapter/budget multiplicity.
    std::unordered_map<cpu::DriverNode*, std::size_t> node_users_;

    FrameInputSnapshot input_snapshot_;
    InputChange input_changes_pending_ = InputChange::None;
    EvaluationTransaction evaluation_history_;
    Context context_{};
    cpu::Frame* frame_ = nullptr;
    const Plan<cpu::Component>* plan_ = nullptr;

    communication::DomainRuntime domain_{};
    device::DeviceRuntime device_{};
    /// Heap-held so a moved Runtime keeps the address its Context publishes.
    std::unique_ptr<CaptureBuffer> capture_ = std::make_unique<CaptureBuffer>();

    /// Per-slot, per-op dense timing handles, precomputed in `build_components()`
    /// so the hot `step()` loop never calls `component->name()`.
    std::vector<std::vector<int>> op_handle_;

    /// Runtime-level comm timing handles (forward / exchange / borders /
    /// reverse). LAMMPS's `Comm` section covers exactly these calls; they
    /// are Runtime-driven rather than plan ops, so without their own
    /// handles the breakdown table under-reports Comm by the whole halo.
    int comm_handle_fwd_ = -1;
    int comm_handle_exch_ = -1;
    int comm_handle_bord_ = -1;
    int comm_handle_rev_ = -1;

    VerletComm verlet_{};
    bool verlet_setup_done_ = false;
    /// The communicator compiles its wires from `ctx.md.ghost` in `setup()`.
    /// Every refold (build, live mount, nested plan in or out) sets this so
    /// the next rebuild boundary re-runs `setup()` before exchanging.
    bool comm_contract_stale_ = false;
};

/// Active-plan lifetime for a nested loop. Domain ownership remains shared.
class Runtime::ScopedPlan {
  public:
    ScopedPlan(Runtime& runtime, const Plan<cpu::Component>& plan,
               BigInt step_idx = 0);
    ~ScopedPlan();
    ScopedPlan(const ScopedPlan&) = delete;
    ScopedPlan& operator=(const ScopedPlan&) = delete;

  private:
    Runtime& runtime_;
    const Plan<cpu::Component>* saved_plan_;
    DynamicsServices saved_md_;
    VerletComm saved_verlet_;
    std::vector<std::vector<int>> saved_handles_;
    std::vector<StageSlot<cpu::Component>> saved_force_bindings_;
    bool saved_contracts_, saved_setup_, saved_forces_;
    bool changed_, physical_;
};

} // namespace cpu
} // namespace ATV
