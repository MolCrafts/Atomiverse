/// @file cpu/driver/runtime.h
/// @brief cpu::Runtime -- the CPU loop runtime: Plan dispatch + component
///        lifecycle, composed from communication / device / asynchronous
///        sub-runtimes.
///
/// backend-mirror-05-dissolve replaced the former god-class: domain
/// decomposition now lives in `communication::DomainRuntime`, thread + timing
/// setup in `device::DeviceRuntime`, the async pipeline in
/// `asynchronous::AnalysisRuntime`. This type owns those three, binds the Frame,
/// and dispatches Plans (one per driver node in the tree). It operates directly
/// on the user's cpu::Frame (zero-copy, zero transfer).
///
/// One Runtime is shared by a whole driver tree: a sub-driver runs its own Plan
/// on its parent's Runtime (published on `Context::exec.runtime`), so nesting
/// does not double the runtime, the decomposition, or the frame binding.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/asynchronous/analysis_runtime.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/device/device_runtime.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

class NeighborProvider;

namespace md {
namespace mpi {
struct DomainDecomposition;
} // namespace mpi
} // namespace md

/// @brief CPU loop runtime -- lifecycle manager + Plan dispatcher.
class Runtime {
  public:
    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    /// @brief Configure threads (device), detect MPI, allocate the DD
    ///        (communication), start the analysis pipeline, publish services.
    void init();

    /// @brief Bind a compiled Plan for execution (non-owning).
    void load(const Plan<cpu::Component>& plan) { plan_ = &plan; }

    /// @brief Set the Frame to operate on (zero-copy) and bind the DD to it.
    ///
    /// Also decides the fate of the Frame tag index: the loaded Plan is walked
    /// for components (and nested loop nodes) that declared
    /// `Component::requires_tag_index()`, and the answer is handed to
    /// `DomainRuntime::bind_frame`. This is the only place that knows the whole
    /// driver tree *and* still runs before the frame is bound.
    void set_frame(cpu::Frame& frame);

    /// @brief Build all components in Plan order.
    void build_components();

    /// @brief Execute one simulation step.
    void step(int step_idx);

    // ── Nested plans: a sub-driver runs its own Plan on THIS runtime ──────────
    void build_plan(const Plan<cpu::Component>& plan);
    void step_plan(const Plan<cpu::Component>& plan, int step_idx);
    void destroy_plan(const Plan<cpu::Component>& plan);

    /// @brief Re-run only the bindings of a single phase (line-search trials).
    /// @tparam PhaseEnum The workflow Phase enum this runtime's Plan was compiled from.
    template <typename PhaseEnum>
    void run_stage(PhaseEnum phase, int step_idx) {
        run_slot_(ATV::slot_of(phase), step_idx);
    }

    /// @brief Execute multiple simulation steps.
    void run(int n_steps);

    /// @brief Destroy all components in reverse Plan order.
    void destroy_components();

    /// @brief Block until the async-analysis executor has drained.
    void drain_analysis() { analysis_.drain(); }

    /// @brief Release all resources.
    void shutdown();

    /// @brief Access the CPU context.
    Context& context() { return context_; }

    /// @brief Cap the OpenMP thread count. Must be called before `init()`.
    void set_threads(int n) { device_.set_threads(context_, n); }

    /// @brief Access the Frame.
    Frame& frame() { return *frame_; }

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
    void set_serial_on_replica() noexcept { domain_.set_serial_on_replica(); }

  private:
    /// @brief Replay one compiled Plan slot. The erased half of `run_stage`.
    void run_slot_(int slot, int step_idx);

    /// @brief NeighborProvider cache for Verlet / linear `ctx.md.comm` dispatch.
    ///        Halo adapters are not cached; Runtime calls one `Comm*`.
    struct VerletComm {
        cpu::NeighborProvider* neighbor = nullptr;
        cpu::Component* neighbor_comp = nullptr;
        int neighbor_slot = -1;
        int neighbor_op = -1;
        /// Columns the halo carries this run, folded from every mounted
        /// `FieldUser` plus the base set. Derived at `build()`; see
        /// `core/schema.h`.
        Schema ghost_fields;
        Schema ghost_reverse_fields;

        /// Widest `NeighborConsumer::halo_cutoff()` [Å] among mounted
        /// components. A many-body style (EAM) declares a ghost-of-ghost
        /// shell of 2·cutoff here, wider than the provider's enumeration
        /// reach; the ghost shell must honour whichever is larger.
        double consumer_halo_cutoff = 0.0;
    };

    void cache_verlet_comm_();
    /// @brief Size `ctx.md.comm` from `NeighborProvider::neighbor_reach()` [Å].
    void apply_ghost_cutoff_();
    /// @brief True when `slot` is Evaluate `evaluate` or Minimize `force`.
    bool is_linear_force_slot_(int slot) const;
    /// @brief nflag XOR: forward, or exchange + borders + neighbor build.
    void comm_before_force_(int slot, int step_idx);
    /// @brief Newton-on reverse of ghost forces [eV/Å].
    void comm_after_force_(int slot, int step_idx);
    /// @brief LAMMPS Verlet::setup: `comm->exchange` then `borders` then
    ///        neighbor build (MPI only). Ghost positions in Å.
    void setup_verlet_comm_();
    /// @brief Evaluate/Minimize: Comm around force/evaluate (Å positions,
    ///        reverse of ghost forces [eV/Å]); skip `verlet_engine` adapters.
    void step_linear_(int step_idx);
    /// @brief LAMMPS Verlet::run comm on `ctx.md.comm`. `nflag==0` →
    ///        `forward_comm` (ghost positions [Å]); else exchange + borders +
    ///        neighbor build. `reverse_comm` every step (newton-on, [eV/Å]).
    void step_verlet_(int step_idx);
    void launch_op_(std::size_t slot, std::size_t op, int step_idx);
    void walk_slot_(int phase, int step_idx, bool skip_engine);
    void launch_engine_(cpu::Component* c, int slot, int op, int phase,
                        int step_idx);

    Context context_{};
    cpu::Frame* frame_ = nullptr;
    const Plan<cpu::Component>* plan_ = nullptr;

    communication::DomainRuntime domain_{};
    device::DeviceRuntime device_{};
    asynchronous::AnalysisRuntime analysis_{};

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
};

} // namespace cpu
} // namespace ATV
