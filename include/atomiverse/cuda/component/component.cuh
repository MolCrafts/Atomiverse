/// @file cuda/component/component.cuh
/// @brief Component -- OOP base class for all simulation components.
///
/// Components follow a strict 3-phase lifecycle:
///   build()   -- one-time resource preparation (after Runtime + DeviceFrame exist)
///   launch()  -- per-step GPU work submission (pure GPU, zero D2H/sync/malloc)
///   destroy() -- resource cleanup
///
/// Two registration paths:
/// - `sim.set_observer(comp)` -- uses default_bindings().
/// - `sim.on(phase).add(comp)` -- explicit phase override.
///
/// CUDA-free header.
#pragma once
#include "atomiverse/core/input_change.h"
#include <cstdint>

#include "atomiverse/core/precision.h"
#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/frame.cuh"

namespace ATV {
namespace cuda {

class NeighborConsumer; // opt-in neighbor protocols (neighbor_protocol.cuh)
class NeighborProvider;
class
    LongRangeParticipant; // opt-in long-range protocol (long_range_protocol.cuh)
class TimestepProvider;   // opt-in timestep protocol (timestep_protocol.cuh)
class TimestepConsumer;

/// @brief Context for one-time component build (after Runtime + DeviceFrame exist).
///
/// Passed to Component::build(). Components may perform cudaMalloc,
/// cudaMemcpy, cudaStreamSynchronize during build -- any host/device work.
///
/// @see Component::build()
struct BuildContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
};

/// @brief Energy / virial a component consumes on the steps with
///        `step % every == phase` (`every` = 0: on no cadence step — only on
///        the run's reported final step and at setup, which evaluate the
///        energy always and the virial whenever any component demands it).
struct EvaluationDemand {
    bool energy = false;
    bool virial = false;
    int every = 0;
    int phase = 0;
};

/// @brief Context for per-step GPU work submission.
///
/// Passed to Component::launch() every simulation step. launch() must
/// remain GPU-pure: only kernel launches, cudaMemsetAsync, cudaMemcpyAsync D2D.
///
/// @see Component::launch()
struct LaunchContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
    int stage;   ///< Current Plan slot index (erased — compare via `at()`).
    BigInt step; ///< Current simulation step index (0-based).
    bool initializing_forces = runtime.exec.initializing_forces;
    /// Set only by the minimizer's `final_output` pass (LAMMPS `Min::run`
    /// forced output): `step` is then the completed-iteration count itself.
    bool forced_final = false;
    bool forces_requested = runtime.exec.forces_requested;
    /// Potential energy wanted this step (LAMMPS `eflag_global`). An MD
    /// step clears it unless a consumer's demand falls on the step.
    bool energy_requested = runtime.exec.energy_requested;
    /// Virial wanted this step (LAMMPS `vflag_global`), same rule.
    bool stress_requested = runtime.exec.stress_requested;
    bool trial_evaluation = runtime.exec.trial_evaluation;

    /// @brief Is the component running in `phase` right now? @see cpu::LaunchContext::at
    /// @tparam PhaseEnum The workflow Phase enum the component is bound in.
    /// @param phase The phase to test against.
    /// @return True when the current stage is that phase.
    template <typename PhaseEnum>
    bool at(PhaseEnum phase) const {
        return stage == ATV::slot_of(phase);
    }
};

/// @brief Context for component resource cleanup.
///
/// Passed to Component::destroy() after the simulation loop completes.
/// Components must release all GPU resources allocated in build().
///
/// @see Component::destroy()
struct DestroyContext {
    cuda::Context& runtime; ///< CUDA execution context (streams).
    cuda::Frame& frame;     ///< Device-resident system state + host metadata.
};

/// @brief Abstract base for simulation components.
///
/// A Component is a CPU-side facade for staged algorithmic behavior.
/// It owns configuration, persistent state, and lifecycle, but not the
/// global loop. Components follow a strict **build / launch / destroy**
/// lifecycle managed by the Backend.
///
/// Two registration paths:
/// @code{.cpp}
/// sim.set_observer(comp);                  // uses default_bindings()
/// sim.on(phase).add(comp, order); // explicit phase override
/// @endcode
///
/// @see Driver for the user-facing registration API
/// @see Backend for lifecycle dispatch
class Component {
  public:
    virtual ~Component() = default;

    /// @brief Human-readable component name (used in timing reports).
    /// @return Component identifier string.
    virtual std::string name() const = 0;

    /// @brief RoleFlag set (mirrors cpu::Component). @see facade/component_role.h
    virtual Role role() const { return 0; }

    /// The backend publishes cell caches after these bindings complete.
    virtual bool changes_box(int stage) const { return false; }

    /// @brief True when this component may make an orthorhombic cell
    ///        triclinic during the run (it writes tilt factors).
    ///
    /// Host-side kernel dispatch picks the orthorhombic or general-cell
    /// specialization from `Context::host_box.ortho`, and the host does not
    /// read back a cell the device changed. The backend scans this at
    /// `load()` and keeps `host_box.ortho` false for such a plan, so pair,
    /// neighbor and mask kernels run the general path (which still takes the
    /// orthorhombic branch at runtime while the published cell is
    /// orthorhombic). Default false.
    virtual bool tilts_box() const { return false; }

    virtual bool requires_system_stress() const { return false; }
    virtual bool provides_system_stress() const { return false; }

    /// @brief True when this plugin contributes a system energy term.
    /// @return Dimensionless predicate. Default false; energy kinds override.
    virtual bool provides_energy() const { return false; }
    /// @brief True when this plugin contributes a force / gradient term.
    /// @return Dimensionless predicate. Default false; force kinds override.
    virtual bool provides_force() const { return false; }
    /// @brief True when this plugin assembles Hessian rows.
    /// @return Dimensionless predicate. Default false.
    virtual bool provides_hessian() const { return false; }

    /// @brief What this component consumes from the force field, and how
    ///        often (LAMMPS `Integrate::ev_setup` / `ev_set`).
    ///
    /// The backend ORs the demands whose cadence hits a step into that
    /// step's `energy_requested` / `stress_requested`; producers skip the
    /// accumulation, reductions and atomics of what nobody asked for. The
    /// last step of every run evaluates the energy (and the virial when
    /// anything demands it) so the retrieved Frame carries it. Default: a
    /// component that
    /// `requires_system_stress()` wants the virial every step.
    virtual EvaluationDemand demand() const {
        EvaluationDemand d;
        if (requires_system_stress()) {
            d.virial = true;
            d.every = 1;
        }
        return d;
    }

    /// @brief Typed accessor for the consumer side of the opt-in
    ///        neighbor protocol (mirrors
    ///        `cpu::Component::as_neighbor_consumer()`).
    ///
    /// Pair evaluators (`cuda::Pair` subclasses) return `this`;
    /// `cuda::md::MD::install_infra_` probes it for the missing-NL gate —
    /// no `dynamic_cast`, no `name()` matching. Default: not a consumer.
    /// @return The consumer view of this component, or `nullptr`.
    virtual const NeighborConsumer* as_neighbor_consumer() const {
        return nullptr;
    }

    /// @brief Typed accessor for the supply side of the opt-in neighbor
    ///        protocol. `cuda::NeighborList` returns `this`. Default:
    ///        not a provider.
    /// @return The provider view of this component, or `nullptr`.
    virtual const NeighborProvider* as_neighbor_provider() const {
        return nullptr;
    }
    /// @brief Typed accessor for split long-range electrostatics.
    ///        `cuda::CoulLong` / `cuda::KSpace` return `this`. Default: none.
    virtual const LongRangeParticipant* as_long_range_participant() const {
        return nullptr;
    }
    /// @brief Supply side of the timestep protocol (the integrators).
    ///        `cuda::md::MD::install_defaults()` probes it. Default: none.
    virtual const TimestepProvider* as_timestep_provider() const {
        return nullptr;
    }
    /// @brief Demand side of the timestep protocol (`cuda::MrecRecorder`).
    ///        Default: not a consumer.
    virtual TimestepConsumer* as_timestep_consumer() { return nullptr; }

    /// @brief Scheduling trait (blocking or deferrable).
    /// @return Scheduling::blocking (default) or Scheduling::deferrable.
    virtual Scheduling scheduling() const { return Scheduling::blocking; }

    /// @brief Execution frequency: 0 = every step, N = every Nth step.
    /// @return Step frequency. 0 means execute on every step.
    virtual int frequency() const { return 0; }

    /// @brief Cadence phase: with `frequency() = N > 0` the backend launches
    ///        this component when `step % N == frequency_phase()`, `step`
    ///        being the zero-based index of the step in progress (the host
    ///        step counter — the gate never reads the device).
    ///
    /// Default 0 (steps 0, N, 2N, …). End-of-step output keyed on the
    /// completed-step count (LAMMPS `update->ntimestep`, incremented at the
    /// top of `Verlet::run`) returns `N - 1`, so it fires after steps
    /// N, 2N, …. A phase outside `[0, N)` matches no step: the component
    /// then runs only where the driver forces it (`samples_final_step`,
    /// `setup_output`) — LAMMPS `thermo 0`.
    virtual int frequency_phase() const { return 0; }

    /// @brief launch() may be captured once into a step graph and replayed.
    ///
    /// True only when launch() enqueues main-stream work that is a pure
    /// function of build-time state, device state and the step's
    /// `energy_requested` / `stress_requested` (never `ctx.step` or other
    /// host state that changes per step; the step is on
    /// `runtime.device_step`). The backend captures runs of such components
    /// into one graph per step signature. Default false: launched directly.
    virtual bool graph_capturable() const { return false; }

    /// @brief Changes whenever state baked into a captured launch changes
    ///        (active count, attached hooks). Part of the step-graph key.
    virtual std::uint64_t graph_epoch() const { return 0; }

    /// @brief Also run on the last step of every run, off cadence (LAMMPS
    ///        prints thermo on the final step). Default false.
    virtual bool samples_final_step() const { return false; }

    /// @brief Default stage bindings. Empty = must bind explicitly via on().add().
    /// @return Vector of DefaultBinding (phase index + order pairs).
    virtual std::vector<DefaultBinding> default_bindings() const { return {}; }

    /// @brief One-time resource preparation.
    ///
    /// Called once after Runtime and DeviceFrame are materialized.
    /// May perform cudaMalloc, D2H reads, sync -- any host/device work.
    /// All persistent GPU resources for repeated launch() must be created here.
    ///
    /// @param ctx Build context with Runtime and DeviceFrame references.
    virtual void build(const BuildContext& ctx) {}

    /// Whether a boundary input edit can preserve this component's lifecycle.
    /// Unknown components reject structural changes before any device mutation.
    virtual bool supports_input_changes(InputChange changes) const {
        return !has_input_change(changes, InputChange::Identity);
    }
    /// Refresh input-derived caches without replaying one-time initialization.
    virtual void invalidate_inputs(InputChange, const BuildContext&) {}

    /// @brief Per-step GPU work submission.
    ///
    /// Called every step during the simulation loop.
    /// Must remain GPU-pure: only kernel launches, cudaMemsetAsync,
    /// cudaMemcpyAsync D2D. Zero cudaMalloc, zero D2H, zero sync.
    ///
    /// @param ctx Launch context with Runtime, DeviceFrame, phase, and step.
    virtual void launch(const LaunchContext& ctx) = 0;

    /// @brief Host-side admission of one in-loop launch, on the stepping
    ///        thread, before its device work is enqueued.
    ///
    /// The backend calls it for every launch that passes the cadence gate
    /// (`frequency()` / `frequency_phase()` / `samples_final_step()`),
    /// whether `launch()` then runs eagerly or the step replays a captured
    /// graph that contains it — so it runs exactly once per in-loop
    /// execution even when `launch()` itself is not called. It must not
    /// touch the device. It is the one place a component may block the
    /// stepping thread, and only for the sanctioned exception to iron law
    /// 10: an asynchronous recorder whose byte budget is full waits here,
    /// on a host condition variable, until its writer frees space (lossless
    /// backpressure); with space it returns at once. A throw stops the run.
    /// Default: nothing.
    /// @param ctx Launch context of the launch being admitted.
    virtual void before_launch(const LaunchContext& ctx) { (void)ctx; }

    /// Opt in to the force-preparation schedule without advancing history.
    virtual void initialize_forces(const LaunchContext& ctx) {}

    /// @brief Output of the current state at the start of a run (LAMMPS
    ///        `Output::setup`, src/output.cpp).
    ///
    /// The MD and minimizer drivers call it once per `run()` — fresh or
    /// continued — after the setup force evaluation and before the first
    /// step, eagerly (never inside a step graph), with `ctx.step` = the
    /// steps / iterations completed so far (LAMMPS `update->ntimestep`; 0 on
    /// a fresh run) and
    /// `ctx.initializing_forces` set (LAMMPS `update->setupflag`). The same
    /// purity rule as `launch()` applies: device work only, no D2H, no sync.
    /// Default: nothing. Recorders override it to write the initial state;
    /// their in-loop end_of_step writes are labelled with the completed-step
    /// count `ctx.step + 1`, so a frame labelled `s` is always the state
    /// after `s` steps.
    virtual void setup_output(const LaunchContext& ctx) {}

    /// @brief Forced output of the final state when a minimization stops
    ///        before its iteration budget (LAMMPS `Min::run`, src/min.cpp:
    ///        on any stop condition but MAXITER, `output->write` runs at the
    ///        final `update->ntimestep` with every dump and thermo due).
    ///
    /// `cuda::minimizer::Minimize` calls it once after its loop — the stop
    /// is already host-visible there, so no device read is added — eagerly,
    /// with `ctx.step` = iterations completed and `ctx.forced_final` set.
    /// Recorders write the current state unless that step was already
    /// written (LAMMPS `last_dump` / `last_thermo`). Same purity rule as
    /// `launch()`. Default: nothing.
    virtual void final_output(const LaunchContext& ctx) {}

    /// @brief Event: the end-of-run retrieve has synchronized the device and
    ///        written the host Frame back.
    ///
    /// The one point after setup where a component may read device-decided
    /// state synchronously (iron law 10) and fail the run: a throw here
    /// propagates out of the driver's `run()`; the host Frame already holds
    /// the retrieved state. Default: nothing.
    ///
    /// @param ctx Runtime and device Frame of the finished run.
    virtual void on_retrieve(const BuildContext& ctx) {}

    /// @brief Resource cleanup.
    ///
    /// Called once after the simulation loop completes.
    /// Must release all resources allocated in build().
    /// Executed in reverse Plan order.
    ///
    /// @param ctx Destroy context with Runtime and DeviceFrame references.
    virtual void destroy(const DestroyContext& ctx) {}

    /// @brief Heap/device bytes this component owns (workspaces, rings,
    ///        comm buffers). Mirrors `cpu::Component::memory_bytes()`
    ///        so both backends expose the same core reporting set; the
    ///        driver-side memory summary consumption lands with
    ///        arch-refactor-04. Default 0 for stateless components.
    virtual std::size_t memory_bytes() const { return 0; }
};

} // namespace cuda
} // namespace ATV
