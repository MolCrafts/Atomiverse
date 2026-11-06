/// @file cpu/component/context.h
/// @brief cpu::Context -- CPU execution context: flat core state plus
/// per-workflow service slices (arch-refactor-05-context).
///
/// The core execution state (threads, MPI identity, timer, the borrowed
/// `loop` slice)
/// stays flat — it is read by every component. Workflow-specific services
/// are grouped into nested plain-aggregate slices, each filled only by its
/// owning preset: `md` (classical force pipeline), `mc`
/// (Metropolis sampling), `opt` (optimizer line search), `capture`
/// (the shared asynchronous capture buffer). Unfilled slice fields stay `nullptr` =
/// service unavailable; components check before use. Cross-slice reads
/// are legitimate (slices group by *publisher*, not by exclusive
/// consumer): `CommBrick::exchange` reads `opt.line_search_trial`, Minimize's
/// force phase reads `md.pair_list`.
///
/// The grouping is zero-cost: one aggregate, stack-constructible,
/// constant-offset member access — no virtual dispatch, no registry, no
/// heap (hot-path invariant of the arch-refactor chain).
#pragma once

#include "atomiverse/core/ghost_contract.h"
#include <functional>
#include <random>
#include <type_traits>

#include "atomiverse/core/iteration_status.h"
#include "atomiverse/core/verdict.h"

namespace ATV {

class TimerRegistry;
class EvaluationTransaction;

namespace cpu {

class CaptureBuffer;
class ReplicaSet;
class Runtime; ///< The Runtime; published on `Context::exec`. @see ExecServices

namespace md {
struct AtomSortState;
struct PairList;
class ChunkForces;
namespace mpi {
struct DomainDecomposition;
} // namespace mpi
} // namespace md

namespace mc {
struct Transaction;
} // namespace mc

class Comm;

/// @brief Classical force-pipeline services (publishers: `Backend`,
///        `NeighborList`, `cpu::md::MD`).
struct DynamicsServices {
    /// @brief Non-owning handle to the MD spatial decomposition.
    /// Allocated by `Backend::init()`, populated by `Backend::set_frame()`.
    /// The full type is deliberately not included here so `cpu::Context`
    /// stays free of MD-specific includes (SCF / minimizer do not need
    /// MPI headers transitively).
    md::mpi::DomainDecomposition* dd = nullptr;

    /// @brief The run's ghost contract — every mounted component's
    ///        `declare_ghosts` folded at `build()`. One handle: each pass
    ///        compiles itself straight off this, so there is no per-wire
    ///        copy to drift. Non-owning; the Driver owns it. Null before
    ///        `build()`, and a `Comm` that finds it null folds the engine's
    ///        base declaration instead.
    /// @see core/ghost_contract.h
    const GhostContract* ghost = nullptr;

    /// @brief Non-owning handle to the atom-sort cadence.
    /// Owned by `NeighborList`; published at `build()`. Null when no list
    /// is mounted. `CommBrick` and the serial neighbor path both stamp
    /// `LaunchContext::step` onto it — they do not increment a private
    /// counter — so a Minimize line-search re-evaluation at a frozen step
    /// cannot walk the sort cadence.
    md::AtomSortState* atom_sort = nullptr;

    /// @brief Non-owning handle to the active neighbor list.
    /// Published by `NeighborList::build()` (address stable across the
    /// run). Kept here — rather than on `cpu::Frame` — so the frame stays
    /// confined to fundamental physical quantities.
    const md::PairList* pair_list = nullptr;

    /// @brief Non-owning handle to the domain communicator (`CommBrick`).
    ///
    /// Published by `communication::DomainRuntime::allocate()`. Address
    /// stable for the run. The swap table is private to `CommBrick`;
    /// Runtime Verlet / linear wrap and plugin packers call this pointer.
    ///
    /// **Null contract:** null only before `allocate()` (or after
    /// DomainRuntime destruction). Serial methods no-op with `n_ghost == 0`.
    Comm* comm = nullptr;

    /// Independent full CSR for many-body consumers; pair_list always means half.
    const md::PairList* full_pair_list = nullptr;

    /// @brief Non-owning handle to the Runtime's per-thread force
    ///        accumulators (`cpu/potentials/force_scratch.h`).
    ///
    /// Published by `Runtime::init()`; the Runtime reduces them into the
    /// frame once at the end of each slot walk, so every threaded kernel of
    /// a step shares one zero and one reduce pass. Null outside a Runtime:
    /// kernels then run with a team of one (`md::ForcePass`).
    md::ChunkForces* chunk_forces = nullptr;

    /// @brief LAMMPS `neighbor->skin` [Å]: the relative displacement the
    ///        active `NeighborProvider` tolerates between reneighbourings
    ///        (`NeighborProvider::validated_skin()`).
    ///
    /// Published by the Runtime with the provider's views, before any
    /// component runs. An owned atom lies at most `skin / 2` outside its
    /// subdomain until the next exchange, which is what a consumer that
    /// sizes per-rank storage from atom positions (KSpacePPPM's ghost
    /// brick, LAMMPS `Grid3d::set_distance`) must allow for. 0 without a
    /// provider: every step then reneighbours and exchanges.
    double neighbor_skin = 0.0;

    /// @brief Global degrees of freedom, `3N - 3 - n_constraints`.
    ///
    /// **The single owner.** Published by `Driver::setup_pass()` after the frame
    /// is bound and BEFORE any component builds, because thermostats cache it in
    /// their own `build()`. Every consumer of Ndof reads it from here — Thermo,
    /// InitVelocity, NoseHoover, NoseHooverNPT, FixCSVR — instead of each
    /// recomputing `3N - 3` for itself, which is what they all did until constraints
    /// arrived and made that formula wrong.
    ///
    /// Temperature is `T = 2 KE / (Ndof kB)`, so an Ndof that ignores constraints
    /// does not merely mis-report T: it makes every thermostat drive the system to
    /// the wrong setpoint. Rigid water would run ~33 % cold.
    int n_dof_global = 0;

    /// @brief True only while the final step of the current `run(n_steps)`
    /// call is executing. Set by the Dynamics driver loop before the last
    /// `step_once`, cleared otherwise. Lets cadence-gated end-of-step
    /// collectives (e.g. `mpi::Reduce` under `set_scalar_reduce_every > 1`)
    /// guarantee globally-reduced scalars are valid when `run()` returns.
    bool is_last_step = false;

    /// @brief LAMMPS `Neighbor::decide()` result (`nflag`) for this step.
    ///
    /// MD Verlet (`Runtime::step` when `Plan::kind == PhaseKind::MD`)
    /// branches on this: `false` → `forward_comm` only; `true` →
    /// `exchange` + `borders` + `neighbor->build`. Default `true` so
    /// linear drivers that never call decide still rebuild ghosts.
    bool reneighbor = true;

    /// @brief A fix forces the next step to reneighbor (LAMMPS
    ///        `Fix::force_reneighbor` / `next_reneighbor`, honoured first in
    ///        `Neighbor::decide`).
    ///
    /// Set by a component whose next `pre_exchange` must run (NvtSllod after
    /// deciding a box flip); MD Verlet consumes and clears it in the next
    /// reneighbor decision. Every rank sets it identically.
    bool reneighbor_requested = false;
};

/// @brief Metropolis-sampling services (publisher: `cpu::mc::MC::run`).
struct SamplingServices {
    /// @brief Non-owning handle to the active MC trial-move changeset.
    /// Address-stable for the run; a `Mover` records its proposed
    /// displacement here each `propose` phase, and the driver reads
    /// `old_*` to roll back a rejected move. Null on non-sampling drivers
    /// (mirrors `md.pair_list` — kept off `cpu::Frame` for Frame purity).
    mc::Transaction* proposal = nullptr;

    /// @brief Non-owning handle to the sampling driver's single seeded RNG
    /// stream. A `Mover` draws atom picks and displacements from it so the
    /// whole trajectory is reproducible from the driver's one seed.
    std::mt19937_64* rng = nullptr;
};

/// @brief Optimizer line-search services (publishers: `cpu::minimizer::Minimize::run`
///        and the active Optimizer component).
struct OptimizeServices {
    /// @brief Re-entrant force re-evaluation handle.
    ///
    /// Published by `cpu::minimizer::Minimize::run` after `setup_pass`; null on
    /// every other driver. Invoking it re-runs the driver's force-phase
    /// bindings (ZeroForces/ZeroEnergy → evaluator → MPI reduce) on the
    /// frame's *current* geometry. An Optimizer component calls it during
    /// the `line_search` phase to evaluate trial geometries `x + alpha p`
    /// without owning a loop — the Driver still owns the outer loop.
    /// `need_grad` is reserved for a future energy-only fast path. Always
    /// check for non-null before calling.
    std::function<void(bool need_grad)> reeval_forces = nullptr;

    /// @brief True while an Optimizer is evaluating line-search trial
    /// geometries via `reeval_forces`. `CommBrick::exchange` skips when
    /// set: the optimizer's per-row buffers are
    /// keyed to the current owned-row order, so reshuffling rows mid-search
    /// would corrupt it. Halo exchange still runs, keeping trial forces
    /// correct for the bounded step.
    bool line_search_trial = false;

    /// @brief Published by the Runtime when a Minimizer plan is compiled: true
    /// when every op of the force phase is `Component::trial_invariant()`, so
    /// a line-search trial's forces/energy at the accepted geometry are the
    /// committed ones.
    bool trial_forces_reusable = false;

    /// @brief Set by an Optimizer during `line_search` when the frame already
    /// holds the forces and energy of the geometry it just committed (the
    /// accepted trial was the last one evaluated and `trial_forces_reusable`
    /// holds). The Runtime then skips that step's force phase and clears the
    /// flag; it is also cleared at the start of every linear step.
    bool forces_current = false;
};

/// @brief Coupled-replica (band) services — publisher: `cpu::PathDriver::run`.
///
/// The slice through which a coupling component reaches the replicas it is
/// coupling to. Null on every other driver, exactly like `md.pair_list` and
/// `mc.proposal`. @see cpu/replica/replica_set.h
struct PathServices {
    /// @brief Non-owning handle to the band. Address-stable for the run.
    cpu::ReplicaSet* replicas = nullptr;

    /// @brief Which replica the backend's active frame currently IS.
    ///
    /// The Path driver runs the `force` phase once per replica, re-pointing the
    /// backend at each replica's Frame in turn; during those passes this says
    /// which one. In the band-wide phases (`couple`, `search_direction`,
    /// `convergence_check`) it is -1: those components see the whole set through
    /// `replicas`, not the one active frame.
    int active = -1;
};

/// @brief The active loop node's status + metrics (publisher: the running Driver).
///
/// Non-owning: a Driver owns its own IterationStatus and ConvergenceMetrics and points these
/// here for the duration of its loop, restoring the previous pair on exit. So a nested
/// driver's components write to the NESTED loop's status, never the parent's.
struct LoopServices {
    /// @brief Stop channel. A component may set `stop_requested`; it must NOT set
    /// `converged` — declaring success is a Criterion's job, not a component's.
    IterationStatus* status = nullptr;

    /// @brief Where components publish what they measured, for criteria to judge.
    ConvergenceMetrics* metrics = nullptr;
};

/// @brief The Runtime executing this node tree (publisher: `Backend::init()`).
///
/// **A Driver does not own a Runtime — it is handed to one to execute.** That is what
/// makes nesting cheap: a sub-driver runs its own Plan on its PARENT's Backend, so the
/// Backend does not double, the domain decomposition is not rebuilt, the frame is not
/// re-bound, and MPI does not split. One Runtime, one Context, one DD — however deep the
/// loop tree goes.
///
/// Non-owning, and deliberately a back-pointer: the Backend owns the Context, and
/// publishes itself here in `init()` so a node reached through a `LaunchContext` can ask
/// the Runtime to execute a Plan. @see cpu::DriverNode
struct ExecServices {
    /// @brief The Backend running this node tree. Null outside a run.
    Runtime* runtime = nullptr;
    /// Physical evaluation transaction; nested solves borrow this ledger.
    EvaluationTransaction* evaluation_transaction = nullptr;
    /// Full replicas opt out of spatial decomposition. This does not establish
    /// an electronic communicator; collective methods must check that contract.
    bool serial_on_replica = false;

    /// @brief The force-request veto channel — how an energy-only phase reaches
    ///        the slots of the drivers nested inside it.
    ///
    /// A slot flag alone (`StageSlot::forces_requested`) stops at the node
    /// boundary: a nested driver runs its OWN Plan, whose slots were stamped by
    /// ITS `phase_wants_forces` — and a method loop asks for forces at every SCF
    /// stage. So the Runtime ANDs the outer slot's flag into this field around a
    /// nested dispatch (saving and restoring it, so a later phase of the same
    /// step is unaffected), and every `LaunchContext` it builds carries
    /// `slot.forces_requested && exec.forces_requested`. The AND is what
    /// propagates one outer veto through arbitrary nesting depth — and what
    /// keeps the same node under MD's `force` phase assembling its gradient.
    ///
    /// True outside any veto, which is every loop but MC's trial `energy` phase.
    bool forces_requested = true;
    /// @brief Stress-request veto propagated through nested driver dispatch.
    bool stress_requested = true;
    /// Evaluate the current nuclear state without advancing propagation history.
    bool initializing_forces = false;
    /// A reversible energy/force trial must not advance temporal history.
    bool trial_evaluation = false;
};

/// @brief Asynchronous-capture services (publisher: `cpu::Runtime::init`).
struct CaptureServices {
    /// @brief Non-owning handle to the driver tree's shared capture buffer
    /// (`cpu::CaptureBuffer`, owned by the Runtime). Every asynchronous
    /// consumer (`cpu::AsyncConsumer`) subscribes to it at build.
    CaptureBuffer* buffer = nullptr;
};

/// @brief CPU execution context: flat core state + per-workflow slices.
struct Context {
    int n_threads = 0; ///< OpenMP threads (0 = auto-detect).
    int mpi_rank = 0;  ///< MPI rank within communicator (0 if no MPI).
    int mpi_size = 1;  ///< MPI communicator size (1 if no MPI).

    /// @brief Non-owning handle to the per-Backend timing registry.
    /// Published by `Backend::init()`. Components reach it (via
    /// `ATV_PROFILE_SCOPE`) to record sub-region timings; null / disabled by
    /// default so timing is opt-in and zero-cost when off. @see core/timer.h
    ATV::TimerRegistry* timer = nullptr;

    /// @brief The **active loop node's** status and metrics.
    ///
    /// Both are OWNED BY THE LOOP, not by the Context — the Context only points at
    /// whichever loop is currently running. That indirection is the entire fix for a bug
    /// that shipped twice:
    ///
    ///   - probe 2: `LBFGS` wrote `converged = true` onto the one shared status;
    ///   - probe 3: so the first NEB image to converge would have stopped the whole chain.
    ///
    /// With the status living on the loop node, a nested loop physically cannot reach its
    /// parent's — it is not a rule to remember, it is a pointer that does not go there.
    ///
    /// A component **measures** into `metrics` and may **request a stop** via `status`.
    /// It does not declare convergence: that is a `Criterion`'s job.
    LoopServices loop;

    ExecServices exec;       ///< The Runtime executing this node tree.
    DynamicsServices md;     ///< Classical force-pipeline slice.
    SamplingServices mc;     ///< Metropolis-sampling slice.
    OptimizeServices opt;    ///< Optimizer line-search slice.
    PathServices path;       ///< Coupled-replica (band) slice.
    CaptureServices capture; ///< Asynchronous-capture slice.
};

// Zero-cost-grouping guards: the slices stay plain aggregates and the
// Context stays non-polymorphic (constant-offset access, stack
// construction). OptimizeServices is exempt from trivial copyability —
// its std::function predates this grouping.
static_assert(std::is_trivially_copyable_v<DynamicsServices>);
static_assert(std::is_trivially_copyable_v<SamplingServices>);
static_assert(std::is_trivially_copyable_v<PathServices>);
static_assert(std::is_trivially_copyable_v<LoopServices>);
static_assert(std::is_trivially_copyable_v<ExecServices>);
static_assert(std::is_trivially_copyable_v<CaptureServices>);
static_assert(!std::is_polymorphic_v<Context>);

} // namespace cpu
} // namespace ATV
