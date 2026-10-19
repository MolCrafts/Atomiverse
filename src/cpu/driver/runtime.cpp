/// @file cpu/driver/runtime.cpp
/// @brief cpu::Runtime implementation. Plan dispatch + component lifecycle;
///        DD / device / analysis state lives in the composed sub-runtimes.
// Communication/sort scheduling follows LAMMPS src/verlet.cpp,
// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b. Copyright (2003) Sandia
// Corporation. Under Contract DE-AC04-94AL85000, the U.S. Government retains
// certain rights. Distributed under the GNU General Public License.

#include "atomiverse/cpu/driver/runtime.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/pbc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <utility>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

namespace {

/// LAMMPS `Neighbor::check_distance` ends in an
/// `MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world)`
/// (`src/neighbor.cpp`): the reneighbour decision is collective, because
/// the displacement trigger is a per-rank quantity. A rank-local flag
/// deadlocks the step — the ranks that decided `true` enter
/// exchange/borders while the rest post the refresh pattern, and the two
/// message shapes never match.
void sync_reneighbor_flag(cpu::Context& context) {
#ifdef ATV_HAS_MPI
    if (context.mpi_size <= 1 || context.md.dd == nullptr
        || context.md.dd->cart_comm == MPI_COMM_NULL)
        return;
    int flag = context.md.reneighbor ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX,
                  context.md.dd->cart_comm);
    context.md.reneighbor = (flag != 0);
#else
    (void)context;
#endif
}

} // namespace

Runtime::Runtime() = default;
Runtime::~Runtime() { shutdown(); } // idempotent; releases cart_comm + executor
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

void Runtime::init() {
    failed_ = false;
    // Device: OpenMP thread configuration (+ NUMA-bind warning).
    device_.configure_threads(context_);

#ifdef ATV_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &context_.mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &context_.mpi_size);
    }
#endif

    // Rank-aware logging auto-init: gates console to rank 0 and honors ATV_LOG_*
    // env knobs. Skipped if the application already called `ATV::log_init(...)`.
    if (!ATV::log_initialized()) {
        LogConfig lc;
        lc.mpi_rank = context_.mpi_rank;
        lc.mpi_size = context_.mpi_size;
        ATV::log_init(lc);
    }

#ifdef ATV_HAS_MPI
    ATV_INFO("CPU runtime initialized: {} threads, MPI rank {}/{}",
             context_.n_threads, context_.mpi_rank, context_.mpi_size);
#else
    ATV_INFO("CPU runtime initialized: {} threads", context_.n_threads);
#endif

    // Device: publish the timing registry + apply the enable preference.
    device_.init_timing(context_);

    // Publish the Runtime. A node reached through a LaunchContext asks the
    // Runtime here to run its own Plan — this is what lets a sub-driver nest.
    context_.exec.runtime = this;

    // Communication: allocate + publish the DomainDecomposition; build topology.
    domain_.allocate(context_);

    // Asynchronous: start the analysis pipeline.
    analysis_.start(context_);
}

void Runtime::set_frame(cpu::Frame& frame) {
    bool needs_tag_index = false;
    if (plan_) {
        for (const auto* component : plan_->components()) {
            if (multi_rank_dd(context_) && component->requires_full_frame())
                throw std::runtime_error(
                    component->name()
                    + ": requires a complete Frame; use "
                      "set_serial_on_replica() "
                      "before build, or run without spatial decomposition");
            needs_tag_index |= component->requires_tag_index();
        }
    }
    if (plan_ && plan_->requires_full_frame && multi_rank_dd(context_))
        throw std::runtime_error("driver requires a complete Frame; use "
                                 "set_serial_on_replica() before build");
    domain_.bind_frame(frame, context_, needs_tag_index);
    frame_ = &frame;
    forces_ready_ = false;
}

void Runtime::load(const Plan<cpu::Component>& plan, int step_idx) {
    plan_ = &plan;
    if (components_built_)
        refresh_plan_(step_idx);
}

void Runtime::adopt_dd(std::unique_ptr<md::mpi::DomainDecomposition> dd) {
    domain_.adopt(std::move(dd), context_);
}

std::unique_ptr<md::mpi::DomainDecomposition> Runtime::release_dd() {
    return domain_.release(context_);
}

void Runtime::build_components() {
    try {
        build_plan(*plan_);
    } catch (...) {
        if (driver_owned_)
            unwind_failed_run();
        throw;
    }
    try {
        refresh_plan_();
        components_built_ = true;
    } catch (...) {
        if (driver_owned_)
            unwind_failed_run();
        else
            destroy_plan(*plan_);
        throw;
    }
}

void Runtime::refresh_plan_(int step_idx) {
    contracts_ready_ = false;
    std::vector<StageSlot<cpu::Component>> force_bindings;
    if (plan_->kind == PhaseKind::MD) {
        for (int phase = slot_of(ATV::md::Phase::pre_force);
             phase <= slot_of(ATV::md::Phase::post_force); ++phase)
            force_bindings.push_back(plan_->slots[phase]);
    }
    if (force_bindings != force_bindings_)
        forces_ready_ = false;
    force_bindings_ = std::move(force_bindings);
    analysis_.drain();
    TimerRegistry& timing = device_.timing();
    // Register one timing handle per op (component name + phase + category
    // rollup), precomputed so the hot step loop never re-classifies / allocates.
    op_handle_.assign(plan_->slots.size(), {});
    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        auto& slot = plan_->slots[s];
        op_handle_[s].reserve(slot.ops.size());
        for (auto& op : slot.ops) {
            const std::string nm = op.component->name();
            op_handle_[s].push_back(timing.register_op(
                TimerKey{nm, slot.phase,
                         static_cast<int>(md::StepTiming::classify(nm)), ""}));
        }
    }
    ATV_INFO("all components built");
    const int comm_cat = static_cast<int>(md::StepTiming::Comm);
    comm_handle_fwd_ =
        timing.register_op(TimerKey{"comm_forward", -1, comm_cat, ""});
    comm_handle_exch_ =
        timing.register_op(TimerKey{"comm_exchange", -1, comm_cat, ""});
    comm_handle_bord_ =
        timing.register_op(TimerKey{"comm_borders", -1, comm_cat, ""});
    comm_handle_rev_ =
        timing.register_op(TimerKey{"comm_reverse", -1, comm_cat, ""});
    cache_verlet_comm_();
    // The fold changed: CommBrick compiled its border/forward/reverse wires
    // from the previous `ctx.md.ghost` and caches them until told otherwise.
    // A nested physical child or a live-mounted style that declares a new
    // ghost column (a Coulomb style's charge) would otherwise read rows the
    // halo never filled, silently.
    comm_contract_stale_ = true;
    // The schema is now folded; hold the Frame to it before anything runs.
    // Iron law 4, applied once for the whole component set instead of each
    // component remembering to check its own fields.
    if (frame_ != nullptr) {
        check_ghosts(verlet_.ghost, *frame_, "driver build");
    }
    verlet_setup_done_ = false;
    if (plan_
        && (plan_->kind == PhaseKind::MD || plan_->kind == PhaseKind::Evaluate
            || plan_->kind == PhaseKind::Minimizer))
        setup_verlet_comm_(step_idx);
    contracts_ready_ = true;
}

void Runtime::launch_op_(std::size_t s, std::size_t o, int step_idx) {
    DispatchScope dispatch(this);
    auto& slot = plan_->slots[s];
    auto& op = slot.ops[o];
    const int saved_slot = active_slot_;
    struct RestoreSlot {
        int& slot;
        int saved;
        bool& pipeline;
        bool saved_pipeline;
        ~RestoreSlot() {
            slot = saved;
            pipeline = saved_pipeline;
        }
    } restore_slot{active_slot_, saved_slot, physical_pipeline_active_,
                   physical_pipeline_active_};
    active_slot_ = static_cast<int>(s);
    physical_pipeline_active_ |=
        plan_->kind == PhaseKind::Evaluate
        || plan_->kind == PhaseKind::Minimizer
        || (plan_->kind == PhaseKind::MD
            && active_slot_ != slot_of(ATV::md::Phase::end_of_step));
    if (op.frequency > 0 && step_idx % op.frequency != 0)
        return;
    LaunchContext ctx{context_,
                      *frame_,
                      slot.phase,
                      step_idx,
                      /*forces_requested=*/slot.forces_requested
                          && context_.exec.forces_requested,
                      slot.stress_requested && context_.exec.stress_requested};
    const auto saved_window = frame_->resolution_window();
    struct RestoreWindow {
        Frame& frame;
        ResolutionWindow saved;
        ~RestoreWindow() { frame.restore_resolution_window(saved); }
    } restore{*frame_, saved_window};
    frame_->begin_resolution_window(kResolveBudgetPerLaunch);
    TimerRegistry& timing_reg = device_.timing();
    const bool timing = timing_reg.enabled();
    const bool have_handles = s < op_handle_.size() && o < op_handle_[s].size();
    if (timing && have_handles) {
        const auto t0 = std::chrono::steady_clock::now();
        op.component->launch(ctx);
        const auto t1 = std::chrono::steady_clock::now();
        timing_reg.record(op_handle_[s][o],
                          std::chrono::duration<double>(t1 - t0).count());
    } else {
        op.component->launch(ctx);
    }
    frame_->end_resolution_window(op.component->name().c_str());
}

void Runtime::walk_slot_(int phase, int step_idx, bool skip_engine) {
    if (plan_ == nullptr || phase < 0
        || static_cast<std::size_t>(phase) >= plan_->slots.size())
        return;
    const std::size_t s = static_cast<std::size_t>(phase);
    auto& slot = plan_->slots[s];
    for (std::size_t o = 0; o < slot.ops.size(); ++o) {
        if (skip_engine && slot.ops[o].component
            && slot.ops[o].component->verlet_engine())
            continue;
        launch_op_(s, o, step_idx);
    }
}

void Runtime::launch_engine_(cpu::Component* c, int slot, int op, int phase,
                             int step_idx) {
    if (c == nullptr)
        return;
    LaunchContext ctx{context_, *frame_, phase, step_idx,
                      /*forces_requested=*/true};
    if (plan_ && slot >= 0
        && static_cast<std::size_t>(slot) < plan_->slots.size()) {
        const auto& sl = plan_->slots[static_cast<std::size_t>(slot)];
        ctx.forces_requested =
            sl.forces_requested && context_.exec.forces_requested;
        ctx.stress_requested =
            sl.stress_requested && context_.exec.stress_requested;
    }
    const auto saved_window = frame_->resolution_window();
    struct RestoreWindow {
        Frame& frame;
        ResolutionWindow saved;
        ~RestoreWindow() { frame.restore_resolution_window(saved); }
    } restore{*frame_, saved_window};
    frame_->begin_resolution_window(kResolveBudgetPerLaunch);
    TimerRegistry& timing_reg = device_.timing();
    const bool timing = timing_reg.enabled();
    const bool have_handles =
        slot >= 0 && op >= 0 && op_handle_.size() == plan_->slots.size()
        && static_cast<std::size_t>(slot) < op_handle_.size()
        && static_cast<std::size_t>(op)
               < op_handle_[static_cast<std::size_t>(slot)].size();
    if (timing && have_handles) {
        const auto t0 = std::chrono::steady_clock::now();
        c->launch(ctx);
        const auto t1 = std::chrono::steady_clock::now();
        timing_reg.record(op_handle_[static_cast<std::size_t>(slot)]
                                    [static_cast<std::size_t>(op)],
                          std::chrono::duration<double>(t1 - t0).count());
    } else {
        c->launch(ctx);
    }
    frame_->end_resolution_window(c->name().c_str());
}

void Runtime::cache_verlet_comm_() {
    verlet_ = {};
    context_.md.pair_list = nullptr;
    context_.md.full_pair_list = nullptr;
    context_.md.atom_sort = nullptr;
    if (plan_ == nullptr)
        return;
    // The halo's own product comes first, so it keeps the leading wire
    // positions no matter which components are mounted.
    verlet_.ghost.declare(base_fields::kEngine);
    context_.md.ghost = &verlet_.ghost;
    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        const auto& slot = plan_->slots[s];
        for (std::size_t o = 0; o < slot.ops.size(); ++o) {
            cpu::Component* c = slot.ops[o].component;
            if (c == nullptr)
                continue;
            if (cpu::NeighborProvider* np = c->neighbor_provider()) {
                if (verlet_.neighbor_comp == c)
                    continue;
                if (verlet_.neighbor_comp)
                    throw std::runtime_error(
                        "runtime plan has multiple NeighborProvider owners");
                (void)np->validated_skin();
                verlet_.neighbor = np;
                verlet_.neighbor_comp = c;
                verlet_.neighbor_slot = static_cast<int>(s);
                verlet_.neighbor_op = static_cast<int>(o);
            }
        }
    }
    auto active_components = plan_->components();
    std::unordered_set<cpu::Component*> seen(active_components.begin(),
                                             active_components.end());
    for (std::size_t i = 0; i < active_components.size(); ++i) {
        auto* node = active_components[i]->nested_node();
        const auto* child = node ? node->execution_plan() : nullptr;
        if (!child || child->kind == PhaseKind::MD
            || child->kind == PhaseKind::Evaluate
            || child->kind == PhaseKind::Minimizer
            || child->kind == PhaseKind::Sampling)
            continue;
        for (auto* component : child->components())
            if (seen.insert(component).second)
                active_components.push_back(component);
    }
    bool want_half = false, want_full = false, want_ghost_rows = false;
    for (const auto* c : active_components) {
        if (const auto* consumer = c->as_neighbor_consumer()) {
            if (verlet_.neighbor)
                verlet_.neighbor->validate_coverage(
                    *consumer, verlet_.neighbor_comp->name(), c->name());
            if (consumer->requires_ghost_neighbor_rows()
                && consumer->neighbor_list_mode() != NeighborListMode::Full)
                throw std::runtime_error(
                    "ghost-centred half neighbour lists are not supported");
            want_ghost_rows |= consumer->requires_ghost_neighbor_rows();
            want_half |=
                consumer->neighbor_list_mode() == NeighborListMode::Half;
            want_full |=
                consumer->neighbor_list_mode() == NeighborListMode::Full;
        }
    }
    for (auto* c : active_components)
        if (auto* provider = c->neighbor_provider())
            provider->configure_lists(want_half || !want_full, want_full,
                                      want_ghost_rows);
    // Preserve Plan order; pointers serve only to deduplicate identities.
    for (const auto* c : active_components) {
        verlet_.ghost.declare(c->declare_ghosts());
        if (const auto* nc = c->as_neighbor_consumer()) {
            const double halo = nc->halo_cutoff();
            if (!std::isfinite(halo) || halo < 0)
                throw std::runtime_error(c->name() + ": invalid halo cutoff");
            verlet_.consumer_halo_cutoff =
                std::max(verlet_.consumer_halo_cutoff, halo);
        }
        if (const auto* np = c->as_neighbor_provider())
            verlet_.provider_reach =
                std::max(verlet_.provider_reach,
                         static_cast<double>(np->neighbor_reach()));
    }
}

void Runtime::apply_ghost_cutoff_() {
    if (verlet_.neighbor)
        verlet_.neighbor->activate(context_);
    cpu::Comm* comm = context_.md.comm;
    if (!comm)
        return;
    // A wider coordinate dependency needs the same relative-displacement
    // margin as the neighbor list, not a fixed percentage of its radius.
    const double skin =
        verlet_.neighbor ? verlet_.neighbor->validated_skin() : 0;
    const double reach =
        std::max(verlet_.provider_reach, verlet_.consumer_halo_cutoff + skin);
    comm->set_ghost_cutoff(reach);
}

bool Runtime::is_linear_force_slot_(int slot) const {
    if (plan_ == nullptr)
        return false;
    if (plan_->kind == PhaseKind::Evaluate)
        return slot == ATV::slot_of(ATV::evaluate::Phase::evaluate);
    if (plan_->kind == PhaseKind::Minimizer)
        return slot == ATV::slot_of(ATV::minimizer::Phase::force);
    if (plan_->kind == PhaseKind::Sampling)
        return slot == ATV::slot_of(ATV::sampling::Phase::energy);
    return false;
}

namespace {

// Setup is outside the measured step loop. Preserve the user's timing setting
// and accumulated production samples, including when initialization throws.
class SuspendTiming {
  public:
    explicit SuspendTiming(TimerRegistry& registry)
        : registry_(registry), enabled_(registry.enabled()) {
        registry_.set_enabled(false);
    }
    ~SuspendTiming() { registry_.set_enabled(enabled_); }

  private:
    TimerRegistry& registry_;
    bool enabled_;
};

/// @brief Time one Runtime-level comm call into its registered handle.
template <typename Fn>
inline void timed_comm(TimerRegistry& reg, int handle, Fn&& fn) {
    if (!reg.enabled() || handle < 0) {
        fn();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    reg.record(handle, std::chrono::duration<double>(t1 - t0).count());
}

} // namespace

void Runtime::comm_refresh_(int slot, int step) {
    apply_ghost_cutoff_();
    cpu::Comm* comm = context_.md.comm;
    if (!comm)
        return;
    LaunchContext comm_ctx{context_, *frame_, slot, step,
                           /*forces_requested=*/true};
    timed_comm(device_.timing(), comm_handle_fwd_,
               [&] { comm->forward_comm(comm_ctx); });
}

void Runtime::decide_neighbor_() {
    // Without a provider nothing can judge the Verlet skin, so every step
    // migrates and rebuilds its ghosts (the conservative default of
    // `NeighborProvider::decide_reneighbor`); a bonded-only or electronic
    // force on several ranks still needs migration and a fresh halo.
    // MC proposals/rollbacks are discontinuous changes, not MD drift. Their
    // energy phase always rebuilds, independent of the MD delay/every cadence.
    // LAMMPS Min::init drops delay/every/check for a minimization; only an MD
    // plan runs the cadence.
    const bool md_cadence = plan_ && plan_->kind == PhaseKind::MD;
    context_.md.reneighbor =
        !verlet_.neighbor || (plan_ && plan_->kind == PhaseKind::Sampling)
        || verlet_.neighbor->decide_reneighbor(context_, *frame_, md_cadence);
    sync_reneighbor_flag(context_);
}

void Runtime::comm_rebuild_(int exchange_slot, int borders_slot, int step) {
    LaunchContext comm_ctx{context_, *frame_, exchange_slot, step,
                           /*forces_requested=*/true};
    // The caller may mutate Frame::box between continuable runs. Refresh
    // fractional/Cartesian slab bounds at the same boundary as the halo,
    // including orthorhombic-to-triclinic transitions.
    if (context_.md.dd)
        context_.md.dd->compute_sublo_subhi(frame_->box);
    // LAMMPS Domain::pbc before Comm::exchange (src/verlet.cpp,
    // a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b). Copyright (2003)
    // Sandia Corporation. GPL-2.0-only. Migration and explicit periodic
    // ghost construction require owned rows in the primary cell. CellList
    // then bins Cartesian owner/image rows without applying another image.
    // Without either consumer, preserve molecular cluster coordinates.
    // Line-search trials retain the optimizer's coordinates.
    if (!context_.opt.line_search_trial
        && (multi_rank_dd(context_) || verlet_.neighbor_comp != nullptr))
        md::wrap_owned(*frame_, owned_count(comm_ctx));
    apply_ghost_cutoff_();
    cpu::Comm* comm = context_.md.comm;
    TimerRegistry& reg = device_.timing();
    if (comm && comm_contract_stale_) {
        // Recompile the wires off the current fold before the first exchange
        // that will use them; the ghost shell below is rebuilt from scratch.
        comm->setup(comm_ctx);
        comm_contract_stale_ = false;
    }
    if (comm)
        timed_comm(reg, comm_handle_exch_, [&] { comm->exchange(comm_ctx); });
    // LAMMPS Verlet::run: exchange -> atom->sort -> borders -> neighbor.
    // Sorting owns no communication policy; every rank count follows this
    // one boundary, before borders replaces all row-indexed send lists.
    if (auto* sort = context_.md.atom_sort) {
        sort->step = step;
        if (sort->due()) {
            if (md::sort_owned_atoms(
                    *frame_, frame_->nlocal(),
                    sort->resolved_binsize(verlet_.provider_reach)))
                md::notify_sorted(context_.md.dd, *frame_, nullptr);
            sort->mark_sorted();
        }
    }
    comm_ctx.stage = borders_slot;
    if (comm)
        timed_comm(reg, comm_handle_bord_, [&] { comm->borders(comm_ctx); });
}

void Runtime::comm_before_force_(int slot, int step_idx) {
    if (!context_.md.reneighbor) {
        comm_refresh_(slot, step_idx);
        return;
    }
    comm_rebuild_(slot, slot, step_idx);
    if (verlet_.neighbor_comp && plan_ != nullptr && verlet_.neighbor_slot >= 0
        && static_cast<std::size_t>(verlet_.neighbor_slot)
               < plan_->slots.size()) {
        const auto& nslot =
            plan_->slots[static_cast<std::size_t>(verlet_.neighbor_slot)];
        launch_engine_(verlet_.neighbor_comp, verlet_.neighbor_slot,
                       verlet_.neighbor_op, nslot.phase, step_idx);
    }
}

void Runtime::comm_after_force_(int slot, int step_idx) {
    cpu::Comm* comm = context_.md.comm;
    if (!comm)
        return;
    TimerRegistry& reg = device_.timing();
    LaunchContext comm_ctx{context_, *frame_, slot, step_idx,
                           /*forces_requested=*/true};
    timed_comm(reg, comm_handle_rev_, [&] { comm->reverse_comm(comm_ctx); });
}

void Runtime::setup_verlet_comm_(int step_idx) {
    // LAMMPS Verlet::setup (src/verlet.cpp): pbc + exchange + borders +
    // neighbor->build before the run loop, on every rank count, so the first
    // force sees wrapped owned rows and the first nflag==0 step can
    // forward_comm. Copyright (2003) Sandia Corporation. GPL-2.0-only.
    if (verlet_setup_done_)
        return;
    SuspendTiming suspend_timing(device_.timing());
    context_.md.reneighbor = true;
    comm_rebuild_(ATV::slot_of(ATV::md::Phase::pre_exchange),
                  ATV::slot_of(ATV::md::Phase::pre_neighbor), step_idx);
    if (verlet_.neighbor_comp) {
        launch_engine_(verlet_.neighbor_comp, verlet_.neighbor_slot,
                       verlet_.neighbor_op,
                       ATV::slot_of(ATV::md::Phase::neighbor), step_idx);
    }
    verlet_setup_done_ = true;
}

void Runtime::step_linear_(int step_idx) {
    const bool wrap = plan_
                      && (plan_->kind == PhaseKind::Evaluate
                          || plan_->kind == PhaseKind::Minimizer
                          || plan_->kind == PhaseKind::Sampling);
    if (!wrap) {
        for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
            if (plan_->slots[s].ops.empty())
                continue;
            for (std::size_t o = 0; o < plan_->slots[s].ops.size(); ++o)
                launch_op_(s, o, step_idx);
        }
        return;
    }

    // LAMMPS Min::energy_force (src/min.cpp): nflag XOR on comm around
    // the force evaluation; reverse_comm if newton. GPL-2.0-only.
    const int force_slot = plan_->kind == PhaseKind::Evaluate
                               ? ATV::slot_of(ATV::evaluate::Phase::evaluate)
                           : plan_->kind == PhaseKind::Sampling
                               ? ATV::slot_of(ATV::sampling::Phase::energy)
                               : ATV::slot_of(ATV::minimizer::Phase::force);

    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        if (static_cast<int>(s) == force_slot) {
            decide_neighbor_();
            comm_before_force_(force_slot, step_idx);
        }
        walk_slot_(static_cast<int>(s), step_idx, /*skip_engine=*/true);
        if (static_cast<int>(s) == force_slot)
            comm_after_force_(force_slot, step_idx);
    }
}

void Runtime::step_verlet_(int step_idx) {
    // Structural copy of LAMMPS Verlet::run (src/verlet.cpp) communication
    // block on ctx.md.comm: nflag = neighbor->decide(); if 0: forward_comm;
    // else exchange + borders + neighbor->build; reverse_comm if newton.
    // Copyright (2003) Sandia Corporation. GPL-2.0-only.
    using P = ATV::md::Phase;
    walk_slot_(ATV::slot_of(P::initial_integrate), step_idx, true);
    walk_slot_(ATV::slot_of(P::post_integrate), step_idx, true);

    decide_neighbor_();
    const bool nflag = context_.md.reneighbor;

    const int pre_n = ATV::slot_of(P::pre_neighbor);
    const int neigh = ATV::slot_of(P::neighbor);
    cpu::Comm* comm = context_.md.comm;

    if (!nflag) {
        comm_refresh_(pre_n, step_idx);
    } else {
        walk_slot_(ATV::slot_of(P::pre_exchange), step_idx, true);
        comm_rebuild_(ATV::slot_of(P::pre_exchange), pre_n, step_idx);
        walk_slot_(pre_n, step_idx, true);
        if (verlet_.neighbor_comp) {
            launch_engine_(verlet_.neighbor_comp, verlet_.neighbor_slot,
                           verlet_.neighbor_op, neigh, step_idx);
        }
        walk_slot_(neigh, step_idx, true);
    }

    walk_slot_(ATV::slot_of(P::pre_force), step_idx, true);
    walk_slot_(ATV::slot_of(P::force), step_idx, true);

    if (comm) {
        LaunchContext comm_ctx{context_, *frame_, ATV::slot_of(P::post_force),
                               step_idx,
                               /*forces_requested=*/true};
        timed_comm(device_.timing(), comm_handle_rev_,
                   [&] { comm->reverse_comm(comm_ctx); });
    }
    walk_slot_(ATV::slot_of(P::post_force), step_idx, true);
    walk_slot_(ATV::slot_of(P::final_integrate), step_idx, true);
    walk_slot_(ATV::slot_of(P::end_of_step), step_idx, true);
    // Stamp after end_of_step: a barostat or Deform rescaling the cell there
    // is the loop's own update (LAMMPS fix press/berendsen integrates the next
    // step on the forces it has), while a cell edit between `run()` calls
    // still differs from this stamp and re-initializes forces in prepare_run.
    force_box_revision_ = frame_->box.revision();
}

void Runtime::prepare_run(int step_idx) {
    DispatchScope dispatch(this);
    if (plan_ && plan_->kind == PhaseKind::MD
        && !context_.exec.forces_requested)
        throw std::runtime_error(
            "MD requires forces; it cannot run inside an energy-only request");
    if (plan_ && plan_->kind == PhaseKind::MD && forces_ready_
        && force_box_revision_ != frame_->box.revision()) {
        // An external cell edit must invalidate forces before the first
        // half kick, not merely rebuild neighbors later in that step.
        forces_ready_ = false;
        contracts_ready_ = false;
    }
    if (!contracts_ready_)
        refresh_plan_(step_idx);
    if (!plan_ || plan_->kind != PhaseKind::MD || forces_ready_)
        return;
    SuspendTiming suspend_timing(device_.timing());
    // LAMMPS src/verlet.cpp::setup, a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
    // Copyright (2003) Sandia Corporation. GPL-2.0-only.
    const bool saved = context_.exec.initializing_forces;
    struct Restore {
        bool& flag;
        bool saved;
        bool& ready;
        bool& contracts;
        ~Restore() {
            flag = saved;
            // A failed force assembly may leave ghost accumulators dirty.
            // Retry through the existing halo/list setup, never reuse them.
            if (!ready)
                contracts = false;
        }
    } restore{context_.exec.initializing_forces, saved, forces_ready_,
              contracts_ready_};
    context_.exec.initializing_forces = true;
    for (int phase = slot_of(ATV::md::Phase::pre_force);
         phase <= slot_of(ATV::md::Phase::end_of_step); ++phase) {
        if (phase == slot_of(ATV::md::Phase::final_integrate))
            continue;
        if (phase == slot_of(ATV::md::Phase::post_force))
            comm_after_force_(phase, step_idx);
        const auto& slot = plan_->slots[phase];
        const int saved_slot = active_slot_;
        struct RestoreSlot {
            int& slot;
            int saved;
            bool& pipeline;
            bool saved_pipeline;
            ~RestoreSlot() {
                slot = saved;
                pipeline = saved_pipeline;
            }
        } restore_slot{active_slot_, saved_slot, physical_pipeline_active_,
                       physical_pipeline_active_};
        active_slot_ = phase;
        physical_pipeline_active_ = true;
        for (const auto& op : slot.ops) {
            if (op.component->verlet_engine())
                continue;
            LaunchContext ctx{
                context_,
                *frame_,
                phase,
                step_idx,
                slot.forces_requested && context_.exec.forces_requested,
                slot.stress_requested && context_.exec.stress_requested};
            const auto window = frame_->resolution_window();
            struct RestoreWindow {
                Frame& frame;
                ResolutionWindow saved;
                ~RestoreWindow() { frame.restore_resolution_window(saved); }
            } restore_window{*frame_, window};
            frame_->begin_resolution_window(kResolveBudgetPerLaunch);
            if (phase == slot_of(ATV::md::Phase::force))
                op.component->launch(ctx);
            else
                op.component->initialize_forces(ctx);
            frame_->end_resolution_window(op.component->name().c_str());
        }
    }
    force_box_revision_ = frame_->box.revision();
    forces_ready_ = true;
}

void Runtime::step(int step_idx) {
    DispatchScope dispatch(this);
    prepare_run(step_idx);
    TimerRegistry& timing_reg = device_.timing();
    const bool timing = timing_reg.enabled();
    const auto step_t0 = timing ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    if (plan_ && plan_->kind == PhaseKind::MD)
        step_verlet_(step_idx);
    else
        step_linear_(step_idx);
    if (timing) {
        const auto step_t1 = std::chrono::steady_clock::now();
        timing_reg.set_loop(
            std::chrono::duration<double>(step_t1 - step_t0).count(),
            static_cast<int>(frame_->nlocal()),
            static_cast<int>(frame_->nghost()));
        timing_reg.tick_step();
    }
}

void Runtime::step_stage(int slot, int step_idx) {
    DispatchScope dispatch(this);
    if (!plan_ || slot < 0
        || static_cast<std::size_t>(slot) >= plan_->slots.size())
        throw std::out_of_range("Runtime::step_stage: invalid phase");
    if (plan_->kind == PhaseKind::MD)
        throw std::logic_error("MD stages require the coordinated Verlet step");
    prepare_run(step_idx);
    auto& timing = device_.timing();
    const auto start = timing.enabled()
                           ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    const bool physical = plan_->kind == PhaseKind::Evaluate
                          || plan_->kind == PhaseKind::Minimizer
                          || plan_->kind == PhaseKind::Sampling;
    const bool force = is_linear_force_slot_(slot);
    if (force)
        decide_neighbor_();
    if (force)
        comm_before_force_(slot, step_idx);
    walk_slot_(slot, step_idx, physical);
    if (force)
        comm_after_force_(slot, step_idx);
    const bool last = static_cast<std::size_t>(slot + 1) == plan_->slots.size();
    if (timing.enabled()) {
        timing.set_loop(std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        static_cast<int>(frame_->nlocal()),
                        static_cast<int>(frame_->nghost()));
        if (last)
            timing.tick_step();
    }
}

// ── Nested plans ─────────────────────────────────────────────────────────────
//
// A sub-driver runs its own Plan on ITS PARENT'S runtime — this one. No second
// runtime, no second Context, no rebuilt domain decomposition, no re-bound
// frame. No timing: `op_handle_` is dense over the BOUND plan's slots, so
// `step_plan` (and nested driver ops dispatched from `run_slot_`'s trial
// replay, below) never index into it. Nested ops are deliberately untimed on
// both replay paths — extending TimerRegistry to sparse nested handles is a
// separate decision (deferred by esmm-contract-03-runslot-nested).
//
// ## Resolution windows on the replay paths
//
// `Runtime::step` is the reference semantics: it opens a window around each
// COMPONENT launch and closes it — `end_resolution_window` deactivates — before
// the next op, so a driver op is dispatched with NO constrained window active.
// A nested node's own driver-level lookups (MeanFieldMethod's force-span
// epilogue, for one; its SCF loop parks its own window already) are therefore
// charged to nobody.
//
// Both replay paths below can be entered from INSIDE a live window: `run_slot_`
// is called from an Optimizer's `launch` via `ctx.opt.reeval_forces`, and
// `step_plan` from a driver nested under such a call. So every op here —
// component and driver alike — saves the caller's window first and restores it
// afterwards. `restore_resolution_window` writes count, budget and active back
// together, so the caller resumes with its own count untouched and never pays
// for the nested work (before this, a nested RHF billed each line-search trial
// to the enclosing optimizer and blew its component budget after a few trials).
//
// A driver op is handed the default-constructed window — inactive, the exact
// state a step-level dispatch sees. It deliberately gets no
// `end_resolution_window`: that pairing exists only to run the NAMED overrun
// check, and a driver op has no component-sized budget to check against.
// `restore` alone is the complete counterpart to a saved snapshot (there is no
// window stack to unwind — `ResolutionState` is three atomics).

void Runtime::build_plan(const Plan<cpu::Component>& plan) {
    BuildContext ctx{context_, *frame_};
    std::vector<cpu::Component*> acquired;
    try {
        for (auto* c : plan.components()) {
            if (multi_rank_dd(context_) && c->requires_full_frame())
                throw std::runtime_error(
                    c->name()
                    + ": requires a complete Frame and cannot be mounted on a "
                      "spatial partition");
            auto [it, inserted] = component_users_.emplace(c, 0);
            if (!inserted && it->second == 0)
                throw std::logic_error("cyclic component lifecycle: "
                                       + c->name());
            if (inserted) {
                auto* node = c->nested_node();
                bool first_node = false;
                try {
                    if (node) {
                        auto [node_it, node_inserted] =
                            node_users_.emplace(node, 0);
                        first_node = node_inserted;
                        if (!first_node && node_it->second == 0)
                            throw std::logic_error(
                                "cyclic nested driver lifecycle");
                    }
                    if (!node || first_node) {
                        if (c->requires_tag_index()
                            && frame_->meta.get(kTagIndex).empty())
                            frame_->rebuild_tag_index(0, frame_->nlocal());
                        c->build(ctx);
                    }
                    if (node)
                        ++node_users_.at(node);
                } catch (...) {
                    if (first_node)
                        node_users_.erase(node);
                    component_users_.erase(c);
                    throw;
                }
                component_order_.push_back(c);
            }
            ++component_users_.at(c);
            acquired.push_back(c);
        }
    } catch (...) {
        for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
            try {
                release_component_(*it);
            } catch (...) {
                ATV_WARN("component destroy failed during build rollback");
            }
        }
        throw;
    }
}

Runtime::ScopedPlan::ScopedPlan(Runtime& rt, const Plan<cpu::Component>& plan,
                                int step_idx)
    : runtime_(rt), saved_plan_(rt.plan_), saved_md_(rt.context_.md),
      saved_contracts_(rt.contracts_ready_),
      saved_setup_(rt.verlet_setup_done_), saved_forces_(rt.forces_ready_),
      changed_(rt.plan_ != &plan),
      physical_(plan.kind == PhaseKind::MD || plan.kind == PhaseKind::Evaluate
                || plan.kind == PhaseKind::Minimizer
                || plan.kind == PhaseKind::Sampling) {
    if (!changed_)
        return;
    if (physical_ && rt.physical_pipeline_active_)
        throw std::runtime_error(
            "a nested physical driver cannot replace geometry/accumulators "
            "during its parent's force/integration pipeline; mount it in an "
            "outer workflow or at MD end_of_step");
    if (physical_)
        saved_verlet_ = std::move(rt.verlet_);
    saved_handles_ = std::move(rt.op_handle_);
    saved_force_bindings_ = std::move(rt.force_bindings_);
    rt.plan_ = &plan;
    rt.contracts_ready_ = true;
    rt.verlet_setup_done_ = false;
    rt.forces_ready_ = false;
    if (physical_) {
        try {
            rt.refresh_plan_(step_idx);
        } catch (...) {
            rt.plan_ = saved_plan_;
            rt.verlet_ = std::move(saved_verlet_);
            rt.op_handle_ = std::move(saved_handles_);
            rt.force_bindings_ = std::move(saved_force_bindings_);
            rt.context_.md = saved_md_;
            rt.context_.md.ghost = &rt.verlet_.ghost;
            rt.contracts_ready_ = false;
            rt.forces_ready_ = false;
            throw;
        }
    }
    // Electronic iterations borrow the existing communication services without
    // copying ghost contracts or rebuilding spatial state.
}

Runtime::ScopedPlan::~ScopedPlan() {
    if (!changed_)
        return;
    auto& rt = runtime_;
    rt.plan_ = saved_plan_;
    if (physical_)
        rt.verlet_ = std::move(saved_verlet_);
    rt.op_handle_ = std::move(saved_handles_);
    rt.force_bindings_ = std::move(saved_force_bindings_);
    rt.context_.md = saved_md_;
    rt.context_.md.ghost = &rt.verlet_.ghost;
    // A physical child may migrate/sort atoms and overwrite force accumulators.
    // Rebuild the parent halo/list and reseed before its next integration.
    rt.contracts_ready_ = saved_contracts_ && !physical_;
    rt.verlet_setup_done_ = saved_setup_ && !physical_;
    rt.forces_ready_ = saved_forces_ && !physical_;
}

void Runtime::step_plan(const Plan<cpu::Component>& plan, int step_idx) {
    DispatchScope dispatch(this);
    ScopedPlan active(*this, plan, step_idx);
    prepare_run(step_idx);
    if (plan.kind == PhaseKind::MD)
        step_verlet_(step_idx);
    else
        step_linear_(step_idx);
}

void Runtime::destroy_plan(const Plan<cpu::Component>& plan) {
    const auto objects = plan.components();
    std::exception_ptr failure;
    for (auto it = objects.rbegin(); it != objects.rend(); ++it) {
        try {
            release_component_(*it);
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}

void Runtime::release_component_(cpu::Component* component) {
    auto it = component_users_.find(component);
    if (it == component_users_.end())
        return;
    if (--it->second != 0)
        return;
    component_users_.erase(it);
    component_order_.erase(std::remove(component_order_.begin(),
                                       component_order_.end(), component),
                           component_order_.end());
    if (auto* node = component->nested_node()) {
        const auto node_it = node_users_.find(node);
        if (node_it == node_users_.end())
            return;
        if (--node_it->second != 0)
            return;
        node_users_.erase(node_it);
    }
    component->destroy(DestroyContext{context_, *frame_});
}

void Runtime::run_slot_(int slot_index, int step_idx) {
    DispatchScope dispatch(this);
    const int saved_slot = active_slot_;
    struct RestoreSlot {
        int& slot;
        int saved;
        bool& pipeline;
        bool saved_pipeline;
        ~RestoreSlot() {
            slot = saved;
            pipeline = saved_pipeline;
        }
    } restore_slot{active_slot_, saved_slot, physical_pipeline_active_,
                   physical_pipeline_active_};
    active_slot_ = slot_index;
    physical_pipeline_active_ |=
        plan_->kind == PhaseKind::Evaluate
        || plan_->kind == PhaseKind::Minimizer
        || (plan_->kind == PhaseKind::MD
            && active_slot_ != slot_of(ATV::md::Phase::end_of_step));
    // The engine's own work below (skin test, halo refresh/rebuild, neighbor
    // build) resolves Frame columns too. It is not the calling optimizer's
    // lookups, so it runs outside that component's resolution window; the
    // caller's window, count included, comes back on exit. Each replayed op
    // still gets its own window.
    const auto caller_window = frame_->resolution_window();
    struct RestoreCallerWindow {
        Frame& frame;
        ResolutionWindow saved;
        ~RestoreCallerWindow() { frame.restore_resolution_window(saved); }
    } restore_caller_window{*frame_, caller_window};
    frame_->restore_resolution_window(ResolutionWindow{});
    // Replay every binding of one phase on the current frame — used for
    // line-search trial evaluations. Re-entrant: must not touch the timing
    // registry (op_handle_ indices belong to the outer step) and must not
    // recurse. Frequency gating is bypassed: a trial is not a simulation step.
    // Nested driver ops dispatch exactly the way step_plan does, resolution
    // window and force-request veto included: this replay runs INSIDE the
    // calling component's window (an Optimizer's, via ctx.opt.reeval_forces),
    // so the node is handed the step-level state and the caller's window is put
    // back afterwards — see the section note above step_plan. A line-search
    // trial inherits the enclosing phase's force policy, which for Minimize's
    // `force` phase is "yes": the search direction IS the gradient.
    // Nested ops are deliberately untimed on both replay paths (here and
    // step_plan): op_handle_ indices belong to the bound outer plan — extending
    // TimerRegistry to sparse nested handles is a separate decision (deferred
    // by esmm-contract-03-runslot-nested).
    // Evaluate/Minimize force replay wraps `ctx.md.comm` the same way
    // `step_linear_` does (nflag XOR + reverse), otherwise a line-search
    // trial would see stale ghosts after the Plan lost CommBrick/Recv.
    const bool wrap = is_linear_force_slot_(slot_index);
    if (wrap) {
        decide_neighbor_();
        comm_before_force_(slot_index, step_idx);
    }
    for (auto& slot : plan_->slots) {
        if (slot.phase != slot_index)
            continue;
        for (auto& op : slot.ops) {
            if (wrap && op.component && op.component->verlet_engine())
                continue;
            LaunchContext ctx{context_,
                              *frame_,
                              slot.phase,
                              step_idx,
                              /*forces_requested=*/slot.forces_requested
                                  && context_.exec.forces_requested,
                              slot.stress_requested
                                  && context_.exec.stress_requested};
            const auto saved = frame_->resolution_window();
            struct RestoreWindow {
                Frame& frame;
                ResolutionWindow saved;
                ~RestoreWindow() { frame.restore_resolution_window(saved); }
            } restore{*frame_, saved};
            frame_->begin_resolution_window(kResolveBudgetPerLaunch);
            op.component->launch(ctx);
            frame_->end_resolution_window(op.component->name().c_str());
        }
    }
    if (wrap)
        comm_after_force_(slot_index, step_idx);
}

void Runtime::run(int n_steps) {
    ATV_INFO("running {} steps ({} atoms)", n_steps, frame_->nlocal());
    for (int s = 0; s < n_steps; ++s)
        step(s);
    ATV_INFO("run complete");
}

std::vector<std::pair<std::string, std::size_t>>
Runtime::collect_component_memory() const {
    std::vector<std::pair<std::string, std::size_t>> out;
    if (plan_) {
        for (const auto* c : plan_->components()) {
            const std::size_t b = c->memory_bytes();
            if (b)
                out.emplace_back(c->name(), b);
        }
    }
    // Per-atom SoA + topology + tag map (the dominant "Atoms" bucket).
    if (frame_)
        out.emplace_back("[atoms]", frame_->block("atoms").memory_bytes());
    // Domain-decomposition tag→local map + ghost-tag list (Comm bucket).
    if (const std::size_t b = domain_.memory_bytes())
        out.emplace_back("[domain]", b);
    return out;
}

void Runtime::destroy_components() {
    if (!components_built_)
        return;
    // Drain and join analysis workers first so any in-flight `on_snapshot`
    // completes against live subscribers before any component enters destroy().
    analysis_.shutdown_workers();

    components_built_ = false;
    destroy_plan(*plan_);
}

void Runtime::unwind_failed_run() noexcept {
    failed_ = true;
    // Join readers before destroy callbacks release their captured state.
    try {
        analysis_.shutdown_workers();
    } catch (...) {
    }
    components_built_ = false;
    while (!component_order_.empty()) {
        auto* component = component_order_.back();
        // A node's destroy_on releases its children through this same registry.
        // Force the final release regardless of how many bindings shared it.
        component_users_[component] = 1;
        try {
            release_component_(component);
        } catch (...) {
            ATV_WARN("component destroy failed while unwinding a run");
        }
    }
    component_users_.clear();
    node_users_.clear();
    contracts_ready_ = false;
    forces_ready_ = false;
    verlet_setup_done_ = false;
    try {
        shutdown();
    } catch (...) {
    }
}

void Runtime::shutdown() {
    ATV_DEBUG("CPU runtime shutting down");
    analysis_.teardown(context_);
    context_.exec.runtime =
        nullptr; // the Runtime is gone; no node may reach it
    domain_.release_comm(context_);
    frame_ = nullptr;
    plan_ = nullptr;
}

} // namespace cpu
} // namespace ATV
