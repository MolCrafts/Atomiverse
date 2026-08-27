/// @file cpu/driver/runtime.cpp
/// @brief cpu::Runtime implementation. Plan dispatch + component lifecycle;
///        DD / device / analysis state lives in the composed sub-runtimes.

#include "atomiverse/cpu/driver/runtime.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/molrs_contract.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/driver/step_timing.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
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

/// LAMMPS Verlet: `nflag = neighbor->decide()` before exchange/borders.
void decide_reneighbor_for_plan(const Plan<cpu::Component>* plan,
                                cpu::Context& context, cpu::Frame& frame) {
    context.md.reneighbor = true;
    context.md.reneighbor_decided = false;
    if (plan == nullptr)
        return;
    for (const auto& slot : plan->slots) {
        for (const auto& op : slot.ops) {
            if (op.component == nullptr)
                continue;
            cpu::NeighborProvider* np = op.component->neighbor_provider();
            if (np == nullptr)
                continue;
            context.md.reneighbor = np->decide_reneighbor(context, frame);
            sync_reneighbor_flag(context);
            context.md.reneighbor_decided = true;
            return;
        }
    }
}

} // namespace

Runtime::Runtime() = default;
Runtime::~Runtime() { shutdown(); } // idempotent; releases cart_comm + executor
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

void Runtime::init() {
    // A local ATV_MOLRS_PATH is an alternate source location, not permission to
    // run against a different interface. Fail before any execution service is
    // initialized or any component sees a Frame.
#ifdef ATV_HAS_MOLRS
    ATV::molrs_contract::require_compatible();
#endif

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
    frame_ = &frame;
    bool needs_tag_index = false;
    if (plan_) {
        for (const auto& slot : plan_->slots) {
            for (const auto& op : slot.ops) {
                if (op.component)
                    needs_tag_index =
                        needs_tag_index || op.component->requires_tag_index();
                if (needs_tag_index)
                    break;
            }
            if (needs_tag_index)
                break;
        }
    }
    domain_.bind_frame(frame, context_, needs_tag_index);
}

void Runtime::adopt_dd(std::unique_ptr<md::mpi::DomainDecomposition> dd) {
    domain_.adopt(std::move(dd), context_);
}

std::unique_ptr<md::mpi::DomainDecomposition> Runtime::release_dd() {
    return domain_.release(context_);
}

void Runtime::build_components() {
    BuildContext ctx{context_, *frame_};
    TimerRegistry& timing = device_.timing();
    // Register one timing handle per op (component name + phase + category
    // rollup), precomputed so the hot step loop never re-classifies / allocates.
    op_handle_.assign(plan_->slots.size(), {});
    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        auto& slot = plan_->slots[s];
        op_handle_[s].reserve(slot.ops.size());
        for (auto& op : slot.ops) {
            const std::string nm = op.component->name();
            ATV_DEBUG("building component '{}'", nm);
            op.component->build(ctx);
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
    // The schema is now folded; hold the Frame to it before anything runs.
    // Iron law 4, applied once for the whole component set instead of each
    // component remembering to check its own fields.
    if (frame_ != nullptr) {
        check_schema(verlet_.ghost_fields, *frame_, "driver build");
        check_schema(verlet_.ghost_reverse_fields, *frame_, "driver build");
        check_schema(verlet_.ghost_refresh_fields, *frame_, "driver build");
    }
    verlet_setup_done_ = false;
    if (plan_
        && (plan_->kind == PhaseKind::MD || plan_->kind == PhaseKind::Evaluate
            || plan_->kind == PhaseKind::Minimizer))
        setup_verlet_comm_();
}

void Runtime::launch_op_(std::size_t s, std::size_t o, int step_idx) {
    auto& slot = plan_->slots[s];
    auto& op = slot.ops[o];
    if (op.frequency > 0 && step_idx % op.frequency != 0)
        return;
    LaunchContext ctx{context_, *frame_, slot.phase, step_idx,
                      /*forces_requested=*/slot.forces_requested
                          && context_.exec.forces_requested};
    frame_->begin_resolution_window(kResolveBudgetPerLaunch);
    TimerRegistry& timing_reg = device_.timing();
    const bool timing = timing_reg.enabled();
    const bool have_handles = op_handle_.size() == plan_->slots.size();
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
    }
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
    if (plan_ == nullptr)
        return;
    // The halo's own product comes first, so it keeps the leading wire
    // positions no matter which components are mounted.
    base_fields::border(verlet_.ghost_fields);
    base_fields::reverse(verlet_.ghost_reverse_fields);
    base_fields::forward(verlet_.ghost_refresh_fields);
    context_.md.ghost_fields = &verlet_.ghost_fields;
    context_.md.ghost_reverse_fields = &verlet_.ghost_reverse_fields;
    context_.md.ghost_refresh_fields = &verlet_.ghost_refresh_fields;
    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        const auto& slot = plan_->slots[s];
        for (std::size_t o = 0; o < slot.ops.size(); ++o) {
            cpu::Component* c = slot.ops[o].component;
            if (c == nullptr)
                continue;
            if (cpu::NeighborProvider* np = c->neighbor_provider()) {
                verlet_.neighbor = np;
                verlet_.neighbor_comp = c;
                verlet_.neighbor_slot = static_cast<int>(s);
                verlet_.neighbor_op = static_cast<int>(o);
            }
            if (const cpu::NeighborConsumer* nc = c->as_neighbor_consumer()) {
                const double halo = static_cast<double>(nc->halo_cutoff());
                if (halo > verlet_.consumer_halo_cutoff)
                    verlet_.consumer_halo_cutoff = halo;
            }
            // Same fold, one level up: instead of the widest cutoff, the
            // union of the columns anybody reads on a ghost row. Insertion
            // order is Plan order, so every rank builds the identical
            // schema without a collective.
            c->declare_fields(verlet_.ghost_fields);
            c->declare_reverse_fields(verlet_.ghost_reverse_fields);
            c->declare_refresh_fields(verlet_.ghost_refresh_fields);
        }
    }
}

void Runtime::apply_ghost_cutoff_() {
    cpu::Comm* comm = context_.md.comm;
    if (!comm || !verlet_.neighbor)
        return;
    // The provider covers pair enumeration (cutoff + skin); a many-body
    // consumer may declare a wider ghost-of-ghost shell (EAM: 2·cutoff).
    // 1.1× on the consumer shell keeps the declared reach valid while
    // atoms drift between reneighbourings (80c5f4a6 contract).
    const double reach =
        std::max(static_cast<double>(verlet_.neighbor->neighbor_reach()),
                 1.1 * verlet_.consumer_halo_cutoff);
    if (reach > 0.0)
        comm->set_ghost_cutoff(reach);
}

bool Runtime::is_linear_force_slot_(int slot) const {
    if (plan_ == nullptr)
        return false;
    if (plan_->kind == PhaseKind::Evaluate)
        return slot == ATV::slot_of(ATV::evaluate::Phase::evaluate);
    if (plan_->kind == PhaseKind::Minimizer)
        return slot == ATV::slot_of(ATV::minimizer::Phase::force);
    return false;
}

namespace {

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

void Runtime::comm_rebuild_(int exchange_slot, int borders_slot, int step) {
    apply_ghost_cutoff_();
    cpu::Comm* comm = context_.md.comm;
    if (!comm)
        return;
    TimerRegistry& reg = device_.timing();
    LaunchContext comm_ctx{context_, *frame_, exchange_slot, step,
                           /*forces_requested=*/true};
    timed_comm(reg, comm_handle_exch_, [&] { comm->exchange(comm_ctx); });
    comm_ctx.stage = borders_slot;
    timed_comm(reg, comm_handle_bord_, [&] { comm->borders(comm_ctx); });
}

void Runtime::comm_before_force_(int slot, int step_idx) {
    cpu::Comm* comm = context_.md.comm;
    if (!comm)
        return;
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

void Runtime::setup_verlet_comm_() {
    // LAMMPS Verlet::setup (src/verlet.cpp): exchange + borders +
    // neighbor->build before the run loop so the first nflag==0 step can
    // forward_comm. Copyright (2003) Sandia Corporation. GPL-2.0-only.
    if (verlet_setup_done_ || context_.mpi_size <= 1)
        return;
    LaunchContext ctx{context_, *frame_,
                      ATV::slot_of(ATV::md::Phase::pre_exchange), 0,
                      /*forces_requested=*/true};
    comm_rebuild_(ATV::slot_of(ATV::md::Phase::pre_exchange),
                  ATV::slot_of(ATV::md::Phase::pre_neighbor), 0);
    if (verlet_.neighbor_comp) {
        context_.md.reneighbor = true;
        context_.md.reneighbor_decided = true;
        ctx.stage = ATV::slot_of(ATV::md::Phase::neighbor);
        verlet_.neighbor_comp->launch(ctx);
        context_.md.reneighbor_decided = false;
    }
    verlet_setup_done_ = true;
}

void Runtime::step_linear_(int step_idx) {
    decide_reneighbor_for_plan(plan_, context_, *frame_);
    const bool wrap = plan_
                      && (plan_->kind == PhaseKind::Evaluate
                          || plan_->kind == PhaseKind::Minimizer);
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
                               : ATV::slot_of(ATV::minimizer::Phase::force);

    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        if (static_cast<int>(s) == force_slot)
            comm_before_force_(force_slot, step_idx);
        walk_slot_(static_cast<int>(s), step_idx, /*skip_engine=*/true);
        if (static_cast<int>(s) == force_slot)
            comm_after_force_(force_slot, step_idx);
    }
    context_.md.reneighbor_decided = false;
}

void Runtime::step_verlet_(int step_idx) {
    // Structural copy of LAMMPS Verlet::run (src/verlet.cpp) communication
    // block on ctx.md.comm: nflag = neighbor->decide(); if 0: forward_comm;
    // else exchange + borders + neighbor->build; reverse_comm if newton.
    // Copyright (2003) Sandia Corporation. GPL-2.0-only.
    using P = ATV::md::Phase;
    walk_slot_(ATV::slot_of(P::initial_integrate), step_idx, true);
    walk_slot_(ATV::slot_of(P::post_integrate), step_idx, true);

    bool nflag = false;
    if (verlet_.neighbor) {
        nflag = verlet_.neighbor->decide_reneighbor(context_, *frame_);
    }
    context_.md.reneighbor = nflag;
    sync_reneighbor_flag(context_);
    nflag = context_.md.reneighbor;
    context_.md.reneighbor_decided = true;

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

    context_.md.reneighbor_decided = false;
}

void Runtime::step(int step_idx) {
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
        auto* dd = domain_.dd();
        timing_reg.set_loop(
            std::chrono::duration<double>(step_t1 - step_t0).count(),
            dd ? dd->n_local : static_cast<int>(frame_->nlocal()),
            dd ? dd->n_ghost : static_cast<int>(frame_->nghost()));
        timing_reg.tick_step();
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
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            ATV_DEBUG("building nested component '{}'", op.component->name());
            op.component->build(ctx);
        }
    }
}

void Runtime::step_plan(const Plan<cpu::Component>& plan, int step_idx) {
    decide_reneighbor_for_plan(&plan, context_, *frame_);
    for (const auto& slot : plan.slots) {
        if (slot.ops.empty())
            continue;
        for (const auto& op : slot.ops) {
            if (op.frequency > 0 && step_idx % op.frequency != 0)
                continue;
            LaunchContext ctx{context_, *frame_, slot.phase, step_idx,
                              /*forces_requested=*/slot.forces_requested
                                  && context_.exec.forces_requested};
            const auto saved = frame_->resolution_window();
            frame_->begin_resolution_window(kResolveBudgetPerLaunch);
            op.component->launch(ctx);
            frame_->end_resolution_window(op.component->name().c_str());
            frame_->restore_resolution_window(saved);
        }
    }
}

void Runtime::destroy_plan(const Plan<cpu::Component>& plan) {
    DestroyContext ctx{context_, *frame_};
    for (int s = static_cast<int>(plan.slots.size()) - 1; s >= 0; --s) {
        const auto& slot = plan.slots[s];
        for (int o = static_cast<int>(slot.ops.size()) - 1; o >= 0; --o) {
            const auto& op = slot.ops[o];
            ATV_DEBUG("destroying nested component '{}'", op.component->name());
            op.component->destroy(ctx);
        }
    }
}

void Runtime::run_slot_(int slot_index, int step_idx) {
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
    if (wrap)
        comm_before_force_(slot_index, step_idx);
    for (auto& slot : plan_->slots) {
        if (slot.phase != slot_index)
            continue;
        for (auto& op : slot.ops) {
            if (wrap && op.component && op.component->verlet_engine())
                continue;
            LaunchContext ctx{context_, *frame_, slot.phase, step_idx,
                              /*forces_requested=*/slot.forces_requested
                                  && context_.exec.forces_requested};
            const auto saved = frame_->resolution_window();
            frame_->begin_resolution_window(kResolveBudgetPerLaunch);
            op.component->launch(ctx);
            frame_->end_resolution_window(op.component->name().c_str());
            frame_->restore_resolution_window(saved);
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
    // Unique components across all plan slots (a component may bind to several
    // phases) — dedup by pointer so its memory is counted once.
    if (plan_) {
        std::vector<const cpu::Component*> seen;
        for (const auto& slot : plan_->slots) {
            for (const auto& op : slot.ops) {
                const cpu::Component* c = op.component;
                if (!c || std::find(seen.begin(), seen.end(), c) != seen.end())
                    continue;
                seen.push_back(c);
                const std::size_t b = c->memory_bytes();
                if (b)
                    out.emplace_back(c->name(), b);
            }
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
    // Drain and join analysis workers first so any in-flight `on_snapshot`
    // completes against live subscribers before any component enters destroy().
    analysis_.shutdown_workers();

    DestroyContext ctx{context_, *frame_};
    for (int s = static_cast<int>(plan_->slots.size()) - 1; s >= 0; --s) {
        auto& slot = plan_->slots[s];
        for (int o = static_cast<int>(slot.ops.size()) - 1; o >= 0; --o) {
            auto& op = slot.ops[o];
            ATV_DEBUG("destroying component '{}'", op.component->name());
            op.component->destroy(ctx);
        }
    }
}

void Runtime::shutdown() {
    ATV_DEBUG("CPU runtime shutting down");
    analysis_.teardown(context_);
    context_.exec.runtime =
        nullptr; // the Runtime is gone; no node may reach it
    domain_.release_comm();
    frame_ = nullptr;
    plan_ = nullptr;
}

} // namespace cpu
} // namespace ATV
