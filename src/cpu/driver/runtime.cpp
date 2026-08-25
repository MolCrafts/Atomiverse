/// @file cpu/driver/runtime.cpp
/// @brief cpu::Runtime implementation. Plan dispatch + component lifecycle;
///        DD / device / analysis state lives in the composed sub-runtimes.

#include "atomiverse/cpu/driver/runtime.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/molrs_contract.h"
#include "atomiverse/cpu/communication/domain.h"
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
}

void Runtime::step(int step_idx) {
    TimerRegistry& timing_reg = device_.timing();
    const bool timing = timing_reg.enabled();
    const bool have_handles = op_handle_.size() == plan_->slots.size();
    const auto step_t0 = timing ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    for (std::size_t s = 0; s < plan_->slots.size(); ++s) {
        auto& slot = plan_->slots[s];
        if (slot.ops.empty())
            continue;
        for (std::size_t o = 0; o < slot.ops.size(); ++o) {
            auto& op = slot.ops[o];
            if (op.frequency > 0 && step_idx % op.frequency != 0)
                continue;
            LaunchContext ctx{context_, *frame_, slot.phase, step_idx,
                              /*forces_requested=*/slot.forces_requested
                                  && context_.exec.forces_requested};
            // Resolution-discipline window (frame-block-01): a component that
            // looks mapping keys up inside a loop blows the budget and throws.
            frame_->begin_resolution_window(kResolveBudgetPerLaunch);
            if (timing && have_handles) {
                const auto t0 = std::chrono::steady_clock::now();
                op.component->launch(ctx);
                const auto t1 = std::chrono::steady_clock::now();
                timing_reg.record(
                    op_handle_[s][o],
                    std::chrono::duration<double>(t1 - t0).count());
            } else {
                op.component->launch(ctx); // zero-overhead path when off
            }
            frame_->end_resolution_window(op.component->name().c_str());
        }
    }
    if (timing) {
        const auto step_t1 = std::chrono::steady_clock::now();
        auto* dd = domain_.dd();
        timing_reg.set_loop(
            std::chrono::duration<double>(step_t1 - step_t0).count(),
            dd ? dd->n_local : static_cast<int>(frame_->nlocal()));
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
    for (auto& slot : plan_->slots) {
        if (slot.phase != slot_index)
            continue;
        for (auto& op : slot.ops) {
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
