/// @file cuda/driver/driver.cu
/// @brief CUDA Backend implementation + Driver explicit instantiations.

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/precision.h"
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

#include "atomiverse/core/driver_template_impl.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/log_callback.h"
#include "atomiverse/cuda/device/timer.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

// ── Backend ────────────────────────────────────────────────────────

void Backend::init() {
    cuda::register_log_callback();
    cudaStream_t main, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&main));
    context_.stream = main;
    // Retain each acquired stream immediately so failure cleanup can release
    // the first stream if creation of the second one fails.
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    context_.defer_stream = defer;
    context_.exec.backend = this;
    context_.exec.evaluation_transaction = &evaluation_;
}

namespace {
// Exactly the input columns consumed by cpu::to_cuda; arbitrary host-only
// custom fields do not change the single-GPU execution identity.
bool device_input(std::string_view block, const cpu::ColumnRef& column) {
    if (block != "atoms")
        return false;
    switch (static_cast<AtomFieldId>(column.wire_id)) {
    case AtomFieldId::x:
    case AtomFieldId::y:
    case AtomFieldId::z:
    case AtomFieldId::vx:
    case AtomFieldId::vy:
    case AtomFieldId::vz:
    case AtomFieldId::mass:
    case AtomFieldId::charge:
    case AtomFieldId::type:
    case AtomFieldId::tag:
    case AtomFieldId::image:
    case AtomFieldId::num_bond:
    case AtomFieldId::bond_atom:
    case AtomFieldId::bond_type:
    case AtomFieldId::num_angle:
    case AtomFieldId::angle_atom:
    case AtomFieldId::angle_type:
    case AtomFieldId::num_dihedral:
    case AtomFieldId::dihedral_atom:
    case AtomFieldId::dihedral_type:
        return true;
    default:
        return false;
    }
}

__global__ void publish_cell(const geom_t* cell, Box seed, Box* cache) {
    const Box previous = *cache;
    seed.resident = nullptr;
    seed.assign_cell(cell);
    bool changed = false;
    for (int k = 0; k < 9; ++k)
        changed |= !(seed.cell[k] == previous.cell[k]);
    seed.revision = previous.revision + (changed ? 1 : 0);
    *cache = seed;
}
} // namespace

void Backend::publish_box_() {
    publish_cell<<<1, 1, 0, static_cast<cudaStream_t>(context_.stream)>>>(
        frame_.box, context_.host_box,
        const_cast<Box*>(context_.host_box.resident));
    ATV_CUDA_CHECK(cudaGetLastError());
}

void Backend::materialize(const cpu::Frame& frame) {
    forces_ready_ = false;
    frame_ = cpu::to_cuda(frame);
    context_.md.n_dof_global = ATV::md::degrees_of_freedom(frame.nlocal());

    auto& box = context_.host_box;
    geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = frame.box.cell()(i, j);
    box.assign_cell(cell);
    Box* resident = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&resident, sizeof(Box)));
    // Zeroed cache: the first publish is a cell change (revision 1).
    ATV_CUDA_CHECK(cudaMemset(resident, 0, sizeof(Box)));
    box.resident = resident;
    publish_box_();
    input_snapshot_.capture(frame, device_input);
    inputs_dirty_ = identity_dirty_ = false;
}

void Backend::load(const Plan<Component>& plan) {
    std::vector<std::pair<int, Component*>> bindings;
    if (plan.kind == PhaseKind::MD || plan.kind == PhaseKind::Minimizer)
        for (const auto& slot : plan.slots)
            for (const auto& op : slot.ops)
                bindings.emplace_back(slot.phase, op.component);
    if (bindings != force_bindings_)
        forces_ready_ = false;
    force_bindings_ = std::move(bindings);
    plan_ = &plan;
}

void Backend::build_components() {
    build_plan(*plan_);
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream)));
    if (timer_)
        timer_->impl()->flush();
}

void Backend::build_plan(const Plan<Component>& plan) {
    evaluation_.reserve(plan.components().size());
    BuildContext build{context_, frame_};
    DestroyContext destroy{context_, frame_};
    lifecycle_.build(
        plan, [&](Component& c) { c.build(build); },
        [&](Component& c) { c.destroy(destroy); });
}

void Backend::destroy_plan(const Plan<Component>& plan) {
    DestroyContext ctx{context_, frame_};
    lifecycle_.destroy(plan, [&](Component& c) { c.destroy(ctx); });
}

void Backend::prepare_run(BigInt step_idx) {
    if (!plan_
        || (plan_->kind != PhaseKind::MD && plan_->kind != PhaseKind::Minimizer)
        || forces_ready_)
        return;
    struct DispatchGuard {
        bool& active;
        bool saved;
        ~DispatchGuard() { active = saved; }
    } guard{dispatching_, dispatching_};
    dispatching_ = true;
    plan_->prepare_forces(
        [](int) {},
        [&](const auto& slot, const auto& op, bool force_slot) {
            LaunchContext ctx{context_, frame_, slot.phase, step_idx};
            ctx.initializing_forces = true;
            ctx.forces_requested &= slot.forces_requested;
            ctx.stress_requested &= slot.stress_requested;
            if (force_slot)
                op.component->launch(ctx);
            else
                op.component->initialize_forces(ctx);
        });
    forces_ready_ = true;
}

void Backend::execute_phases(BigInt step_idx, int first_phase, int end_phase) {
    if (!plan_)
        throw std::invalid_argument("CUDA Backend has no loaded plan");
    execute_phases(*plan_, step_idx, first_phase, end_phase);
}

void Backend::execute_phases(const Plan<Component>& plan, BigInt step_idx,
                             int first_phase, int end_phase) {
    if (first_phase < 0 || end_phase < first_phase
        || end_phase > static_cast<int>(plan.slots.size()))
        throw std::invalid_argument(
            "CUDA Backend phase range is outside the loaded plan");
    const bool timed = timer_ && !dispatching_;
    struct DispatchGuard {
        bool& active;
        bool saved;
        ~DispatchGuard() { active = saved; }
    } guard{dispatching_, dispatching_};
    dispatching_ = true;
    for (int phase = first_phase; phase < end_phase; ++phase) {
        auto& slot = plan.slots[static_cast<std::size_t>(phase)];
        if (slot.ops.empty())
            continue;
        for (auto& op : slot.ops) {
            if (op.frequency > 0 && step_idx % op.frequency != 0)
                continue;
            auto stream = static_cast<cudaStream_t>(
                (op.scheduling == Scheduling::deferrable)
                    ? context_.defer_stream
                    : context_.stream);
            LaunchContext ctx{context_, frame_, slot.phase, step_idx};
            ctx.forces_requested &= slot.forces_requested;
            ctx.stress_requested &= slot.stress_requested;
            if (timed)
                timer_->impl()->begin(stream);
            try {
                op.component->launch(ctx);
            } catch (...) {
                if (timed) {
                    try {
                        timer_->impl()->end(stream, slot.phase,
                                            op.component->name());
                    } catch (...) {
                        // Profiling failure must not replace the component error.
                    }
                    try {
                        timer_->impl()->flush();
                    } catch (...) {
                        // Consume the pending batch even if recording end failed.
                    }
                }
                throw;
            }
            if (op.component->changes_box(slot.phase))
                publish_box_();
            if (timed)
                timer_->impl()->end(stream, slot.phase, op.component->name());
        }
        // Per-phase syncs intentionally removed
        // (cuda-md-review-cleanup-03-launch-purity ac-005):
        //   - Components within a phase already serialize via CUDA's
        //     in-stream FIFO ordering. The previous unconditional
        //     `cudaStreamSynchronize(stream) + cudaStreamSynchronize(defer)`
        //     drained both streams 6× per step (per-phase × main + defer),
        //     ~80 µs/step pure host wait on Hopper for no semantic gain.
        //   - Cross-stream dependencies (defer_stream reads main_stream's
        //     buffers) are explicitly ordered by AsyncBridge::launch via
        //     cudaEventRecord(main) + cudaStreamWaitEvent(defer) per
        //     deferred op, so cross-stream reads see consistent values
        //     without per-phase sync.
        //   - End-of-run sync happens in Backend::sync() / retrieve();
        //     individual components issue event syncs when they need
        //     specific host-visible values (NPT, NeighborList).
        if (timed)
            timer_->impl()->flush();
    }
}

void Backend::step(BigInt step_idx) {
    EvaluationTransaction::Scope evaluation(evaluation_);
    prepare_run(step_idx);
    execute_phases(step_idx, 0, static_cast<int>(plan_->slots.size()));
    if (!evaluation_.empty()) {
        sync();
        throw_if_force_fault(frame_);
    }
    if (timer_)
        timer_->step_done();
    evaluation.accept();
}

void Backend::run(int n_steps) {
    for (int s = 0; s < n_steps; ++s)
        step(s);
}

void Backend::destroy_components() {
    DestroyContext ctx{context_, frame_};
    lifecycle_.destroy_all([&](Component& c) { c.destroy(ctx); });
}

void Backend::retrieve(cpu::Frame& frame, bool copy_forces) {
    sync();
    cuda::to_cpu(frame_, frame, copy_forces);
    input_snapshot_.capture(frame, device_input);
    inputs_dirty_ = identity_dirty_ = false;
}

InputChange Backend::validate_inputs(const cpu::Frame& frame) const {
    auto changes = input_snapshot_.changes(frame, device_input);
    if (identity_dirty_ || frame.nlocal() != frame_.n_atoms())
        changes |= InputChange::Identity;
    if (changes == InputChange::None && !inputs_dirty_)
        return changes;
    // Preflight the complete plan before touching device state or any cache.
    // A retry restores numeric inputs even if the caller returned to the last
    // checkpoint; that restoration is not a newly edited charge/identity.
    const auto invalidation =
        inputs_dirty_ ? changes | InputChange::Geometry | InputChange::Cell
                      : changes;
    for (auto* component : plan_->components())
        if (!component->supports_input_changes(invalidation))
            throw std::invalid_argument(
                "CUDA component '" + component->name()
                + "' does not support these input changes; its input-derived "
                  "caches require a new run object");
    return changes;
}

void Backend::refresh_inputs(const cpu::Frame& frame) {
    const auto changes = validate_inputs(frame);
    if (changes == InputChange::None && !inputs_dirty_)
        return;
    const auto invalidation =
        inputs_dirty_ ? changes | InputChange::Geometry | InputChange::Cell
                      : changes;
    sync();
    const bool restore_all = inputs_dirty_;
    mark_inputs_dirty();
    if (has_input_change(changes, InputChange::Identity)) {
        // Build replacement storage before releasing the current allocation.
        // The Frame object stays at the same address for borrowed method plans.
        auto replacement = cpu::to_cuda(frame);
        cuda::destroy(frame_);
        frame_ = replacement;
        identity_dirty_ = true;
    } else {
        auto upload = changes;
        if (restore_all)
            upload |= InputChange::Geometry | InputChange::Velocity
                      | InputChange::Charge | InputChange::Cell;
        cuda::update_inputs(frame, frame_, upload);
    }
    auto& box = context_.host_box;
    geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = frame.box.cell()(i, j);
    box.assign_cell(cell);
    publish_box_();
    context_.md.n_dof_global = ATV::md::degrees_of_freedom(frame.nlocal());
    // A reported scientific force fault belongs to the discarded evaluation.
    // Reset it only at an explicit input/retry boundary, never during a step.
    const ForceFault clean_fault{};
    ATV_CUDA_CHECK(cudaMemcpy(frame_.force_fault, &clean_fault,
                              sizeof(clean_fault), cudaMemcpyHostToDevice));
    for (auto* component : plan_->components())
        component->invalidate_inputs(invalidation, {context_, frame_});
}

void Backend::sync() {
    ATV_CUDA_CHECK(
        cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream)));
    ATV_CUDA_CHECK(cudaStreamSynchronize(
        static_cast<cudaStream_t>(context_.defer_stream)));
}

void Backend::shutdown() {
    std::exception_ptr failure;
    auto cleanup = [&](auto&& operation) {
        try {
            operation();
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    };
    if (!lifecycle_.empty()) {
        // Drain both streams even when one reports an asynchronous error.
        cleanup([&] {
            ATV_CUDA_CHECK(cudaStreamSynchronize(
                static_cast<cudaStream_t>(context_.stream)));
        });
        cleanup([&] {
            ATV_CUDA_CHECK(cudaStreamSynchronize(
                static_cast<cudaStream_t>(context_.defer_stream)));
        });
        DestroyContext ctx{context_, frame_};
        lifecycle_.destroy_all([&](Component& component) {
            cleanup([&] { component.destroy(ctx); });
        });
    }
    if (context_.host_box.resident)
        cleanup([&] {
            ATV_CUDA_CHECK(
                cudaFree(const_cast<Box*>(context_.host_box.resident)));
        });
    cuda::destroy(frame_);
    frame_ = {};
    auto main = static_cast<cudaStream_t>(context_.stream);
    auto defer = static_cast<cudaStream_t>(context_.defer_stream);
    if (main)
        cleanup([&] { ATV_CUDA_CHECK(cudaStreamDestroy(main)); });
    if (defer)
        cleanup([&] { ATV_CUDA_CHECK(cudaStreamDestroy(defer)); });
    context_ = {};
    input_snapshot_.clear();
    inputs_dirty_ = identity_dirty_ = false;
    forces_ready_ = false;
    plan_ = nullptr;
    cuda::unregister_log_callback();
    // Cleanup attempts every release; a poisoned CUDA context can still require
    // process restart. Reporting an error does not leave a live host-side ledger.
    if (failure)
        std::rethrow_exception(failure);
}

} // namespace cuda

// ── Explicit instantiations (CUDA) ───────────────────────────────

template class Driver<md::Phase, cuda::device_tag>;
template class Driver<minimizer::Phase, cuda::device_tag>;
template class Driver<evaluate::Phase, cuda::device_tag>;
template class Driver<scf::Phase, cuda::device_tag>;
#ifdef ATV_HAS_SEMIEMPIRICAL
template class Driver<nddo::Phase, cuda::device_tag>;
#endif

} // namespace ATV
