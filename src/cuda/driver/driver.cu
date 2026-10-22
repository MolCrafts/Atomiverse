/// @file cuda/driver/driver.cu
/// @brief CUDA Backend implementation + Driver explicit instantiations.

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/precision.h"
#include <cmath>
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
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    context_.stream = main;
    context_.defer_stream = defer;
}

namespace {
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
    plan_->prepare_forces(
        [](int) {},
        [&](const auto& slot, const auto& op, bool force_slot) {
            LaunchContext ctx{context_, frame_, slot.phase,
                              step_idx, true,   slot.forces_requested};
            if (force_slot)
                op.component->launch(ctx);
            else
                op.component->initialize_forces(ctx);
        });
    forces_ready_ = true;
}

void Backend::execute_phases(BigInt step_idx, int first_phase, int end_phase) {
    if (!plan_ || first_phase < 0 || end_phase < first_phase
        || end_phase > static_cast<int>(plan_->slots.size()))
        throw std::invalid_argument(
            "CUDA Backend phase range is outside the loaded plan");
    for (int phase = first_phase; phase < end_phase; ++phase) {
        auto& slot = plan_->slots[static_cast<std::size_t>(phase)];
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
            if (timer_)
                timer_->impl()->begin(stream);
            op.component->launch(ctx);
            if (op.component->changes_box(slot.phase))
                publish_box_();
            if (timer_)
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
        if (timer_)
            timer_->impl()->flush();
    }
}

void Backend::step(BigInt step_idx) {
    prepare_run(step_idx);
    execute_phases(step_idx, 0, static_cast<int>(plan_->slots.size()));
    if (timer_)
        timer_->step_done();
}

void Backend::run(int n_steps) {
    for (int s = 0; s < n_steps; ++s)
        step(s);
}

void Backend::destroy_components() {
    DestroyContext ctx{context_, frame_};
    lifecycle_.destroy_all([&](Component& c) { c.destroy(ctx); });
}

void Backend::retrieve(cpu::Frame& frame) { cuda::to_cpu(frame_, frame); }

void Backend::sync() {
    cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream));
    cudaStreamSynchronize(static_cast<cudaStream_t>(context_.defer_stream));
}

void Backend::shutdown() {
    if (!lifecycle_.empty()) {
        sync();
        destroy_components();
    }
    if (context_.host_box.resident)
        ATV_CUDA_CHECK(cudaFree(const_cast<Box*>(context_.host_box.resident)));
    cuda::destroy(frame_);
    frame_ = {};
    auto main = static_cast<cudaStream_t>(context_.stream);
    auto defer = static_cast<cudaStream_t>(context_.defer_stream);
    if (main)
        cudaStreamDestroy(main);
    if (defer)
        cudaStreamDestroy(defer);
    context_ = {};
    plan_ = nullptr;
    cuda::unregister_log_callback();
}

} // namespace cuda

// ── Explicit instantiations (CUDA) ───────────────────────────────

template class Driver<md::Phase, cuda::device_tag>;
template class Driver<minimizer::Phase, cuda::device_tag>;
template class Driver<evaluate::Phase, cuda::device_tag>;
template class Driver<scf::Phase, cuda::device_tag>;

} // namespace ATV
