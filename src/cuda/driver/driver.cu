/// @file cuda/driver/driver.cu
/// @brief CUDA Backend implementation + Driver explicit instantiations.

#include <cmath>

#include "atomiverse/core/driver_template_impl.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/log_callback.h"
#include "atomiverse/cuda/device/timer.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

// ── Backend ────────────────────────────────────────────────────────

void Backend::init() {
    cuda::register_log_callback();
    cudaStream_t main, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&main));
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    context_.stream = main;
    context_.defer_stream = defer;
}

void Backend::materialize(const cpu::Frame& frame) {
    frame_ = cpu::to_cuda(frame);

    // Populate the host-side cuda::Box snapshot on the Context. All
    // integrator + bonded + pair kernels read from this field at
    // launch instead of issuing a per-step D2H of frame.box[0,4,8],
    // and can pass the trivially-copyable struct as a kernel arg by
    // value (no per-thread 1/Lx FP64 division — uses cached inv_l).
    // For NPT runs, the integrator (`cuda::md::NoseHooverNPT`) updates
    // this field after `propagate_x_and_box_` writes the new
    // diagonal; for fixed-box (NVE/NVT) runs the build-time value
    // here is the value forever.
    auto& hb = context_.host_box;
    const auto& cell = frame.box.cell();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            hb.cell[i * 3 + j] = static_cast<real_t>(cell(i, j));
    // Ortho test: off-diagonal entries within FP roundoff. Threshold
    // lives on cuda::Box so it stays in lock-step with cpu::Box.
    hb.ortho = (std::abs(hb.cell[1]) < cuda::Box::kOrthoEps)
               && (std::abs(hb.cell[2]) < cuda::Box::kOrthoEps)
               && (std::abs(hb.cell[3]) < cuda::Box::kOrthoEps)
               && (std::abs(hb.cell[5]) < cuda::Box::kOrthoEps)
               && (std::abs(hb.cell[6]) < cuda::Box::kOrthoEps)
               && (std::abs(hb.cell[7]) < cuda::Box::kOrthoEps);
    // Derived caches: inv_l (per-axis), half_l, full inv_cell.
    for (int d = 0; d < 3; ++d) {
        real_t L = hb.cell[d * 3 + d];
        hb.inv_l[d] = (L > real_t(0)) ? real_t(1) / L : real_t(0);
        hb.half_l[d] = real_t(0.5) * L;
    }
    // Full 3x3 inverse for triclinic min_image (orthorhombic only uses
    // the per-axis inv_l above, but the inv_cell entries stay populated
    // so a path that goes triclinic-aware still works).
    const real_t a = hb.cell[0], b = hb.cell[1], c = hb.cell[2];
    const real_t d = hb.cell[3], e = hb.cell[4], f = hb.cell[5];
    const real_t g = hb.cell[6], h = hb.cell[7], i9 = hb.cell[8];
    const real_t det =
        a * (e * i9 - f * h) - b * (d * i9 - f * g) + c * (d * h - e * g);
    if (std::abs(det) > real_t(0)) {
        const real_t inv_det = real_t(1) / det;
        hb.inv_cell[0] = (e * i9 - f * h) * inv_det;
        hb.inv_cell[1] = (c * h - b * i9) * inv_det;
        hb.inv_cell[2] = (b * f - c * e) * inv_det;
        hb.inv_cell[3] = (f * g - d * i9) * inv_det;
        hb.inv_cell[4] = (a * i9 - c * g) * inv_det;
        hb.inv_cell[5] = (c * d - a * f) * inv_det;
        hb.inv_cell[6] = (d * h - e * g) * inv_det;
        hb.inv_cell[7] = (b * g - a * h) * inv_det;
        hb.inv_cell[8] = (a * e - b * d) * inv_det;
    }
    hb.periodic[0] = hb.periodic[1] = hb.periodic[2] = true;
}

void Backend::build_components() {
    BuildContext ctx{context_, frame_};
    for (auto& slot : plan_->slots)
        for (auto& op : slot.ops) {
            if (timer_)
                timer_->impl()->begin(
                    static_cast<cudaStream_t>(context_.stream));
            op.component->build(ctx);
            if (timer_) {
                timer_->impl()->end(static_cast<cudaStream_t>(context_.stream),
                                    slot.phase,
                                    op.component->name() + "/build");
            }
        }
    cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream));
    if (timer_)
        timer_->impl()->flush();
}

void Backend::build_plan(const Plan<Component>& plan) {
    BuildContext ctx{context_, frame_};
    for (auto& slot : plan.slots)
        for (auto& op : slot.ops)
            op.component->build(ctx);
}

void Backend::destroy_plan(const Plan<Component>& plan) {
    DestroyContext ctx{context_, frame_};
    for (int s = static_cast<int>(plan.slots.size()) - 1; s >= 0; --s) {
        auto& slot = plan.slots[static_cast<std::size_t>(s)];
        for (int o = static_cast<int>(slot.ops.size()) - 1; o >= 0; --o)
            slot.ops[static_cast<std::size_t>(o)].component->destroy(ctx);
    }
}

void Backend::step(int step_idx) {
    for (auto& slot : plan_->slots) {
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
    if (timer_)
        timer_->step_done();
}

void Backend::run(int n_steps) {
    for (int s = 0; s < n_steps; ++s)
        step(s);
}

void Backend::destroy_components() {
    DestroyContext ctx{context_, frame_};
    for (int s = static_cast<int>(plan_->slots.size()) - 1; s >= 0; --s) {
        auto& slot = plan_->slots[s];
        for (int o = static_cast<int>(slot.ops.size()) - 1; o >= 0; --o)
            slot.ops[o].component->destroy(ctx);
    }
}

void Backend::retrieve(cpu::Frame& frame) { cuda::to_cpu(frame_, frame); }

void Backend::sync() {
    cudaStreamSynchronize(static_cast<cudaStream_t>(context_.stream));
    cudaStreamSynchronize(static_cast<cudaStream_t>(context_.defer_stream));
}

void Backend::shutdown() {
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

} // namespace ATV
