/// @file cuda/async/async_bridge.cu
/// @brief AsyncBridge sidecar boilerplate: device-ring freeze on the compute
///        stream + on-device-completion host callback via `cudaLaunchHostFunc`.
///
/// Hot-path contract: `launch()` freezes the captured payload into a device-ring
/// slot on the compute stream (one D2D per subclass via `freeze_slot`), records
/// a per-slot event, has the defer stream wait on it, issues exactly one D2H of
/// the frozen slot per subclass (via `d2h_slot`), records a host callback to run
/// `on_data_ready(ctx, slot)` when the D2H completes, and returns immediately.
/// No `cudaStreamSynchronize` — the main thread never waits during the per-step
/// loop. The compute-stream freeze is what closes the cross-stream
/// read-after-write tearing hazard: the snapshot is captured before the next
/// step mutates the frame, not from live device memory while it races ahead.

#include "atomiverse/cuda/async/async_bridge.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace {

/// @brief Per-call callback payload. Heap-allocated by `launch()` and
///        freed by `cuda_callback_` after the user hook returns.
///
/// `LaunchContext` is non-trivially-copyable (it holds reference members), so
/// we capture its scalar fields and the addresses of its reference targets
/// here, plus the device-ring slot the freeze/D2H used. Lifetime: from the
/// `cudaLaunchHostFunc` enqueue in `launch()` until the worker thread invokes
/// `cuda_callback_`. The defer stream's FIFO ordering serializes callbacks, so
/// each instance has at most ~K outstanding callbacks queued (K = how many
/// steps the GPU is ahead of the host) — heap traffic is bounded and not on the
/// GPU critical path (~50 ns / step on Hopper).
struct CallbackPayload {
    AsyncBridge* self;
    cuda::Context* runtime;
    cuda::Frame* frame;
    int stage;
    int step;
    int slot;
};

} // anonymous namespace

void AsyncBridge::build(const BuildContext& ctx) {
    for (int i = 0; i < kRingDepth; ++i) {
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        pack_events_[i] = ev;
        cudaEvent_t dv;
        cudaEventCreateWithFlags(&dv, cudaEventDisableTiming);
        d2h_events_[i] = dv;
        d2h_recorded_[i] = false;
    }
    ring_head_ = 0;
    on_build(ctx);
}

void AsyncBridge::launch(const LaunchContext& ctx) {
    if (!can_run(ctx))
        return;

    const int slot = ring_head_;
    ring_head_ = (ring_head_ + 1) % kRingDepth;

    auto main_stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto defer_stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    auto pack_event = static_cast<cudaEvent_t>(pack_events_[slot]);
    auto d2h_event = static_cast<cudaEvent_t>(d2h_events_[slot]);

    // 0. Slot-reuse backpressure: make the COMPUTE stream wait for this slot's
    //    PRIOR D2H (defer stream) to finish before the freeze below overwrites
    //    it. Without this, a hot loop faster than the consumer laps the K-deep
    //    ring and a freeze clobbers a slot whose D2H is still draining -> torn
    //    snapshot. Device-side wait only (no host sync): the hot path simply
    //    throttles to the drain rate once K captures are in flight.
    if (d2h_recorded_[slot])
        cudaStreamWaitEvent(main_stream, d2h_event, 0);

    // 1. Freeze the captured payload into the device-ring slot on the COMPUTE
    //    stream. Ordered after this step's physics kernels and before the next
    //    step's kernels overwrite the frame -- this closes the tearing hazard.
    freeze_slot(ctx, slot);

    // 2. Order the defer stream after the freeze (non-blocking).
    cudaEventRecord(pack_event, main_stream);
    cudaStreamWaitEvent(defer_stream, pack_event, 0);

    // 3. D2H the FROZEN slot (never the live frame) on the defer stream.
    d2h_slot(ctx, slot);

    // 3b. Record this slot's D2H-completion on the defer stream so a later
    //     freeze into the same slot (step + kRingDepth) waits at step 0 above.
    cudaEventRecord(d2h_event, defer_stream);
    d2h_recorded_[slot] = true;

    // 4. Snapshot boundary: enqueue a host callback that runs after the D2H
    //    above completes. The callback fires on a CUDA worker thread; the main
    //    thread is free to issue the next step's kernels in parallel.
    auto* payload = new CallbackPayload{this,      &ctx.runtime, &ctx.frame,
                                        ctx.stage, ctx.step,     slot};
    cudaLaunchHostFunc(defer_stream, &AsyncBridge::cuda_callback_, payload);
}

void AsyncBridge::destroy(const DestroyContext& ctx) {
    // Drain any in-flight defer-stream callbacks before tearing down pinned
    // buffers — the worker-thread callback would dereference freed memory
    // otherwise. This sync is at teardown only (not on the per-step path), so
    // it does not affect the zero-sync hot-path contract.
    auto defer_stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    cudaStreamSynchronize(defer_stream);

    on_destroy(ctx);
    for (int i = 0; i < kRingDepth; ++i) {
        if (pack_events_[i]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(pack_events_[i]));
            pack_events_[i] = nullptr;
        }
        if (d2h_events_[i]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(d2h_events_[i]));
            d2h_events_[i] = nullptr;
        }
    }
}

void AsyncBridge::cuda_callback_(void* user_data) {
    // CUDA worker thread. Reconstruct the LaunchContext from the captured
    // pointer/scalar fields and invoke the virtual user hook for the frozen
    // slot. Then free the payload — single owner pattern.
    auto* payload = static_cast<CallbackPayload*>(user_data);
    LaunchContext ctx{*payload->runtime, *payload->frame, payload->stage,
                      payload->step};
    payload->self->on_data_ready(ctx, payload->slot);
    delete payload;
}

} // namespace cuda
} // namespace ATV
