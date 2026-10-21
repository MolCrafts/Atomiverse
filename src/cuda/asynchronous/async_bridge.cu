/// @file cuda/asynchronous/async_bridge.cu
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

#include "atomiverse/cuda/asynchronous/async_bridge.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace {

__global__ void capture_coordinates(Frame::Atoms atoms, motion_t* x,
                                    motion_t* y, motion_t* z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    x[i] = load_position(atoms.x, atoms.x_correction, i);
    y[i] = load_position(atoms.y, atoms.y_correction, i);
    z[i] = load_position(atoms.z, atoms.z_correction, i);
}

__global__ void capture_metadata(int2* destination, int stage, int step) {
    *destination = make_int2(stage, step);
}

/// Immutable build-owned callback descriptor; stream-ordered metadata carries
/// the varying stage/step so host queuing cannot overwrite an in-flight slot.
struct CallbackPayload {
    AsyncBridge* self;
    cuda::Context* runtime;
    cuda::Frame* frame;
    const int2* metadata;
    int slot;
};

} // anonymous namespace

void AsyncBridge::capture_positions(const Frame& frame, motion_t* x,
                                    motion_t* y, motion_t* z, void* stream) {
    const int n = frame.n_atoms();
    if (n == 0)
        return;
    capture_coordinates<<<grid_size(n), kDefaultBlock, 0,
                          static_cast<cudaStream_t>(stream)>>>(frame.atoms, x,
                                                               y, z, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void AsyncBridge::build(const BuildContext& ctx) {
    callback_error_ = nullptr;
    int2* device_metadata = nullptr;
    int2* host_metadata = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&device_metadata, kRingDepth * sizeof(int2)));
    ATV_CUDA_CHECK(cudaMallocHost(&host_metadata, kRingDepth * sizeof(int2)));
    device_metadata_ = device_metadata;
    host_metadata_ = host_metadata;
    auto* payloads = new CallbackPayload[kRingDepth];
    callback_payloads_ = payloads;
    for (int slot = 0; slot < kRingDepth; ++slot)
        payloads[slot] = {this, &ctx.runtime, &ctx.frame, host_metadata + slot,
                          slot};
    for (int i = 0; i < kRingDepth; ++i) {
        cudaEvent_t ev;
        ATV_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        pack_events_[i] = ev;
        cudaEvent_t dv;
        ATV_CUDA_CHECK(cudaEventCreateWithFlags(&dv, cudaEventDisableTiming));
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
        ATV_CUDA_CHECK(cudaStreamWaitEvent(main_stream, d2h_event, 0));

    // 1. Freeze the captured payload into the device-ring slot on the COMPUTE
    //    stream. Ordered after this step's physics kernels and before the next
    //    step's kernels overwrite the frame -- this closes the tearing hazard.
    freeze_slot(ctx, slot);
    capture_metadata<<<1, 1, 0, main_stream>>>(
        static_cast<int2*>(device_metadata_) + slot, ctx.stage, ctx.step);
    ATV_CUDA_CHECK(cudaGetLastError());

    // 2. Order the defer stream after the freeze (non-blocking).
    ATV_CUDA_CHECK(cudaEventRecord(pack_event, main_stream));
    ATV_CUDA_CHECK(cudaStreamWaitEvent(defer_stream, pack_event, 0));

    // 3. D2H the FROZEN slot (never the live frame) on the defer stream.
    d2h_slot(ctx, slot);
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(static_cast<int2*>(host_metadata_) + slot,
                        static_cast<const int2*>(device_metadata_) + slot,
                        sizeof(int2), cudaMemcpyDeviceToHost, defer_stream));

    // 3b. Record this slot's D2H-completion on the defer stream so a later
    //     freeze into the same slot (step + kRingDepth) waits at step 0 above.
    ATV_CUDA_CHECK(cudaEventRecord(d2h_event, defer_stream));
    d2h_recorded_[slot] = true;

    // 4. Snapshot boundary: enqueue a host callback that runs after the D2H
    //    above completes. The callback fires on a CUDA worker thread; the main
    //    thread is free to issue the next step's kernels in parallel.
    auto* payload = static_cast<CallbackPayload*>(callback_payloads_) + slot;
    ATV_CUDA_CHECK(cudaLaunchHostFunc(defer_stream,
                                      &AsyncBridge::cuda_callback_, payload));
}

void AsyncBridge::destroy(const DestroyContext& ctx) {
    // Drain any in-flight defer-stream callbacks before tearing down pinned
    // buffers — the worker-thread callback would dereference freed memory
    // otherwise. This sync is at teardown only (not on the per-step path), so
    // it does not affect the zero-sync hot-path contract.
    auto defer_stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    ATV_CUDA_CHECK(cudaStreamSynchronize(defer_stream));

    std::exception_ptr failure = callback_error_;
    try {
        on_destroy(ctx);
    } catch (...) {
        if (!failure)
            failure = std::current_exception();
    }
    delete[] static_cast<CallbackPayload*>(callback_payloads_);
    ATV_CUDA_CHECK(cudaFree(device_metadata_));
    ATV_CUDA_CHECK(cudaFreeHost(host_metadata_));
    callback_payloads_ = nullptr;
    device_metadata_ = nullptr;
    host_metadata_ = nullptr;
    for (int i = 0; i < kRingDepth; ++i) {
        if (pack_events_[i]) {
            ATV_CUDA_CHECK(
                cudaEventDestroy(static_cast<cudaEvent_t>(pack_events_[i])));
            pack_events_[i] = nullptr;
        }
        if (d2h_events_[i]) {
            ATV_CUDA_CHECK(
                cudaEventDestroy(static_cast<cudaEvent_t>(d2h_events_[i])));
            d2h_events_[i] = nullptr;
        }
    }
    callback_error_ = nullptr;
    if (failure)
        std::rethrow_exception(failure);
}

void AsyncBridge::cuda_callback_(void* user_data) {
    // CUDA worker thread. Reconstruct the LaunchContext from the captured
    // pointer/scalar fields and invoke the virtual user hook for the frozen
    // slot. The payload stays allocated until destroy() drains callbacks.
    auto* payload = static_cast<CallbackPayload*>(user_data);
    LaunchContext ctx{*payload->runtime, *payload->frame, payload->metadata->x,
                      payload->metadata->y};
    // A CUDART callback must never unwind through the driver. Report failures
    // on the owning thread after destroy() drains and frees every ring slot.
    if (!payload->self->callback_error_) {
        try {
            payload->self->on_data_ready(ctx, payload->slot);
        } catch (...) {
            payload->self->callback_error_ = std::current_exception();
        }
    }
}

} // namespace cuda
} // namespace ATV
