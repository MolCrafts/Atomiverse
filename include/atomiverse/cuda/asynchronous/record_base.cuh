/// @file cuda/asynchronous/record_base.cuh
/// @brief `cuda::AsyncRecordBase<Snapshot>` -- zero-block base class for
///        CUDA-side recorder/thermo/observer components.
///
/// Provides the **non-blocking** sidecar pattern that every CUDA-side
/// recorder must follow: the subclass stages a snapshot into the device
/// staging buffer `device_snapshot_` on the compute stream; `submit_async`
/// then freezes it (D2D) into a K-deep device ring slot on the compute
/// stream — ordered before the next step overwrites the staging buffer —
/// enqueues a `cudaMemcpyAsync` D2H of that frozen slot into the matching
/// host-ring slot on the **defer stream** (NOT the compute stream), records
/// a per-slot `cudaEvent`, and submits a closure to an in-process host
/// worker thread that calls `cudaEventSynchronize` off the hot path and then
/// invokes the user-supplied `work_fn(host_ring[slot], step)`. The
/// HostWorker queue is unbounded, so a capture whose ring slot is still in
/// flight is dropped (drop_oldest, counted by `n_dropped()`) rather than
/// overwriting an undelivered snapshot.
///
/// The hot-path contract is:
///   * `submit_async(...)` does NOT call `cudaStreamSynchronize` on the
///     caller's thread (the MD step loop).
///   * The compute stream is never blocked by this base class.
///   * `cudaEventSynchronize` runs on the worker thread, not the hot path.
///
/// This contract is verified by `tests/cuda/async/test_hot_path_no_sync.cu`
/// via the opt-in counter in `cuda/async/sync_guard.cuh`.
///
/// The host worker is owned by `AsyncRecordBase` itself — a minimum-viable
/// single-thread worker that processes a queue of `std::function<void()>`.
/// `destroy()` drains the queue and joins the worker before freeing the
/// device staging buffer, the K device-ring slots, and the per-slot events,
/// so the recorder's CSV is complete when the simulation returns.
///
/// Concrete subclasses (e.g. the CUDA-side `CudaCsvRecorder` in
/// `benchmarks/md/polyethylene_tension.cpp`) launch a tiny pack-snapshot
/// kernel on the compute stream that fills `device_snapshot_`, then call
/// `submit_async(step, [](const Snapshot& s, int64_t step) { ... })`.
///
/// @see cuda/async/sync_guard.cuh for the sync-counter wrapper.
/// @see .claude/specs/async-analysis-snapshot-ring-02-device-ring.md
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

namespace detail {

/// @brief Minimum-viable single-thread worker that processes a FIFO queue
///        of `std::function<void()>` closures.
///
/// Owned by `AsyncRecordBase` (one worker per recorder instance). Lifetime:
///   * Constructed by `AsyncRecordBase::build()` -- spawns the worker thread.
///   * `submit(fn)` pushes a closure onto the queue; worker dequeues and runs.
///   * `join()` is called by `AsyncRecordBase::destroy()` to drain the queue
///     and stop the thread cleanly.
///
/// Defined in `src/cuda/async/record_base.cu` so the std::thread / mutex /
/// condition_variable machinery lives in a single translation unit.
class HostWorker {
  public:
    HostWorker();
    ~HostWorker();

    HostWorker(const HostWorker&) = delete;
    HostWorker& operator=(const HostWorker&) = delete;
    HostWorker(HostWorker&&) = delete;
    HostWorker& operator=(HostWorker&&) = delete;

    /// @brief Lazily spawn the worker thread on the first call.
    void start();

    /// @brief Push a closure onto the worker queue.
    /// @param fn Closure invoked on the worker thread (must be self-contained).
    void submit(std::function<void()> fn);

    /// @brief Drain the queue and join the worker thread. Idempotent.
    void join();

  private:
    void loop();

    std::thread thread_;
    std::deque<std::function<void()>> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    bool started_{false};
    bool joined_{false};
};

} // namespace detail

/// @brief Driver-agnostic base for CUDA recorders/observers that need
///        host-thread access to per-step device state.
///
/// @tparam Snapshot Plain-POD struct holding the fields the recorder
///         needs on the host. Must be trivially copyable so a single
///         `cudaMemcpyAsync` can stage it from device to host.
///
/// **Zero-block contract:** `submit_async(...)` returns to the caller
/// without calling `cudaStreamSynchronize`. The D2H copy uses the defer
/// stream (`ctx.runtime.defer_stream` resolved at `build()` time), so the
/// compute stream is never blocked.
///
/// **Lifecycle:**
///   * `build(ctx)` -- allocates `device_snapshot_` (staging) + K `device_ring_`
///     slots via `cudaMalloc` (owning), creates K `pack_done_event_` + K
///     `copy_done_event_` via `cudaEventCreateWithFlags(disable_timing)`
///     (owning), captures `compute_stream_`/`defer_stream_` from `ctx.runtime`
///     (non-owning), starts the host worker.
///   * `submit_async(step, work_fn)` -- freeze staging→ring slot (compute) +
///     D2H ring slot→host slot (defer) + per-slot event records + worker job,
///     or drop if the slot is still in flight.
///   * `destroy(ctx)` -- joins the worker, frees `device_snapshot_` + the K
///     ring slots, destroys the 2·K events, resets members to nullptr.
template <typename Snapshot>
class AsyncRecordBase : public Component {
  public:
    AsyncRecordBase() = default;

    /// @brief Allocate staging buffer + K device-ring slots (owning) + 2·K
    ///        cudaEvents (owning); start host worker.
    /// @param ctx Build context — supplies the compute + defer stream
    ///        pointers (non-owning handles borrowed from `cuda::Backend`).
    ///
    /// Two per-slot event arrays are owned by this base (indexed by ring slot):
    ///   * `pack_done_event_[slot]` — recorded on the compute stream right
    ///     after the freeze D2D into the ring slot; lets the defer stream
    ///     `cudaStreamWaitEvent` on it so the D2H reads fully-frozen bytes.
    ///   * `copy_done_event_[slot]` — recorded on the defer stream after the
    ///     D2H; the worker thread `cudaEventSynchronize`s on this one.
    void build(const BuildContext& ctx) override {
        if (device_snapshot_ != nullptr || device_ring_[0] != nullptr) {
            throw std::runtime_error(
                "cuda::AsyncRecordBase::build called twice without destroy");
        }
        auto alloc = [](Snapshot** p) {
            cudaError_t e =
                cudaMalloc(reinterpret_cast<void**>(p), sizeof(Snapshot));
            if (e != cudaSuccess)
                throw std::runtime_error(
                    std::string("cuda::AsyncRecordBase: cudaMalloc failed: ")
                    + cudaGetErrorString(e));
        };
        // Subclass staging buffer (written by the subclass pack kernel) plus a
        // K-deep device-ring of frozen slots so back-to-back captures never
        // overwrite a slot whose D2H has not yet drained.
        alloc(&device_snapshot_);
        for (int i = 0; i < kRingDepth; ++i) {
            alloc(&device_ring_[i]);
            cudaEventCreateWithFlags(&pack_done_event_[i],
                                     cudaEventDisableTiming);
            cudaEventCreateWithFlags(&copy_done_event_[i],
                                     cudaEventDisableTiming);
            slot_busy_[i].store(false, std::memory_order_relaxed);
        }
        compute_stream_ = static_cast<cudaStream_t>(ctx.runtime.stream);
        defer_stream_ = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        ring_head_ = 0;
        n_dropped_.store(0, std::memory_order_relaxed);
        worker_.start();
    }

    /// @brief Tear down: drain worker, free device buffer, destroy events.
    /// @param ctx Destroy context (unused; signature mandated by base).
    void destroy(const DestroyContext& /*ctx*/) override {
        worker_.join();
        if (device_snapshot_) {
            cudaFree(device_snapshot_);
            device_snapshot_ = nullptr;
        }
        for (int i = 0; i < kRingDepth; ++i) {
            if (device_ring_[i]) {
                cudaFree(device_ring_[i]);
                device_ring_[i] = nullptr;
            }
            if (pack_done_event_[i]) {
                cudaEventDestroy(pack_done_event_[i]);
                pack_done_event_[i] = nullptr;
            }
            if (copy_done_event_[i]) {
                cudaEventDestroy(copy_done_event_[i]);
                copy_done_event_[i] = nullptr;
            }
        }
        compute_stream_ = nullptr;
        defer_stream_ = nullptr;
    }

    /// @brief Convenience: copy @p n_bytes from a device-resident source
    ///        into the snapshot buffer on the compute stream. Subclasses
    ///        call this once per snapshot field they want staged, then
    ///        call `submit_async`. Wraps `cudaMemcpyAsync` D2D so the
    ///        subclass never names the CUDA runtime API directly.
    /// @param dst Pointer inside `device_snapshot_` to write into.
    /// @param src Device-resident source.
    /// @param n_bytes Byte count.
    /// @param compute_stream Stream on which the simulation's prior step
    ///        kernels ran. Pass `ctx.runtime.stream` from `launch`.
    void stage_device_field(void* dst, const void* src, std::size_t n_bytes,
                            cudaStream_t compute_stream) {
        cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyDeviceToDevice,
                        compute_stream);
    }

    /// @brief Hot-path entry. Enqueues a non-blocking D2H of
    ///        `device_snapshot_` onto the defer stream, records the
    ///        completion event, and submits a closure to the host worker
    ///        that synchronizes on the event and invokes @p work_fn.
    /// @param step Logical simulation step index (passed verbatim to @p work_fn).
    /// @param work_fn Host-side analysis/recording function. Runs on the
    ///        worker thread, never on the hot path.
    void
    submit_async(std::int64_t step,
                 std::function<void(const Snapshot&, std::int64_t)> work_fn) {
        const int slot = ring_head_;
        ring_head_ = (ring_head_ + 1) % kRingDepth;
        // drop_oldest backpressure: the HostWorker queue is unbounded, so the
        // hot loop can outrun the consumer by far more than K steps. If this
        // slot's previous job has not finished, reusing it would overwrite an
        // in-flight snapshot (the freeze D2D and the D2H both target this
        // slot). Drop this capture instead — never block the hot loop, never
        // deliver an overwritten snapshot. A delivered snapshot is therefore
        // always its own step's frozen bytes.
        bool expected = false;
        if (!slot_busy_[slot].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            n_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // 1. Freeze the subclass-staged snapshot into device-ring slot `slot`
        //    on the compute stream. Ordered after the subclass's pack kernel
        //    and before the next step overwrites device_snapshot_.
        cudaMemcpyAsync(device_ring_[slot], device_snapshot_, sizeof(Snapshot),
                        cudaMemcpyDeviceToDevice, compute_stream_);
        cudaEventRecord(pack_done_event_[slot], compute_stream_);
        // 2. Order the defer stream after the freeze, then D2H the frozen slot
        //    into its host-ring mirror (never device_snapshot_, which the next
        //    step may already be overwriting).
        cudaStreamWaitEvent(defer_stream_, pack_done_event_[slot], 0);
        cudaMemcpyAsync(&host_ring_[slot], device_ring_[slot], sizeof(Snapshot),
                        cudaMemcpyDeviceToHost, defer_stream_);
        // 3. Record completion event on the defer stream.
        cudaEventRecord(copy_done_event_[slot], defer_stream_);
        // 4. Submit closure to worker. The host-ring slot is a member kept
        //    alive by destroy() joining the worker before tearing down. The
        //    worker frees the slot after consuming it, so it can be reused.
        cudaEvent_t ev = copy_done_event_[slot];
        Snapshot* snap_ptr = &host_ring_[slot];
        std::atomic<bool>* busy = &slot_busy_[slot];
        worker_.submit(
            [ev, snap_ptr, busy, step, work_fn = std::move(work_fn)]() {
                cudaEventSynchronize(ev);
                work_fn(*snap_ptr, step);
                busy->store(false, std::memory_order_release);
            });
        // submit_async returns immediately: no cudaStreamSynchronize on the
        // caller's thread. This is the load-bearing invariant verified by
        // tests/cuda/async/test_hot_path_no_sync.cu.
    }

    /// @brief Snapshots dropped because their ring slot was still in flight.
    /// @return Cumulative drop count since `build()`.
    unsigned long long n_dropped() const noexcept {
        return n_dropped_.load(std::memory_order_relaxed);
    }

  public:
    /// @brief Device-side snapshot ring depth K. Each `submit_async` freezes
    ///        the staged snapshot into ring slot `(ring_head_++) % kRingDepth`,
    ///        so back-to-back captures up to K deep never overwrite a slot
    ///        whose D2H has not yet drained. Matches AsyncBridge / the CPU
    ///        host-ring default K=3.
    static constexpr int kRingDepth = 3;

  protected:
    /// @brief Device-resident staging buffer (owning, freed in `destroy()`).
    ///        Subclasses' `launch()` kernels write into it on the compute
    ///        stream before calling `submit_async`, which then freezes it into
    ///        the device-ring.
    Snapshot* device_snapshot_ = nullptr;

    /// @brief Device-resident snapshot ring (owning). `submit_async` D2D-copies
    ///        `device_snapshot_` into the current slot on the compute stream so
    ///        the frozen bytes survive until the host worker consumes them.
    Snapshot* device_ring_[kRingDepth] = {};

    /// @brief Host-resident snapshot ring filled by the async D2H. Slot read on
    ///        the worker thread (after `cudaEventSynchronize`), never on the
    ///        hot path.
    Snapshot host_ring_[kRingDepth] = {};

    /// @brief Per-slot event: freeze (D2D into the ring slot) done on the
    ///        compute stream. The defer stream waits on it before the D2H.
    cudaEvent_t pack_done_event_[kRingDepth] = {};

    /// @brief Per-slot event: D2H of the ring slot done on the defer stream;
    ///        the worker thread polls it before reading the host-ring slot.
    cudaEvent_t copy_done_event_[kRingDepth] = {};

    /// @brief Round-robin device-ring slot index, advanced once per submit.
    int ring_head_ = 0;

    /// @brief Per-slot in-flight flag. Set when a slot's freeze+D2H is enqueued,
    ///        cleared by the worker after it consumes the host-ring slot. A
    ///        capture whose slot is still set is dropped (drop_oldest).
    std::atomic<bool> slot_busy_[kRingDepth] = {};

    /// @brief Count of captures dropped because their slot was in flight.
    std::atomic<unsigned long long> n_dropped_{0};

    /// @brief Compute-stream handle resolved from `ctx.runtime.stream`
    ///        at `build()`. **Non-owning** — `cuda::Backend` owns the stream.
    cudaStream_t compute_stream_ = nullptr;

    /// @brief Defer-stream handle resolved from `ctx.runtime.defer_stream`
    ///        at `build()`. **Non-owning** — `cuda::Backend` owns the stream.
    cudaStream_t defer_stream_ = nullptr;

  private:
    detail::HostWorker worker_;
};

} // namespace cuda
} // namespace ATV
