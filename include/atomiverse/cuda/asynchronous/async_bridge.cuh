/// @file cuda/asynchronous/async_bridge.cuh
/// @brief AsyncBridge -- base class for all CUDA async-D2H observer/recorder components.
///
/// Encapsulates the sidecar pattern shared by every CUDA observer and recorder,
/// with a **device-side snapshot ring** that fixes the cross-stream
/// read-after-write tearing hazard: the captured payload is first frozen into a
/// device-ring slot on the **compute/main stream** (ordered after this step's
/// physics and before the next step mutates the frame), and only then copied
/// D2H from the frozen slot on the defer stream. The host therefore never sees
/// a torn snapshot (some atoms from step N, some from step N+1).
///
/// @code
///   build():   create K per-slot cudaEvents + on_build(ctx)
///   launch():  slot = ring_head++ % K
///              freeze_slot(ctx, slot)                 [compute/main stream, D2D]
///              EventRecord(pack_events_[slot], main)
///              StreamWaitEvent(defer, pack_events_[slot])
///              d2h_slot(ctx, slot)                    [defer stream, D2H frozen slot]
///              cudaLaunchHostFunc(defer, &cuda_callback_)
///              → [worker thread: on_data_ready(ctx, slot) → free payload]
///   destroy(): cudaStreamSynchronize(defer)  [drain pending callbacks]
///              → on_destroy(ctx) → destroy K cudaEvents
/// @endcode
///
/// Subclasses implement five hooks:
///   - on_build()       allocate K device-ring slots + K pinned host slots, files
///   - freeze_slot()    D2D copy the live payload into device-ring slot on the
///                      compute stream (ctx.runtime.stream) -- this is the
///                      tearing fix; it must NOT touch the defer stream
///   - d2h_slot()       cudaMemcpyAsync the frozen device-ring slot into the
///                      matching pinned host slot on the defer stream
///   - on_data_ready()  call the CPU-side observer/recorder using pinned host slot
///   - on_destroy()     free device-ring + pinned host slots, close files
///
/// The K cudaEvents and the sidecar stream-ordering logic are handled entirely
/// by this base class. Subclasses never touch cudaEventCreate/Destroy.
///
/// CPU-side logic MUST live in cpu/observer/ or cpu/recorder/ — not in subclasses.
///
/// @see cpu/observer/rdf.h, cpu/recorder/thermo.h, cpu/io/frame_writer.h
/// @see .claude/specs/async-analysis-snapshot-ring-02-device-ring.md
#pragma once

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Base class for CUDA async-D2H observer and recorder components.
///
/// All observers and recorders that follow the defer-stream sidecar pattern
/// must extend this class instead of Component.
class AsyncBridge : public Component {
  public:
    /// @brief Device-side snapshot ring depth K. Each capture freezes its
    /// payload into ring slot `(ring_head_++) % kRingDepth` on the compute
    /// stream, so back-to-back captures up to K deep never overwrite a slot
    /// whose D2H has not yet drained. Matches the CPU host-ring default K=3.
    static constexpr int kRingDepth = 3;

    /// @brief All AsyncBridge subclasses are deferrable by definition.
    Scheduling scheduling() const override { return Scheduling::deferrable; }

    /// @brief Create K per-slot CUDA events; call on_build().
    /// @param ctx Build context.
    void build(const BuildContext& ctx) final;

    /// @brief Sidecar boilerplate: freeze the payload into a device-ring slot
    ///        on the compute stream, order the defer stream after the freeze,
    ///        D2H the frozen slot, then dispatch on_data_ready via
    ///        cudaLaunchHostFunc.
    ///
    /// Calls can_run(ctx) first; returns immediately if false. The host hook
    /// runs on a CUDA worker thread after the D2H completes; the main thread
    /// is free to queue subsequent steps in parallel.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) final;

    /// @brief Call on_destroy(); destroy the K CUDA events.
    /// @param ctx Destroy context.
    void destroy(const DestroyContext& ctx) final;

  protected:
    /// @brief Allocate K device-ring slots + K pinned host slots; open files.
    ///
    /// The K CUDA events are already created before this is called.
    /// @param ctx Build context.
    virtual void on_build(const BuildContext& ctx) = 0;

    /// @brief Freeze the captured payload into device-ring slot @p slot on the
    ///        COMPUTE/main stream (`ctx.runtime.stream`).
    ///
    /// This is the tearing fix: the D2D copy is ordered on the same stream that
    /// mutates the frame, so it captures this step's state before the next step
    /// overwrites it. Subclasses must enqueue only device-to-device copies here
    /// and must NOT touch the defer stream.
    /// @param ctx  Launch context.
    /// @param slot Device-ring slot index in `[0, kRingDepth)`.
    virtual void freeze_slot(const LaunchContext& ctx, int slot) = 0;

    /// @brief Copy the frozen device-ring slot @p slot into the matching pinned
    ///        host slot on the defer stream (`ctx.runtime.defer_stream`).
    ///
    /// The defer stream has already been ordered after the freeze via
    /// cudaStreamWaitEvent before this is called. Subclasses must read only
    /// their device-ring slot here, never the live frame.
    /// @param ctx  Launch context.
    /// @param slot Device-ring slot index in `[0, kRingDepth)`.
    virtual void d2h_slot(const LaunchContext& ctx, int slot) = 0;

    /// @brief CPU-side implementation invoked after the D2H of slot @p slot.
    ///
    /// At this point pinned host slot @p slot contains valid frozen data.
    /// Subclasses call the corresponding cpu/observer/ or cpu/recorder/ function.
    /// Runs on a CUDA worker thread (defer-stream FIFO order), not the hot path.
    /// @param ctx  Launch context.
    /// @param slot Pinned host slot index in `[0, kRingDepth)`.
    virtual void on_data_ready(const LaunchContext& ctx, int slot) = 0;

    /// @brief Free device-ring + pinned host slots; close files.
    ///
    /// Called before the CUDA events are destroyed.
    /// @param ctx Destroy context.
    virtual void on_destroy(const DestroyContext& ctx) = 0;

    /// @brief Guard hook — return false to skip one launch iteration entirely.
    ///
    /// Default: always run. Override when the component requires certain frame
    /// data to be present (e.g. pairs.n_pairs, scf.P).
    /// @param ctx Launch context.
    /// @return true if launch should proceed.
    virtual bool can_run(const LaunchContext& ctx) const { return true; }

    /// @brief Per-slot CUDA events (cudaEvent_t as void*) owned by this base;
    ///        record freeze completion on the compute stream.
    void* pack_events_[kRingDepth] = {};
    /// @brief Per-slot D2H-completion events recorded on the defer stream. The
    ///        compute stream waits on a slot's D2H-done event before the next
    ///        freeze overwrites it — backpressure that prevents a fast hot loop
    ///        from lapping the ring and tearing an in-flight D2H. Device-side
    ///        wait only (cudaStreamWaitEvent), so the hot path adds no host sync.
    void* d2h_events_[kRingDepth] = {};
    /// @brief Whether slot @p i has a recorded D2H-done event yet (false until
    ///        its first capture, so the first kRingDepth freezes don't wait).
    bool d2h_recorded_[kRingDepth] = {};
    /// @brief Round-robin device-ring slot index, advanced once per launch.
    int ring_head_ = 0;

  public:
    /// @brief Stream-callback dispatcher — internal, do not call directly.
    ///
    /// Routed via `cudaLaunchHostFunc` on the defer stream after the
    /// subclass's `d2h_slot(ctx, slot)` completes. Runs on a CUDA worker
    /// thread (not the main thread). Reads the heap-allocated callback
    /// args struct (owned per-call), constructs a `LaunchContext`, and
    /// invokes the subclass's `on_data_ready()` hook for the captured slot.
    /// Frees the args.
    ///
    /// Exposed `public` so the C-callable callback signature
    /// (`cudaHostFn_t`) can reach the static dispatch without friend
    /// declarations. Not part of the user-extension API.
    static void cuda_callback_(void* user_data);
};

/// @brief Async observer — `Observer` role + defer-stream sidecar.
class AsyncObserver : public AsyncBridge {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred;
    }
};

/// @brief Async recorder — `Recorder` role + defer-stream sidecar.
class AsyncRecorder : public AsyncBridge {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred
               | RoleFlag::ExternalIO;
    }
};

} // namespace cuda
} // namespace ATV
