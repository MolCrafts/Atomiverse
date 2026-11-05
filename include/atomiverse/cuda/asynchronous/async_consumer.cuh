/// @file cuda/asynchronous/async_consumer.cuh
/// @brief cuda::AsyncConsumer -- the one base of every CUDA asynchronous
///        consumer: recorders (`MrecRecorder`, `XyzRecorder`, `Thermo`,
///        `md::PressureSeries`, `DensityRecorder`) and online analysis
///        observers (`RDF`, `MullikenObs`).
///
/// A consumer subscribes to its driver's shared `cuda::CaptureBuffer` with
/// the columns it reads (and an optional device payload block), receives
/// every captured frame on its own worker thread (`consume`, in step order)
/// and reports a failure by throwing — the run then stops with an error
/// naming the consumer and the cause. Capture is lossless: a full buffer
/// makes the run wait, never skips a step.
///
/// **Iron law 10.** `before_launch` is host-only (ledger admission: no
/// device read; a wait only when the byte budget is exhausted), `launch`
/// enqueues device work only and is graph-capturable (the label comes from
/// the device step), and the frame reaches the host through pinned mapped
/// memory polled off the stepping thread.
///
/// **Step convention (LAMMPS `dump`).** A capture's step is the number of
/// completed steps (LAMMPS `update->ntimestep`) and the frame is the state
/// after exactly that many steps. With `frequency() = N` a consumer
/// receives the run's initial state when its step is a multiple of N
/// (`setup_output`, LAMMPS `Output::setup`) and the state after every N-th
/// completed step; a step is never delivered twice (LAMMPS `last_dump`), so
/// `run(frame, 4); run(frame, 4)` at N = 2 delivers 0, 2, 4, 6, 8. Under
/// `cuda::minimizer::Minimize` a step is an iteration and a minimization that
/// stops early also delivers its final state (`final_output`, LAMMPS
/// `Min::run`). LAMMPS src/output.cpp / src/min.cpp, e8c4c278a8.
#pragma once

#include <cstdint>
#include <string>

#include "atomiverse/core/capture_buffer.h"
#include "atomiverse/core/capture_ledger.h"
#include "atomiverse/core/role.h"
#include "atomiverse/cuda/asynchronous/capture_buffer.cuh"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Base of every CUDA asynchronous consumer.
class AsyncConsumer : public Component, public CaptureSink {
  public:
    /// @brief The per-atom columns `consume` reads.
    virtual CaptureColumns capture_columns() const = 0;

    /// @brief Bytes of this consumer's device payload block, fixed at build
    ///        (0: none). @see payload_source
    virtual std::uint64_t payload_bytes(const BuildContext& ctx) const {
        (void)ctx;
        return 0;
    }

    /// @brief Fires after completed steps N, 2N, … (`frequency() = N`).
    int frequency_phase() const override {
        const int every = frequency();
        return every > 0 ? every - 1 : 0;
    }
    /// @brief The capture kernels read build-time pointers and device state
    ///        (the label is `runtime.device_step + 1`); admission is
    ///        host-side, in `before_launch`. A consumer whose capture is
    ///        conditional (`can_run`) overrides this to false.
    bool graph_capturable() const override { return true; }
    /// @brief The captured kernels bake the buffer's pointers.
    std::uint64_t graph_epoch() const override;

    /// @brief Subscribe to the driver's capture buffer, then `on_build`.
    void build(const BuildContext& ctx) final;
    /// @brief Host admission of this step's capture (may wait while the byte
    ///        budget is full). @throws std::runtime_error carrying a consumer
    ///        failure.
    void before_launch(const LaunchContext& ctx) final;
    /// @brief Enqueue the capture (device work only).
    void launch(const LaunchContext& ctx) final;
    /// @brief Start of a run: the current state when on cadence and not yet
    ///        delivered. `Thermo` overrides it (LAMMPS prints every setup).
    void setup_output(const LaunchContext& ctx) override;
    /// @brief Minimization stopped early: the final state unless delivered.
    void final_output(const LaunchContext& ctx) override;
    /// @brief Unsubscribe (every due capture consumed first), then
    ///        `on_destroy` (the buffer's accounting is still readable).
    void destroy(const DestroyContext& ctx) final;

    /// @brief CaptureSink: the slot's view → `consume`.
    void consume_capture(const CaptureTicket& ticket) final;
    /// @brief CaptureSink: `cuda::<name()>`.
    std::string capture_name() const override { return "cuda::" + name(); }

    /// @brief The buffer this consumer is subscribed to (null before build).
    CaptureBuffer* capture_buffer() const noexcept { return buffer_; }

  protected:
    AsyncConsumer() = default;
    AsyncConsumer(AsyncConsumer&&) noexcept = default;
    AsyncConsumer& operator=(AsyncConsumer&&) noexcept = default;

    /// @brief Allocate consumer state; open files (after subscription).
    virtual void on_build(const BuildContext& ctx) { (void)ctx; }
    /// @brief Worker thread: one captured frame, in step order.
    virtual void consume(const CaptureView& view) = 0;
    /// @brief Release consumer state; close files (after the last frame).
    virtual void on_destroy(const DestroyContext& ctx) { (void)ctx; }
    /// @brief Skip this launch (e.g. the SCF density is not there yet).
    ///        Requires `graph_capturable() == false`.
    virtual bool can_run(const LaunchContext& ctx) const {
        (void)ctx;
        return true;
    }
    /// @brief Refuse a capture before it is admitted (e.g. a closed store):
    ///        throw `std::logic_error`. Default: accept.
    virtual void require_open() const {}
    /// @brief Device source of the payload block (`payload_bytes` bytes),
    ///        resolved at each launch. Default: none.
    virtual const void* payload_source(const LaunchContext& ctx) const {
        (void)ctx;
        return nullptr;
    }

    /// @brief Capture the current state now, eagerly, labelled @p step
    ///        (setup / final passes).
    void capture_now(const LaunchContext& ctx, BigInt step, CaptureKind kind);

    /// @brief Last step delivered (LAMMPS `last_dump`); -1 before any.
    BigInt last_step() const noexcept { return last_step_; }

    friend class CaptureBuffer;

  private:
    CaptureBuffer* buffer_ = nullptr;
    BigInt last_step_ = -1;
    bool admitted_ = false; ///< before_launch admitted this launch
};

/// @brief Asynchronous observer: `Observer` role.
class AsyncObserver : public AsyncConsumer {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred;
    }
};

/// @brief Asynchronous recorder: `Recorder` role (external I/O).
class AsyncRecorder : public AsyncConsumer {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred
               | RoleFlag::ExternalIO;
    }
};

} // namespace cuda
} // namespace ATV
