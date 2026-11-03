/// @file cpu/asynchronous/async_consumer.h
/// @brief `cpu::AsyncConsumer` -- the one base of every CPU asynchronous
///        consumer (online analysis observers, asynchronous recorders).
///
/// A consumer subscribes to its driver's shared `cpu::CaptureBuffer` with the
/// columns it reads and its cadence, receives each captured frame on its own
/// worker thread (`on_snapshot`, in step order) and reports a failure by
/// throwing. Capture is lossless: every step on the consumer's cadence is
/// delivered; a full buffer makes the run wait, never skips a step.
///
/// **Step convention (LAMMPS `dump` / `fix ave/correlate`).** A snapshot's
/// `step` is the number of completed steps (LAMMPS `update->ntimestep`) and
/// the snapshot is the state after exactly that many steps. With
/// `frequency() = N` the consumer receives the run's initial state when its
/// step is a multiple of N (`setup_output`, LAMMPS `Output::setup`), then the
/// state after every N-th completed step; N = 0 means every step. A step is
/// never delivered twice (LAMMPS `last_dump`), so continued runs extend one
/// series. Under a minimizer a step is an iteration, and a minimization that
/// stops early also delivers its final state (`final_output`, LAMMPS
/// `Min::run`).
///
/// Subclass contract:
///   - `capture_columns()` — the per-atom columns `on_snapshot` reads.
///   - `on_snapshot(const FrameSnapshot&)` — worker thread; local work only
///     (MPI collectives belong on the stepping thread). A throw stops the
///     run with an error naming this consumer and the cause.
///   - `finalize()` — once, on the stepping thread, after the last snapshot
///     was consumed (at the consumer's destruction).
///
/// This base declares no default binding (the phase enum is
/// driver-specific); bind explicitly or derive from the driver wrappers
/// `cpu::md::AsyncObserver` / `cpu::minimizer::AsyncObserver`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/capture_buffer.h"
#include "atomiverse/core/capture_ledger.h"
#include "atomiverse/core/types.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

class CaptureBuffer;

/// @brief Base of every CPU asynchronous consumer.
class AsyncConsumer : public cpu::Component, public CaptureSink {
  public:
    /// @brief No default binding (see file comment).
    std::vector<DefaultBinding> default_bindings() const override { return {}; }

    /// @brief Fires after completed steps N, 2N, … (`frequency() = N`).
    int frequency_phase() const override {
        const int every = frequency();
        return every > 0 ? every - 1 : 0;
    }

    /// @brief The per-atom columns `on_snapshot` reads.
    virtual CaptureColumns capture_columns() const = 0;

    /// @brief Subscribe to the driver's capture buffer.
    void build(const BuildContext& ctx) override;
    /// @brief End of a step on cadence: capture (once per step across all
    ///        consumers) and hand the snapshot to this consumer's worker.
    void launch(const LaunchContext& ctx) final;
    /// @brief Start of a run: the current state when on cadence and not
    ///        delivered yet.
    void setup_output(const LaunchContext& ctx) final;
    /// @brief Minimization stopped early: the final state unless delivered.
    void final_output(const LaunchContext& ctx) final;
    /// @brief Drain this consumer's snapshots, unsubscribe, `finalize()`.
    void destroy(const DestroyContext& ctx) final;

    /// @brief Worker thread: one captured frame, in step order.
    virtual void on_snapshot(const FrameSnapshot& snap) = 0;
    /// @brief Once, after the last snapshot (stepping thread).
    virtual void finalize() {}

    /// @brief CaptureSink: view the slot and call `on_snapshot`.
    void consume_capture(const CaptureTicket& ticket) final;
    /// @brief CaptureSink: `cpu::<name()>`.
    std::string capture_name() const override { return "cpu::" + name(); }

    /// @brief The buffer this consumer is subscribed to (null before build).
    CaptureBuffer* capture_buffer() const noexcept { return buffer_; }

  private:
    void capture_(const LaunchContext& ctx, BigInt step, CaptureKind kind);

    CaptureBuffer* buffer_ = nullptr;
    BigInt last_step_ = -1; ///< LAMMPS `last_dump`: last step delivered.
};

} // namespace cpu
} // namespace ATV
