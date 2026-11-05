/// @file cpu/asynchronous/capture_buffer.h
/// @brief cpu::CaptureBuffer -- the CPU driver's one shared, lossless frame
///        buffer for every asynchronous consumer (`cpu::AsyncConsumer`).
///
/// Owned by `cpu::Runtime` (one per driver tree) and published on
/// `Context::capture.buffer`. The admission, fan-out, release and failure
/// semantics are `ATV::CaptureLedger`'s (core/capture_ledger.h), identical to
/// `cuda::CaptureBuffer`; this class adds the storage and the copy.
///
/// @code
///   build      each consumer subscribes with its columns
///   run start  ensure(): lay out the union of the consumers' columns for
///              the frame's atom count and allocate capacity =
///              budget / frame_bytes slots, once (re-done only at a run
///              boundary after the consumer set changed)
///   end_of_step (stepping thread), each due consumer in plan order:
///              join(step): the first opens the capture (waiting while the
///              budget is full) and copies the frame into its slot once;
///              the others join the same capture
///   worker k   on_snapshot(view of the slot); the slot frees after the last
///              due consumer released it
///   run end    drain(): every capture consumed before run() returns
/// @endcode
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "atomiverse/core/capture_buffer.h"
#include "atomiverse/core/capture_ledger.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"

namespace ATV {
namespace cpu {

class Frame;
class AsyncConsumer;

/// @brief One driver's shared capture buffer (host memory).
class CaptureBuffer {
  public:
    CaptureBuffer();
    ~CaptureBuffer();
    CaptureBuffer(const CaptureBuffer&) = delete;
    CaptureBuffer& operator=(const CaptureBuffer&) = delete;

    /// @brief Bytes one captured frame of @p n_atoms atoms with @p columns
    ///        occupies (header included): the smallest valid budget.
    static std::uint64_t frame_bytes(int n_atoms, CaptureColumns columns);

    /// @brief Byte budget [bytes; >= 1]; applied at the next `ensure()`.
    /// @throws std::invalid_argument on 0.
    void set_budget(std::uint64_t bytes);
    std::uint64_t budget() const noexcept { return budget_; }

    /// @brief Register a consumer (its columns join the union). A running
    ///        buffer is drained and stopped first; the next `ensure()`
    ///        re-lays it out.
    void subscribe(AsyncConsumer& consumer);
    /// @brief Remove a consumer (draining and stopping a running buffer).
    void unsubscribe(AsyncConsumer& consumer);
    bool has_consumers() const;

    /// @brief Allocate and start when consumers are subscribed and the
    ///        buffer is not running with their layout (a no-op otherwise, so
    ///        the Runtime may call it before every step). The layout is for
    ///        the frame's atom count at that point; the arena is allocated
    ///        here, once, never in a capture.
    /// @throws std::invalid_argument when the budget cannot hold one frame.
    void ensure(const Frame& frame);
    /// @brief Re-lay out at the next `ensure()` (the population changed at
    ///        a run boundary).
    void invalidate() noexcept { dirty_ = true; }
    bool running() const noexcept { return ledger_.running(); }

    /// @brief Producer: consumer @p consumer needs the state labelled
    ///        @p step of @p frame. Opens a capture (copying the frame once)
    ///        or joins the open one of the same step; may wait for space.
    /// @throws std::runtime_error carrying a consumer failure, or when the
    ///         frame grew past the atom count the buffer was laid out for.
    void capture(AsyncConsumer& consumer, const Frame& frame, BigInt step,
                 CaptureKind kind);

    /// @brief View of a captured slot (worker side).
    FrameSnapshot view(const CaptureTicket& ticket) const;

    /// @brief Per-step check: throw a consumer failure.
    void check() { ledger_.check(); }
    /// @brief Every capture consumed; @p raise throws an unseen failure.
    void drain(bool raise) { ledger_.drain(raise); }
    /// @brief Drain quietly and join the workers.
    void stop() noexcept;
    /// @brief Forget the accounting (a new run of the owning driver).
    void reset_stats() { ledger_.reset_stats(); }

    CaptureStats stats() const { return ledger_.stats(); }
    CaptureColumns columns() const noexcept { return columns_; }

  private:
    struct Layout {
        std::size_t cell = 0, positions = 0, velocities = 0, forces = 0,
                    charges = 0, masses = 0, types = 0, tags = 0, images = 0;
        std::size_t bytes = 0;
    };
    static Layout make_layout_(int n, CaptureColumns columns);
    unsigned char* slot_(std::uint64_t slot) const;

    CaptureLedger ledger_;
    std::uint64_t budget_ = kCaptureBufferBytes;
    CaptureColumns columns_{};
    int n_atoms_ = 0;
    Layout layout_{};
    bool dirty_ = true; ///< Consumer set or budget changed since start.
    std::unique_ptr<unsigned char[]> arena_;
    std::uint64_t arena_bytes_ = 0;
    /// Captures whose copy completed (written by the producer).
    std::atomic<std::uint64_t> published_{0};
};

} // namespace cpu
} // namespace ATV
