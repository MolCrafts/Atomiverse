/// @file cuda/asynchronous/capture_buffer.cuh
/// @brief cuda::CaptureBuffer -- the CUDA driver's one shared, lossless frame
///        buffer for every asynchronous consumer (`cuda::AsyncConsumer`):
///        trajectory / thermo / pressure recorders and online analysis.
///
/// Owned by `cuda::Backend` (one per driver) and published on
/// `Context::capture`. Admission, fan-out, release and failure semantics are
/// `ATV::CaptureLedger`'s (core/capture_ledger.h) — identical to
/// `cpu::CaptureBuffer`; this class adds pinned, host-mapped storage the
/// device writes directly and the capture kernels (iron law 10).
///
/// @code
///   run boundary   ensure(): lay out the union of the consumers' columns
///                  (+ their payload blocks) and allocate capacity =
///                  budget / frame_bytes pinned slots, once; copy the static
///                  identity (tags, types, species) once
///   stepping thread, each due consumer, before its launch is enqueued
///                  join(step) on the host ledger: the first consumer of a
///                  step opens a capture — with space this is host counters
///                  only, no device read, no wait; full: wait on a condition
///                  variable for a release (the sanctioned exception to iron
///                  law 10) — the others join it
///   compute stream, each due consumer (graph-capturable, device step)
///     acquire_k    the first launch of a step takes slot captures % capacity
///                  (a later one of the same step label reuses it)
///     pack_k       the first launch freezes x, v, f, q, m, tag, image, cell,
///                  scalars and the step into the slot — once per step
///     payload_k    this consumer's own payload block (e.g. the SCF density)
///     publish_k    this consumer's completion word := captures (release)
///   worker k       polls its word, consumes the slot, releases it; the slot
///                  frees when the last due consumer released it
///   run end        retrieve: sync, drain — every capture consumed
/// @endcode
///
/// The host never reads device state per step: the cadence is the host step
/// counter, admission is host-side, and a capture's completion travels
/// through a mapped word the worker polls off the stepping thread.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "atomiverse/core/capture_buffer.h"
#include "atomiverse/core/capture_ledger.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/system_scalars.h"

namespace ATV {
namespace cuda {

class AsyncConsumer;
struct BuildContext;
struct LaunchContext;

/// @brief Host view of one captured frame (pinned slot), as a consumer reads
///        it on its worker thread. A column no consumer of the buffer asked
///        for is null; values are FP64 (positions primary + correction,
///        forces decoded). Rows are in device order.
struct CaptureView {
    long long step = 0;                   ///< [completed steps]
    CaptureKind kind = CaptureKind::step; ///< Why it was captured.
    int n_atoms = 0;
    const double* cell = nullptr; ///< 3×3 row-major [Å].
    const SystemScalars* scalars = nullptr;
    const double *x = nullptr, *y = nullptr, *z = nullptr;    ///< [Å]
    const double *vx = nullptr, *vy = nullptr, *vz = nullptr; ///< [Å/fs]
    const double *fx = nullptr, *fy = nullptr, *fz = nullptr; ///< [eV/Å]
    const double* charge = nullptr;                           ///< [e]
    const double* mass = nullptr;                             ///< [amu]
    const std::int64_t* tag = nullptr;                        ///< Global tags.
    const ImageInt* image = nullptr; ///< Row-major 3 × n.
    /// Static identity, copied once when the buffer was allocated (null when
    /// the device frame has no such column).
    const int* type = nullptr;
    const int* atomic_number = nullptr;
    const std::int64_t* tag_at_build = nullptr;
    /// This consumer's payload block (null when it declared none).
    const unsigned char* payload = nullptr;
    std::uint64_t payload_bytes = 0;
};

/// @brief One CUDA driver's shared capture buffer (pinned host memory).
class CaptureBuffer {
  public:
    CaptureBuffer();
    ~CaptureBuffer();
    CaptureBuffer(const CaptureBuffer&) = delete;
    CaptureBuffer& operator=(const CaptureBuffer&) = delete;

    /// @brief Bytes one captured frame of @p n_atoms atoms with @p columns
    ///        and @p payload_bytes of consumer payload occupies (header and
    ///        alignment included): the smallest valid budget.
    static std::uint64_t frame_bytes(int n_atoms, CaptureColumns columns,
                                     std::uint64_t payload_bytes = 0);

    /// @brief Byte budget [bytes; >= 1]; applied at the next `ensure()`.
    void set_budget(std::uint64_t bytes);
    std::uint64_t budget() const noexcept { return budget_; }

    /// @brief Register a consumer (drains and stops a running buffer; the
    ///        next `ensure()` re-lays it out).
    void subscribe(AsyncConsumer& consumer);
    void unsubscribe(AsyncConsumer& consumer);
    bool has_consumers() const;

    /// @brief Allocate and start when consumers are subscribed and the buffer
    ///        is not running with their layout; otherwise a flag test (the
    ///        backend calls it before every step). Run boundary only:
    ///        allocates, and synchronizes once to copy the static identity.
    /// @throws std::invalid_argument when the budget cannot hold one frame.
    void ensure(const BuildContext& ctx);
    void invalidate() noexcept { dirty_ = true; }
    bool running() const noexcept { return ledger_.running(); }
    /// @brief Changes at every (re)allocation: part of the consumers'
    ///        step-graph key, since the captured kernels bake its pointers.
    std::uint64_t generation() const noexcept { return generation_; }

    /// @brief Host admission of @p consumer's capture labelled @p step
    ///        (see `CaptureLedger::join`).
    CaptureLedger::Join join(AsyncConsumer& consumer, long long step,
                             CaptureKind kind);

    /// @brief Enqueue @p consumer's capture on the compute stream. The label
    ///        is `*step + offset` (in-loop: the device step, +1) or `offset`
    ///        (eager, `step` null). `force` opens a new capture whatever the
    ///        device's last label (an eager capture the host just opened);
    ///        otherwise the first launch of a label packs and later launches
    ///        of the same label only publish. Device work only.
    void launch(AsyncConsumer& consumer, const LaunchContext& ctx,
                const long long* step, long long offset, bool force);

    /// @brief View of @p ticket's slot for @p consumer (worker side).
    CaptureView view(const AsyncConsumer& consumer,
                     const CaptureTicket& ticket) const;

    /// @brief Throw a consumer failure (an atomic flag test).
    void check() { ledger_.check(); }
    /// @brief Every capture consumed (precondition: the compute stream was
    ///        synchronized); @p raise throws an unseen failure once.
    void drain(bool raise) { ledger_.drain(raise); }
    /// @brief Drain quietly (stream synchronized first), join the workers.
    void stop() noexcept;
    /// @brief Stop and free the device / pinned resources.
    void release() noexcept;
    void reset_stats() { ledger_.reset_stats(); }

    CaptureStats stats() const { return ledger_.stats(); }
    std::string failure() const { return ledger_.failure(); }
    bool failed() const noexcept { return ledger_.failed(); }
    CaptureColumns columns() const noexcept { return columns_; }

  private:
    struct Impl;

    CaptureLedger ledger_;
    std::uint64_t budget_ = kCaptureBufferBytes;
    CaptureColumns columns_{};
    bool dirty_ = true;
    std::uint64_t generation_ = 0;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace ATV
