/// @file core/capture_ledger.h
/// @brief CaptureLedger -- the host-side admission, fan-out, release and
///        failure engine shared by the CPU and the CUDA capture buffer.
///
/// One ledger per driver (owned by `cpu::CaptureBuffer` /
/// `cuda::CaptureBuffer`, which add the frame storage and the copy). The
/// ledger never touches frame data; it decides, on the stepping thread,
/// which capture a consumer's step belongs to and whether there is room for
/// it, and it runs one worker thread per consumer that hands each capture to
/// that consumer in step order.
///
/// @code
///   stepping thread (producer)            consumer k (worker thread k)
///   join(sink, step, kind)                  waits for a capture it is due on
///     same step as the open capture:          and whose data is complete
///       add sink to it (no wait)              (`ready(k, index)`)
///     else: seal the open capture,          consume_capture(ticket)
///       wait while the budget is full,      release: the slot frees when the
///       open a new capture (index, slot)      last due consumer released it
///   [backend copies the frame once]           and the capture is sealed
///   drain(): seal, wait until every capture is released, raise a failure
/// @endcode
///
/// **Lossless.** A capture is opened only when the frame `capacity` captures
/// earlier has been released by every consumer due on it, so a slot is never
/// overwritten while a consumer may still read it. When the budget is full
/// the producer waits on a condition variable — never a busy wait — while
/// the consumers keep draining; a consumer never waits on the producer. The
/// first wait of a run is logged (naming the consumers behind), the waits
/// of a run are summarised by `drain()`, and the cumulative blocked time,
/// blocking episodes and peak occupancy are kept in `stats()`.
///
/// **Capture once.** While a capture is open (the producer's pin), every
/// consumer due on the same step joins it instead of opening another: one
/// frame of the budget per captured step, whatever the number of consumers.
///
/// **Failures.** A consumer whose `consume_capture` throws is marked failed:
/// it consumes nothing more (its later captures are released unread, so the
/// producer never waits on space that will not free), the first failure is
/// recorded as "<consumer>: … failed at step S: <cause> — the run was
/// stopped", and every later `join`, `check` and the first `drain(true)`
/// throw it as `std::runtime_error`. The other consumers keep consuming the
/// captures already taken.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "atomiverse/core/capture_buffer.h"

namespace ATV {

/// @brief One consumer's handle on one capture.
struct CaptureTicket {
    std::uint64_t index = 0; ///< Capture sequence number since `start()`.
    std::uint64_t slot = 0;  ///< Buffer slot (`index % capacity`).
    long long step = 0;      ///< Step label [completed steps / iterations].
    CaptureKind kind = CaptureKind::step; ///< Why it was captured.
};

/// @brief The consumer side of a capture buffer (one per async consumer).
class CaptureSink {
  public:
    virtual ~CaptureSink() = default;
    /// @brief Consume one capture. Runs on this consumer's worker thread,
    ///        captures in step order; the frame in `ticket.slot` is
    ///        complete and immutable until this returns. A throw stops the
    ///        run (see `CaptureLedger`).
    virtual void consume_capture(const CaptureTicket& ticket) = 0;
    /// @brief Name in errors and warnings, e.g. `cuda::MrecRecorder
    ///        'traj.mrec'`.
    virtual std::string capture_name() const = 0;
};

/// @brief Host-side admission / fan-out / release engine of one driver's
///        capture buffer.
class CaptureLedger {
  public:
    /// Consumers per buffer (one bit each in a capture's due mask).
    static constexpr int kMaxConsumers = 64;

    /// @brief Result of a `join`.
    struct Join {
        CaptureTicket ticket; ///< The capture this consumer reads.
        bool opened = false;  ///< True: a new capture — copy the frame now.
    };

    /// @brief Whether consumer `id`'s data for capture `index` is complete.
    ///        Called under the ledger lock; must not block.
    using ReadyFn = std::function<bool(int id, std::uint64_t index)>;

    /// @param owner Prefix of messages (`cpu::CaptureBuffer`, …).
    explicit CaptureLedger(std::string owner);
    ~CaptureLedger();
    CaptureLedger(const CaptureLedger&) = delete;
    CaptureLedger& operator=(const CaptureLedger&) = delete;

    /// @brief Register a consumer; it is served from the next `start()`.
    /// @return Its id [0, kMaxConsumers).
    /// @throws std::logic_error while running or when full.
    int subscribe(CaptureSink& sink);
    /// @brief Remove a consumer. @throws std::logic_error while running.
    void unsubscribe(CaptureSink& sink);
    /// @brief The consumer's id, or -1.
    int id_of(const CaptureSink& sink) const;
    /// @brief Registered consumers, by id.
    const std::vector<CaptureSink*>& sinks() const noexcept { return sinks_; }

    /// @brief Frames a budget holds: `budget / frame_bytes`.
    /// @throws std::invalid_argument naming both sizes when it is 0.
    static std::uint64_t capacity_for(const std::string& owner,
                                      std::uint64_t budget,
                                      std::uint64_t frame_bytes,
                                      const std::string& what_frame);

    /// @brief Size the ring and start one worker per consumer.
    /// @param budget      Byte budget [bytes; >= frame_bytes].
    /// @param frame_bytes One captured frame [bytes; >= 1].
    /// @param ready       Data-completeness predicate (see `ReadyFn`).
    /// @param what_frame  Description of the frame for the too-small error.
    /// @throws std::invalid_argument when the budget cannot hold one frame,
    ///         naming both sizes.
    void start(std::uint64_t budget, std::uint64_t frame_bytes, ReadyFn ready,
               const std::string& what_frame);
    bool running() const noexcept { return running_; }
    /// @brief Frames the budget holds (0 before `start()`).
    std::uint64_t capacity() const noexcept { return capacity_; }

    /// @brief Producer: consumer `sink` needs the state labelled `step`.
    ///
    /// Joins the open capture of the same step, or seals it, waits while the
    /// budget is full and opens a new capture (`opened`: the caller copies
    /// the frame into `ticket.slot` now).
    /// @throws std::runtime_error carrying a consumer failure;
    ///         std::logic_error before `start()` or for an unknown sink.
    Join join(CaptureSink& sink, long long step, CaptureKind kind);

    /// @brief Producer: wake the consumers after a frame copy completed
    ///        (the backend's `ready` now reports it).
    void notify_published();

    /// @brief A consumer failed (lock-free, for the per-step check).
    bool failed() const noexcept {
        return failed_.load(std::memory_order_acquire);
    }
    /// @brief Throw the recorded failure, if any (every call).
    void check();

    /// @brief Seal the open capture and wait until every capture is consumed
    ///        and released; log this run's waits. Precondition: no frame copy
    ///        is still in flight (CUDA: the stream was synchronized), so a
    ///        capture whose data never became ready never will — it is
    ///        reported as a failure instead of waited for.
    /// @param raise Throw a recorded failure the caller has not seen yet.
    void drain(bool raise);

    /// @brief Stop and join the workers; captures not consumed are abandoned.
    void stop() noexcept;

    /// @brief Forget the accounting (a new backend run).
    void reset_stats();

    /// @brief Accounting so far.
    CaptureStats stats() const;
    /// @brief The recorded failure; empty when none.
    std::string failure() const;

  private:
    struct Consumer {
        CaptureSink* sink = nullptr;
        int id = 0;
        std::uint64_t cursor = 0; ///< Next capture index to examine.
        bool failed = false;
        std::thread thread;
    };

    void worker_(int k);
    void seal_locked_();
    void release_locked_(std::uint64_t index);
    void fail_locked_(std::string message);
    [[noreturn]] void throw_failure_locked_();
    std::string laggards_locked_() const;

    std::string owner_;
    std::vector<CaptureSink*> sinks_;
    std::vector<Consumer> consumers_;
    ReadyFn ready_;

    mutable std::mutex mtx_;
    std::condition_variable work_;  ///< → consumers
    std::condition_variable space_; ///< → producer
    std::condition_variable done_;  ///< → drain
    bool running_ = false;
    bool abandoned_ = false;
    std::uint64_t budget_ = 0;
    std::uint64_t frame_bytes_ = 0;
    std::uint64_t capacity_ = 0;
    std::uint64_t admitted_ = 0; ///< Captures opened since start.
    std::uint64_t tail_ = 0;     ///< Captures fully released since start.
    bool open_ = false;          ///< The newest capture accepts joins.
    long long open_step_ = 0;
    std::vector<std::uint64_t> mask_; ///< Due consumers, per slot.
    std::vector<int> pending_;        ///< Unreleased holds, per slot.
    std::vector<CaptureTicket> ticket_;

    std::atomic<bool> failed_{false};
    std::string failure_;
    bool failure_raised_ = false;

    std::uint64_t captures_ = 0;
    std::uint64_t peak_frames_ = 0;
    std::uint64_t blocked_episodes_ = 0;
    std::chrono::nanoseconds blocked_{0};
    std::uint64_t episodes_reported_ = 0;
    std::chrono::nanoseconds blocked_reported_{0};
};

} // namespace ATV
