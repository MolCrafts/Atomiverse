/// @file cpu/asynchronous/analysis_executor.h
/// @brief `AnalysisExecutor` -- per-rank async worker pool for observers/recorders.
///
/// Owned by `cpu::Runtime`. Lifetime is strictly enclosed by the backend's
/// init/shutdown bracket: no construction in `launch()`, no state carried
/// across plan swaps. Uses `std::thread` + `std::condition_variable`
/// (`std::thread` deliberately — not `std::jthread`).
///
/// @see .claude/specs/async-analysis-snapshot-ring-01-uniformity.md
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "atomiverse/core/types.h"
#include "atomiverse/cpu/asynchronous/frame_snapshot.h"
#include "atomiverse/cpu/component/context.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

class AsyncComponentBase;

/// @brief Triple-buffered snapshot ring + worker pool for async analysis.
///
/// Hot path: `acquire_slot()` returns a ring slot (round-robin), the
/// `Publish` fills it, then `publish()` stamps the step and
/// `submit(obs, slot)` hands the (observer, slot) pair to a worker.
///
/// Workers dequeue (observer, slot) pairs, invoke `observer->on_snapshot(slot)`,
/// and release the slot back to the ring. Backpressure is the sole
/// `drop_oldest` path at constant ring depth K.
class AnalysisExecutor {
  public:
    /// @brief Runtime configuration, captured into the executor on `init()`.
    struct Config {
        int n_threads =
            1; ///< Worker thread count (0 = ring-only, no dispatch).
        /// @brief Global ring depth K -- the single backpressure knob.
        ///
        /// Number of snapshot slots in the ring, fixed for the executor's
        /// lifetime (never grows). The hot path uses the sole `drop_oldest`
        /// policy: when all K slots are in flight, the oldest is overwritten
        /// and `n_dropped()` is incremented. Larger K = fewer drops under a
        /// lagging worker, proportionally more resident snapshot memory
        /// (~ K * N * captured-bytes-per-atom). One global value applies to
        /// every subscriber; there is no per-component override.
        int ring_size = 3;
        /// @brief Pin each worker thread to a distinct logical CPU.
        ///
        /// Linux-only via `pthread_setaffinity_np`; macOS / other platforms
        /// silently ignore the hint. Worker N pins to logical CPU
        /// `n_cpus - 1 - (N mod n_cpus)`, i.e. the highest-numbered cores
        /// first, on the assumption that the OMP team (driving the hot
        /// path) occupies the lower-numbered cores. Off by default --
        /// without an explicit hint the analysis worker runs unpinned and
        /// the kernel scheduler decides placement.
        bool affinity_hint = false;
#ifdef ATV_HAS_MPI
        MPI_Comm parent_comm =
            MPI_COMM_WORLD; ///< Dupped into `analysis_comm()`.
#endif
    };

    AnalysisExecutor() = default;
    ~AnalysisExecutor();

    AnalysisExecutor(const AnalysisExecutor&) = delete;
    AnalysisExecutor& operator=(const AnalysisExecutor&) = delete;
    AnalysisExecutor(AnalysisExecutor&&) = delete;
    AnalysisExecutor& operator=(AnalysisExecutor&&) = delete;

    /// @brief Start worker threads and allocate the initial ring.
    ///
    /// Publishes `this` into `ctx.analysis.executor` so components can reach the
    /// executor from any `BuildContext` / `LaunchContext`. Safe to call
    /// once; a second call with an already-initialized executor is a no-op.
    /// @param cfg Runtime configuration (thread count, ring depth K).
    /// @param ctx CPU execution context into which this executor registers itself.
    void init(const Config& cfg, Context& ctx);

    /// @brief Drain queued jobs, call `finalize()` on every subscriber,
    /// and join all worker threads.
    ///
    /// After `shutdown()` the executor is back to its pre-`init()` state
    /// and can be re-initialized. Idempotent.
    void shutdown();

    /// @brief Block until every queued and in-flight job has completed,
    /// leaving the worker threads alive and the executor ready for more
    /// work.
    ///
    /// Unlike `shutdown()` this does NOT stop workers or call subscriber
    /// `finalize()`. A continuable driver calls it at the end of each
    /// `run()` so observers see every snapshot from that segment
    /// delivered before control returns to the caller. Idempotent;
    /// no-op when uninitialised.
    void quiesce();

    /// @brief Reserve a ring slot for the next snapshot.
    ///
    /// Returns a slot index in `[0, ring_capacity())`. Single `drop_oldest`
    /// behaviour: when the round-robin slot is still in flight, it is
    /// overwritten in place and `n_dropped()` is incremented. Allocation-free
    /// and non-blocking -- the hot path never stalls. Ring depth K is
    /// constant for the executor's lifetime.
    ///
    /// @return Ring slot index in `[0, ring_capacity())`.
    int acquire_slot();

    /// @brief Access the storage for a slot (for `Publish` to fill).
    /// @param idx Slot index returned by `acquire_slot()`.
    /// @return Reference to the underlying `FrameSnapshot`.
    FrameSnapshot& slot(int idx);

    /// @brief Stamp a slot as ready with the simulation step it captured.
    ///
    /// Sets `last_published_slot()` so `AsyncComponentBase::launch()` can
    /// find the slot filled by the publisher at this end-of-step.
    /// @param idx Slot index.
    /// @param step Simulation step index.
    void publish(int idx, int step);

    /// @brief Slot index most recently stamped by `publish()`, or -1 if none.
    /// Consumed by `AsyncComponentBase::launch()` when deciding which slot
    /// to submit itself against. Atomic for thread-safe reads from workers.
    int last_published_slot() const noexcept {
        return last_published_slot_.load(std::memory_order_acquire);
    }

    /// @brief Pre-size every ring slot's vectors to `n_atoms`.
    ///
    /// Called by `Publish::build()` so `launch()` never reallocates
    /// snapshot buffers. Spec §B.1 contract. Safe to call multiple times
    /// (monotonic: does not shrink).
    /// @param n_atoms Size to which each enabled buffer is resized.
    /// @param mask Union of capture masks from all registered subscribers.
    void prealloc_slots(int n_atoms, CaptureMask mask);

    /// @brief Register an async subscriber with the executor.
    ///
    /// Called by `AsyncComponentBase::build()`. Ownership remains with the
    /// driver (or the user, for lvalue-added components); the executor only
    /// holds a non-owning pointer so it can invoke `on_snapshot` on a worker
    /// and `finalize()` during shutdown.
    /// @param sub Non-owning subscriber pointer.
    void register_subscriber(AsyncComponentBase* sub);

    /// @brief Non-owning view of registered subscribers.
    ///
    /// Used by `Publish` to iterate subscribers and decide whether
    /// any are firing at the current step, which selects the zero-subscriber
    /// fast path. Safe on the main thread before `shutdown()`.
    const std::vector<AsyncComponentBase*>& subscribers() const noexcept {
        return subscribers_;
    }

    /// @brief Hand an (observer, slot) pair to the worker queue.
    /// @param sub Subscriber whose `on_snapshot` will be called.
    /// @param slot_idx Slot that has been published.
    void submit(AsyncComponentBase* sub, int slot_idx);

    /// @brief Total number of dropped (overwritten) snapshots since `init()`.
    int n_dropped() const noexcept {
        return n_dropped_.load(std::memory_order_relaxed);
    }

    /// @brief Current ring capacity (constant global depth K after `init()`).
    int ring_capacity() const noexcept {
        return ring_capacity_.load(std::memory_order_relaxed);
    }

#ifdef ATV_HAS_MPI
    /// @brief MPI sub-communicator for observer collectives. Dupped from
    /// `Config::parent_comm`.
    MPI_Comm analysis_comm() const noexcept { return analysis_comm_; }
#endif

  private:
    struct Job {
        AsyncComponentBase* sub;
        int slot_idx;
    };

    void worker_loop(int worker_idx);
    int acquire_slot_locked();                  ///< implementation helper
    void apply_worker_affinity(int worker_idx); ///< Linux-only; no-op elsewhere

    bool initialized_ = false;
    bool affinity_hint_ = false;

    // Ring storage -- contiguous vector at constant depth K (sized in init()).
    std::vector<FrameSnapshot> ring_;
    std::vector<bool>
        slot_busy_; ///< slot is checked-out (publisher writing or worker reading)
    std::atomic<int> ring_capacity_{0};
    std::atomic<int> next_slot_{0};
    std::atomic<int> n_dropped_{0};
    std::atomic<int> last_published_slot_{-1};

    // Worker pool.
    std::vector<std::thread> workers_;
    std::deque<Job> queue_;
    std::mutex mtx_;
    std::condition_variable cv_work_;
    std::atomic<bool> stop_{false};
    std::atomic<int> in_flight_{0}; ///< jobs dequeued but not finished

    std::vector<AsyncComponentBase*> subscribers_;

#ifdef ATV_HAS_MPI
    MPI_Comm analysis_comm_ = MPI_COMM_NULL;
    bool owns_analysis_comm_ = false;
#endif
};

} // namespace cpu
} // namespace ATV
