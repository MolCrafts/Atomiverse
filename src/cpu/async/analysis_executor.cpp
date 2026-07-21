/// @file src/cpu/analysis_executor.cpp
/// @brief `cpu::AnalysisExecutor` implementation.

#include "atomiverse/cpu/async/analysis_executor.h"

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/async/async_component_base.h"

#include <algorithm>
#include <cassert>
#include <chrono>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace ATV {
namespace cpu {

AnalysisExecutor::~AnalysisExecutor() {
    if (initialized_) {
        shutdown();
    }
}

void AnalysisExecutor::init(const Config& cfg, Context& ctx) {
    if (initialized_)
        return;

    affinity_hint_ = cfg.affinity_hint;

    const int ring_size = std::max(1, cfg.ring_size);
    ring_.assign(static_cast<std::size_t>(ring_size), FrameSnapshot{});
    slot_busy_.assign(static_cast<std::size_t>(ring_size), false);
    ring_capacity_.store(ring_size, std::memory_order_relaxed);
    next_slot_.store(0, std::memory_order_relaxed);
    n_dropped_.store(0, std::memory_order_relaxed);
    in_flight_.store(0, std::memory_order_relaxed);
    stop_.store(false, std::memory_order_relaxed);
    queue_.clear();
    subscribers_.clear();

#ifdef ATV_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        // NB: we deliberately *alias* the parent comm rather than
        // MPI_Comm_dup-ing it. OpenMPI 5.0.x on macOS corrupts its internal
        // communicator state when user code runs multiple comm_dup /
        // cart_create cycles in the same process (observed across
        // GoogleTest cases). Aliasing means analysis-thread MPI collectives
        // share the world comm; that is safe as long as the analysis
        // worker does not run concurrently with a collective on the main
        // thread -- Publish orders them strictly.
        analysis_comm_ = cfg.parent_comm;
        owns_analysis_comm_ = false;
    } else {
        analysis_comm_ = MPI_COMM_NULL;
        owns_analysis_comm_ = false;
    }
#endif

    const int n_workers = std::max(0, cfg.n_threads);
    workers_.reserve(static_cast<std::size_t>(n_workers));
    for (int i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this, i] { this->worker_loop(i); });
    }

    ctx.analysis.executor = this;
    initialized_ = true;

    ATV_DEBUG("analysis_executor initialized: {} workers, ring {}", n_workers,
              ring_size);
}

void AnalysisExecutor::quiesce() {
    if (!initialized_)
        return;
    // Wait for the queue to empty and all in-flight jobs to finish.
    // Workers stay alive — this is the mid-run barrier, not teardown.
    std::unique_lock<std::mutex> lk(mtx_);
    cv_work_.wait(lk, [this] {
        return queue_.empty()
               && in_flight_.load(std::memory_order_relaxed) == 0;
    });
}

void AnalysisExecutor::shutdown() {
    if (!initialized_)
        return;

    // Wait for in-flight submissions to drain, then signal stop.
    {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_work_.wait(lk, [this] {
            return queue_.empty()
                   && in_flight_.load(std::memory_order_relaxed) == 0;
        });
        stop_.store(true, std::memory_order_relaxed);
    }
    cv_work_.notify_all();

    for (auto& t : workers_) {
        if (t.joinable())
            t.join();
    }
    workers_.clear();

    // Subscriber finalize() after all workers have joined, on the main thread,
    // exactly once per subscriber.
    for (auto* sub : subscribers_) {
        if (sub)
            sub->finalize();
    }
    subscribers_.clear();

#ifdef ATV_HAS_MPI
    if (owns_analysis_comm_ && analysis_comm_ != MPI_COMM_NULL) {
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Comm_free(&analysis_comm_);
        }
        owns_analysis_comm_ = false;
    }
    analysis_comm_ = MPI_COMM_NULL;
#endif

    ring_.clear();
    slot_busy_.clear();
    ring_capacity_.store(0, std::memory_order_relaxed);
    queue_.clear();
    initialized_ = false;
}

int AnalysisExecutor::acquire_slot() {
    std::unique_lock<std::mutex> lk(mtx_);
    return acquire_slot_locked();
}

int AnalysisExecutor::acquire_slot_locked() {
    // Single drop_oldest policy: round-robin to the next slot at constant
    // ring depth K. If that slot is still in flight, overwrite it in place
    // and count the drop -- never blocks, never allocates.
    const int cap = ring_capacity_.load(std::memory_order_relaxed);
    const int idx = next_slot_.load(std::memory_order_relaxed);

    if (slot_busy_[static_cast<std::size_t>(idx)])
        n_dropped_.fetch_add(1, std::memory_order_relaxed);

    slot_busy_[static_cast<std::size_t>(idx)] = true;
    next_slot_.store((idx + 1) % cap, std::memory_order_relaxed);
    return idx;
}

FrameSnapshot& AnalysisExecutor::slot(int idx) {
    return ring_[static_cast<std::size_t>(idx)];
}

void AnalysisExecutor::publish(int idx, int step) {
    ring_[static_cast<std::size_t>(idx)].step = step;
    last_published_slot_.store(idx, std::memory_order_release);
    // Slot remains marked busy until the last consumer finishes; but with
    // a single publish-to-submit-to-worker pipeline, the slot is released
    // inside the worker loop after `on_snapshot` completes.
}

void AnalysisExecutor::prealloc_slots(int n_atoms, CaptureMask mask) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& snap : ring_) {
        if (mask.positions) {
            if (static_cast<int>(snap.x.size()) < n_atoms)
                snap.x.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.y.size()) < n_atoms)
                snap.y.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.z.size()) < n_atoms)
                snap.z.resize(static_cast<std::size_t>(n_atoms));
        }
        if (mask.velocities) {
            if (static_cast<int>(snap.vx.size()) < n_atoms)
                snap.vx.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.vy.size()) < n_atoms)
                snap.vy.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.vz.size()) < n_atoms)
                snap.vz.resize(static_cast<std::size_t>(n_atoms));
        }
        if (mask.forces) {
            if (static_cast<int>(snap.fx.size()) < n_atoms)
                snap.fx.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.fy.size()) < n_atoms)
                snap.fy.resize(static_cast<std::size_t>(n_atoms));
            if (static_cast<int>(snap.fz.size()) < n_atoms)
                snap.fz.resize(static_cast<std::size_t>(n_atoms));
        }
        if (mask.types && static_cast<int>(snap.type.size()) < n_atoms) {
            snap.type.resize(static_cast<std::size_t>(n_atoms));
        }
        if (mask.tags && static_cast<int>(snap.tag.size()) < n_atoms) {
            snap.tag.resize(static_cast<std::size_t>(n_atoms));
        }
    }
}

void AnalysisExecutor::register_subscriber(AsyncComponentBase* sub) {
    std::lock_guard<std::mutex> lk(mtx_);
    subscribers_.push_back(sub);
}

void AnalysisExecutor::submit(AsyncComponentBase* sub, int slot_idx) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back({sub, slot_idx});
        in_flight_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_work_.notify_one();
}

void AnalysisExecutor::apply_worker_affinity(int worker_idx) {
    if (!affinity_hint_)
        return;
#ifdef __linux__
    const long n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpus <= 0)
        return;
    // Pin worker N to the highest-numbered cores first, on the assumption
    // that the OMP team driving the hot path has the lower cores. Modular
    // wrap so configurations with more workers than cores still succeed
    // (multiple workers share a core but stay off the OMP team's range).
    const int target = static_cast<int>(n_cpus - 1 - (worker_idx % n_cpus));
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (target >= 0)
        CPU_SET(target, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    ATV_DEBUG("analysis_executor: worker {} pinned to CPU {}", worker_idx,
              target);
#else
    (void)worker_idx; // hint is a no-op outside Linux
#endif
}

void AnalysisExecutor::worker_loop(int worker_idx) {
    apply_worker_affinity(worker_idx);
    for (;;) {
        Job job{};
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [this] {
                return stop_.load(std::memory_order_relaxed) || !queue_.empty();
            });
            if (queue_.empty()) {
                if (stop_.load(std::memory_order_relaxed))
                    return;
                continue;
            }
            job = queue_.front();
            queue_.pop_front();
        }

        if (job.sub) {
            job.sub->on_snapshot(ring_[static_cast<std::size_t>(job.slot_idx)]);
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            slot_busy_[static_cast<std::size_t>(job.slot_idx)] = false;
            in_flight_.fetch_sub(1, std::memory_order_relaxed);
        }
        cv_work_.notify_all(); // wake up shutdown-waiters too
    }
}

} // namespace cpu
} // namespace ATV
