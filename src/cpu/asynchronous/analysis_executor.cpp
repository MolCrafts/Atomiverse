/// @file cpu/asynchronous/analysis_executor.cpp
/// @brief `cpu::AnalysisExecutor` implementation.

#include "atomiverse/cpu/asynchronous/analysis_executor.h"

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/asynchronous/async_component_base.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <utility>

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
    slot_readers_.assign(static_cast<std::size_t>(ring_size), 0);
    last_published_slot_.store(-1, std::memory_order_relaxed);
    ring_capacity_.store(ring_size, std::memory_order_relaxed);
    next_slot_.store(0, std::memory_order_relaxed);
    n_dropped_.store(0, std::memory_order_relaxed);
    in_flight_.store(0, std::memory_order_relaxed);
    stop_.store(false, std::memory_order_relaxed);
    queue_.clear();
    active_subscribers_.clear();
    failure_ = nullptr;
    subscribers_.clear();

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
    auto failure = std::exchange(failure_, nullptr);
    lk.unlock();
    if (failure)
        std::rethrow_exception(failure);
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
        if (sub) {
            try {
                sub->finalize();
            } catch (...) {
                if (!failure_)
                    failure_ = std::current_exception();
            }
        }
    }
    subscribers_.clear();
    if (failure_)
        ATV_WARN("analysis executor discarded an unconsumed callback error "
                 "during shutdown; use drain_analysis() to receive errors");

    ring_.clear();
    slot_busy_.clear();
    slot_readers_.clear();
    ring_capacity_.store(0, std::memory_order_relaxed);
    queue_.clear();
    initialized_ = false;
}

int AnalysisExecutor::acquire_slot() {
    std::unique_lock<std::mutex> lk(mtx_);
    return acquire_slot_locked();
}

int AnalysisExecutor::acquire_slot_locked() {
    // The previous publication stays pinned while subscribers enqueue their
    // jobs. Only the next publisher releases that pin; each queued/running
    // reader keeps its own reference until its callback has finished.
    const int previous = last_published_slot_.exchange(-1);
    if (previous >= 0)
        slot_busy_[static_cast<std::size_t>(previous)] = false;
    const int cap = ring_capacity_.load(std::memory_order_relaxed);
    const int start = next_slot_.load(std::memory_order_relaxed);
    for (int offset = 0; offset < cap; ++offset) {
        const int idx = (start + offset) % cap;
        if (slot_busy_[idx] || slot_readers_[idx] != 0)
            continue;
        slot_busy_[idx] = true;
        next_slot_.store((idx + 1) % cap, std::memory_order_relaxed);
        return idx;
    }
    n_dropped_.fetch_add(1, std::memory_order_relaxed);
    return -1; // Drop this capture, never overwrite storage held by a reader.
}

void AnalysisExecutor::abandon_slot(int idx) {
    std::lock_guard<std::mutex> lk(mtx_);
    slot_busy_[static_cast<std::size_t>(idx)] = false;
}

FrameSnapshot& AnalysisExecutor::slot(int idx) {
    return ring_[static_cast<std::size_t>(idx)];
}

void AnalysisExecutor::publish(int idx, BigInt step) {
    ring_[static_cast<std::size_t>(idx)].step = step;
    last_published_slot_.store(idx, std::memory_order_release);
    // Publication pin remains until the next acquire; jobs hold reader refs.
}

void AnalysisExecutor::prealloc_slots(int n_atoms, CaptureMask mask) {
    quiesce();
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

void AnalysisExecutor::unregister_subscriber(AsyncComponentBase* sub) {
    // Destruction/removal must drain without throwing from a destructor path.
    // Keep any callback error for the next explicit quiesce().
    {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_work_.wait(lk, [this] {
            return queue_.empty()
                   && in_flight_.load(std::memory_order_relaxed) == 0;
        });
        auto it = std::find(subscribers_.begin(), subscribers_.end(), sub);
        if (it == subscribers_.end())
            return;
        subscribers_.erase(it);
    }
    try {
        sub->finalize();
    } catch (...) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!failure_)
            failure_ = std::current_exception();
    }
}

void AnalysisExecutor::submit(AsyncComponentBase* sub, int slot_idx) {
    if (workers_.empty())
        return; // Ring-only mode does not queue unserviceable jobs.
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back({sub, slot_idx});
        ++slot_readers_[static_cast<std::size_t>(slot_idx)];
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
            const auto runnable = [this] {
                return std::find_if(
                    queue_.begin(), queue_.end(), [this](const Job& j) {
                        return !active_subscribers_.contains(j.sub);
                    });
            };
            cv_work_.wait(lk, [&] {
                return stop_.load(std::memory_order_relaxed)
                       || runnable() != queue_.end();
            });
            auto next = runnable();
            if (next == queue_.end()) {
                if (stop_.load(std::memory_order_relaxed))
                    return;
                continue;
            }
            job = *next;
            queue_.erase(next);
            active_subscribers_.insert(job.sub);
        }

        try {
            if (job.sub)
                job.sub->on_snapshot(
                    ring_[static_cast<std::size_t>(job.slot_idx)]);
        } catch (...) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!failure_)
                failure_ = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            active_subscribers_.erase(job.sub);
            --slot_readers_[static_cast<std::size_t>(job.slot_idx)];
            in_flight_.fetch_sub(1, std::memory_order_relaxed);
        }
        cv_work_.notify_all(); // wake up shutdown-waiters too
    }
}

} // namespace cpu
} // namespace ATV
