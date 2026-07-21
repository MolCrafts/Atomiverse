/// @file cuda/async/record_base.cu
/// @brief `cuda::detail::HostWorker` implementation + sync-guard counter
///        out-of-line definitions.
///
/// Houses the std::thread / mutex / condition_variable plumbing for the
/// single-thread host worker used by `cuda::AsyncRecordBase<Snapshot>`.
/// Lives in one translation unit so the worker plumbing isn't recompiled
/// into every recorder that instantiates the template.
///
/// Also defines the test-only sync counter accessors declared in
/// `include/atomiverse/cuda/async/sync_guard.cuh`. Under `ATV_BUILD_TESTS`
/// the counter is a file-local `std::atomic<unsigned long long>` ; under
/// a release build the accessors are not emitted (the wrapper does not
/// reference them).

#include "atomiverse/cuda/async/record_base.cuh"
#include "atomiverse/cuda/async/sync_guard.cuh"

#include <atomic>

namespace ATV {
namespace cuda {
namespace detail {

// ── HostWorker ───────────────────────────────────────────────────────

HostWorker::HostWorker() = default;

HostWorker::~HostWorker() { join(); }

void HostWorker::start() {
    if (started_)
        return;
    started_ = true;
    stop_.store(false, std::memory_order_relaxed);
    joined_ = false;
    thread_ = std::thread(&HostWorker::loop, this);
}

void HostWorker::submit(std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push_back(std::move(fn));
    }
    cv_.notify_one();
}

void HostWorker::join() {
    if (joined_ || !started_)
        return;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
    joined_ = true;
    started_ = false;
}

void HostWorker::loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] {
                return !queue_.empty() || stop_.load(std::memory_order_acquire);
            });
            if (!queue_.empty()) {
                job = std::move(queue_.front());
                queue_.pop_front();
            } else if (stop_.load(std::memory_order_acquire)) {
                // Drained + stop requested → exit.
                return;
            }
        }
        if (job)
            job();
    }
}

// ── Sync counter (ATV_BUILD_TESTS only) ─────────────────────────────

#ifdef ATV_BUILD_TESTS

namespace {
std::atomic<unsigned long long> g_sync_count{0};
}

unsigned long long sync_count() noexcept {
    return g_sync_count.load(std::memory_order_acquire);
}

void reset_sync_count() noexcept {
    g_sync_count.store(0, std::memory_order_release);
}

void inc_sync_count() noexcept {
    g_sync_count.fetch_add(1, std::memory_order_acq_rel);
}

#endif // ATV_BUILD_TESTS

} // namespace detail
} // namespace cuda
} // namespace ATV
