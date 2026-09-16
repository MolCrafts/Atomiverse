/// @file cpu/asynchronous/analysis_runtime.cpp
/// @brief cpu::asynchronous::AnalysisRuntime implementation (moved out of the
///        former CPU execution engine by backend-mirror-05-dissolve).

#include "atomiverse/cpu/asynchronous/analysis_runtime.h"

#include "atomiverse/cpu/asynchronous/analysis_executor.h"
#include "atomiverse/cpu/component/context.h"

namespace ATV {
namespace cpu {
namespace asynchronous {

AnalysisRuntime::AnalysisRuntime() = default;
AnalysisRuntime::~AnalysisRuntime() = default;
AnalysisRuntime::AnalysisRuntime(AnalysisRuntime&&) noexcept = default;
AnalysisRuntime&
AnalysisRuntime::operator=(AnalysisRuntime&&) noexcept = default;

void AnalysisRuntime::start(Context& ctx) {
    if (!executor_) {
        executor_ = std::make_unique<AnalysisExecutor>();
        AnalysisExecutor::Config cfg;
        cfg.n_threads = 1;
        cfg.ring_size = 3; // global ring depth K
        executor_->init(cfg, ctx);
    }
}

void AnalysisRuntime::drain() {
    if (executor_)
        executor_->quiesce();
}

void AnalysisRuntime::shutdown_workers() {
    // Drain and join analysis workers so any in-flight `on_snapshot` completes
    // against live subscribers, and `finalize()` runs on the main thread before
    // any component enters its `destroy()` path.
    if (executor_)
        executor_->shutdown();
}

void AnalysisRuntime::teardown(Context& ctx) {
    if (executor_) {
        executor_->shutdown(); // idempotent
        executor_.reset();
    }
    ctx.analysis.executor = nullptr;
}

} // namespace asynchronous
} // namespace cpu
} // namespace ATV
