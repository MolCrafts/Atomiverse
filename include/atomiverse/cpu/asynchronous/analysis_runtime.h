/// @file cpu/asynchronous/analysis_runtime.h
/// @brief cpu::asynchronous::AnalysisRuntime — owns the async-analysis executor
///        and its lifecycle (start / drain / teardown).
///
/// backend-mirror-05-dissolve pulled this out of the former CPU execution
/// engine: the deferred-analysis pipeline is an asynchronous concern, owned here.
#pragma once

#include <memory>

namespace ATV {
namespace cpu {

class Context;
class AnalysisExecutor;

namespace asynchronous {

/// @brief Owns the async-analysis executor and drives its lifecycle.
class AnalysisRuntime {
  public:
    AnalysisRuntime();
    ~AnalysisRuntime();
    AnalysisRuntime(const AnalysisRuntime&) = delete;
    AnalysisRuntime& operator=(const AnalysisRuntime&) = delete;
    AnalysisRuntime(AnalysisRuntime&&) noexcept;
    AnalysisRuntime& operator=(AnalysisRuntime&&) noexcept;

    /// @brief Start the executor (creates + inits it, publishing on the context)
    ///        if none is running.
    void start(Context& ctx);

    /// @brief Block until every queued/in-flight job drains; keep workers alive.
    void drain();

    /// @brief Drain + join workers (idempotent) before components are destroyed.
    void shutdown_workers();

    /// @brief Full teardown: shut down, release the executor, clear the context.
    void teardown(Context& ctx);

  private:
    std::unique_ptr<AnalysisExecutor> executor_;
};

} // namespace asynchronous
} // namespace cpu
} // namespace ATV
