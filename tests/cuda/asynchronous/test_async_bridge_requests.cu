/// @file tests/cuda/asynchronous/test_async_bridge_requests.cu
/// @brief A single deferred callback preserves launch-time request flags.
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include <gtest/gtest.h>

namespace {
class RequestObserver : public ATV::cuda::AsyncObserver {
  public:
    std::string name() const override { return "request_observer"; }
    ATV::cuda::ExecServices observed{};
    int stage = -1;
    ATV::BigInt step = -1;
    int calls = 0;

  protected:
    void on_build(const ATV::cuda::BuildContext&) override {}
    void freeze_slot(const ATV::cuda::LaunchContext&, int) override {}
    void d2h_slot(const ATV::cuda::LaunchContext&, int) override {}
    void on_destroy(const ATV::cuda::DestroyContext&) override {}
    void on_data_ready(const ATV::cuda::LaunchContext& ctx, int) override {
        observed = {ctx.forces_requested, ctx.stress_requested,
                    ctx.initializing_forces, ctx.trial_evaluation};
        stage = ctx.stage;
        step = ctx.step;
        ++calls;
    }
};
} // namespace

TEST(CudaAsyncBridge, CapturesRequestFlagsForDeferredCallback) {
    ATV::cuda::Backend backend;
    backend.init();
    RequestObserver observer;
    auto& runtime = backend.context();
    auto& frame = backend.frame();
    observer.build({runtime, frame});
    ATV::cuda::LaunchContext launch{runtime, frame, 3, 11};
    // These phase-local flags intentionally differ from the live runtime.
    // The callback must observe this launch even if it runs after a nested
    // dispatch has restored the runtime's default requests.
    launch.forces_requested = false;
    launch.stress_requested = false;
    launch.initializing_forces = true;
    launch.trial_evaluation = true;
    observer.launch(launch);
    backend.sync();
    EXPECT_EQ(observer.calls, 1);
    EXPECT_EQ(observer.stage, 3);
    EXPECT_EQ(observer.step, 11);
    EXPECT_FALSE(observer.observed.forces_requested);
    EXPECT_FALSE(observer.observed.stress_requested);
    EXPECT_TRUE(observer.observed.initializing_forces);
    EXPECT_TRUE(observer.observed.trial_evaluation);
    observer.destroy({runtime, frame});
    backend.shutdown();
}
