/// @brief Full-precision coordinate capture and asynchronous ring metadata.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace {
class Capture : public ATV::cuda::AsyncObserver {
  public:
    using AsyncBridge::capture_positions;
    std::string name() const override { return "test_capture"; }
    std::vector<std::array<int, 2>> metadata;
    bool fail = false;

  protected:
    void on_build(const ATV::cuda::BuildContext&) override { metadata.clear(); }
    void freeze_slot(const ATV::cuda::LaunchContext&, int) override {}
    void d2h_slot(const ATV::cuda::LaunchContext&, int) override {}
    void on_data_ready(const ATV::cuda::LaunchContext& ctx, int) override {
        if (fail)
            throw std::runtime_error("test observer failure");
        metadata.push_back({ctx.stage, ctx.step});
    }
    void on_destroy(const ATV::cuda::DestroyContext&) override {}
};
} // namespace

TEST(AsyncBridge, CapturePositionsIncludesResidual) {
    ATV::cpu::Frame host(1);
    host.box = ATV::cpu::Box::cubic(1000);
    const double expected[3] = {512.0 + 1e-7, -3.141592653589793, 1.0 / 7};
    host.span(ATV::kAtom_x)[0] = expected[0];
    host.span(ATV::kAtom_y)[0] = expected[1];
    host.span(ATV::kAtom_z)[0] = expected[2];
    auto device = ATV::cpu::to_cuda(host);
    ATV::cuda::motion_t* output = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&output, 3 * sizeof(ATV::cuda::motion_t)));
    Capture::capture_positions(device, output, output + 1, output + 2, nullptr);
    ATV::cuda::motion_t result[3];
    ATV_CUDA_CHECK(
        cudaMemcpy(result, output, sizeof(result), cudaMemcpyDeviceToHost));
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(result[d], expected[d],
                    sizeof(ATV::cuda::motion_t) == 8 ? 2e-13 : 4e-5);
    ATV_CUDA_CHECK(cudaFree(output));
    ATV::cuda::destroy(device);
}

TEST(AsyncBridge, LaunchKeepsMetadataAcrossRingReuse) {
    ATV::cpu::Frame host(0);
    host.box = ATV::cpu::Box::cubic(10);
    auto device = ATV::cpu::to_cuda(host);
    cudaStream_t compute, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&compute));
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    ATV::cuda::Context runtime;
    runtime.stream = compute;
    runtime.defer_stream = defer;
    Capture capture;
    capture.build({runtime, device});
    constexpr int count = 20;
    for (int i = 0; i < count; ++i)
        capture.launch({runtime, device, i % 3, 100 + i});
    capture.destroy({runtime, device});
    ASSERT_EQ(capture.metadata.size(), count);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(capture.metadata[i][0], i % 3);
        EXPECT_EQ(capture.metadata[i][1], 100 + i);
    }
    ATV_CUDA_CHECK(cudaStreamDestroy(compute));
    ATV_CUDA_CHECK(cudaStreamDestroy(defer));
    ATV::cuda::destroy(device);
}

TEST(AsyncBridge, CallbackFailureRethrowsAfterDrain) {
    ATV::cpu::Frame host(0);
    host.box = ATV::cpu::Box::cubic(10);
    auto device = ATV::cpu::to_cuda(host);
    cudaStream_t compute, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&compute));
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    ATV::cuda::Context runtime;
    runtime.stream = compute;
    runtime.defer_stream = defer;
    Capture capture;
    capture.fail = true;
    capture.build({runtime, device});
    capture.launch({runtime, device, 0, 0});
    EXPECT_THROW(capture.destroy({runtime, device}), std::runtime_error);
    ATV_CUDA_CHECK(cudaStreamDestroy(compute));
    ATV_CUDA_CHECK(cudaStreamDestroy(defer));
    ATV::cuda::destroy(device);
}
