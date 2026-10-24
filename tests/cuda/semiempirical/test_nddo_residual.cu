/// @brief Single density-reduction tests; no SCF loop.
#include "atomiverse/cuda/semiempirical/internal/nddo_workspace.cuh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
class CudaNDDOResidual : public ::testing::Test {
  protected:
    cudaStream_t stream = nullptr;
    double* data = nullptr;
    void* scratch = nullptr;
    std::size_t bytes = 0;
    void SetUp() override {
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&data, 19 * sizeof(double)), cudaSuccess);
        bytes = ATV::cuda::semiempirical::internal::
            density_residual_workspace_bytes(3);
        ASSERT_EQ(cudaMalloc(&scratch, bytes), cudaSuccess);
    }
    void TearDown() override {
        cudaStreamSynchronize(stream);
        cudaFree(scratch);
        cudaFree(data);
        cudaStreamDestroy(stream);
    }
    double reduce(const std::array<double, 9>& a,
                  const std::array<double, 9>& b) {
        EXPECT_EQ(cudaMemcpyAsync(data, a.data(), sizeof(a),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(data + 9, b.data(), sizeof(b),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ATV::cuda::semiempirical::internal::density_residual(
            3, data, data + 9, data + 18, scratch, bytes, stream);
        double result = -1;
        EXPECT_EQ(cudaMemcpyAsync(&result, data + 18, sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return result;
    }
};
TEST_F(CudaNDDOResidual, MaximumComponentAndReusableScratch) {
    const std::array<double, 9> a{1, .2, -.3, .2, 2, .5, -.3, .5, 1};
    auto b = a;
    b[2] -= .125;
    b[6] -= .125;
    b[4] += .0625;
    EXPECT_DOUBLE_EQ(reduce(a, b), .125);
    EXPECT_DOUBLE_EQ(reduce(b, b), 0);
    EXPECT_DOUBLE_EQ(reduce(b, a), .125);
}
TEST_F(CudaNDDOResidual, NonfiniteValuesCannotBeHiddenByMax) {
    const std::array<double, 9> a{};
    auto b = a;
    b[4] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isinf(reduce(a, b)));
    b[4] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(reduce(a, b)));
    EXPECT_TRUE(std::isinf(reduce(b, b)));
    EXPECT_DOUBLE_EQ(reduce(a, a), 0);
}
} // namespace
