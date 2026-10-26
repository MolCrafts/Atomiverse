#include "atomiverse/core/gto.h"
#include "atomiverse/cuda/device/error.cuh"
#include "core/gto_cases.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {
// Independent scalar map; consecutive lanes read consecutive inputs and write
// consecutive outputs. No reduction or shared scratch is required.
__global__ void evaluate_boys(const int* order, const double* x, double* value,
                              int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        value[i] = ATV::boys_fn(order[i], x[i]);
}

// One lane owns one independent input sequence; the bounded local sequence
// has no inter-lane communication. Only the requested component is published
// contiguously, so the test covers downward and upward batch paths.
__global__ void evaluate_boys_sequence(const int* order, const double* x,
                                       double* value, int count, int extra) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double sequence[32];
        ATV::boys_sequence(order[i] + extra, x[i], sequence);
        value[i] = sequence[order[i]];
    }
}
} // namespace
TEST(CudaGaussianMath, BoysIntegralAcrossCancellationAndRecurrenceBoundary) {
    const int count = sizeof(gto_test::cases) / sizeof(gto_test::cases[0]);
    std::vector<int> order(count);
    std::vector<double> x(count), value(count);
    for (int i = 0; i < count; ++i) {
        order[i] = gto_test::cases[i].order;
        x[i] = gto_test::cases[i].argument;
    }
    int* d_order = nullptr;
    double *d_x = nullptr, *d_value = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&d_order, count * sizeof(int)));
    ATV_CUDA_CHECK(cudaMalloc(&d_x, count * sizeof(double)));
    ATV_CUDA_CHECK(cudaMalloc(&d_value, count * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemcpy(d_order, order.data(), count * sizeof(int),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_x, x.data(), count * sizeof(double),
                              cudaMemcpyHostToDevice));
    evaluate_boys<<<(count + 127) / 128, 128>>>(d_order, d_x, d_value, count);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpy(value.data(), d_value, count * sizeof(double),
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
        const auto& c = gto_test::cases[i];
        EXPECT_NEAR(value[i], c.expected, 2e-14 * std::abs(c.expected) + 1e-300)
            << "order=" << c.order << " x=" << c.argument;
    }
    for (int extra : {0, 1, 6}) {
        evaluate_boys_sequence<<<(count + 127) / 128, 128>>>(
            d_order, d_x, d_value, count, extra);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaMemcpy(value.data(), d_value, count * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i) {
            const auto& c = gto_test::cases[i];
            ASSERT_LT(c.order + extra, 32);
            EXPECT_NEAR(value[i], c.expected,
                        2e-14 * std::abs(c.expected) + 1e-300)
                << "order=" << c.order << " x=" << c.argument
                << "max_order=" << c.order + extra;
        }
    }
    // Independent analytic tiny-x limit, which upward recurrence from F_0
    // cannot obtain by subtracting two rounded values near one.
    const int tiny_order = 1;
    const double tiny_x = 1e-20;
    ATV_CUDA_CHECK(
        cudaMemcpy(d_order, &tiny_order, sizeof(int), cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(
        cudaMemcpy(d_x, &tiny_x, sizeof(double), cudaMemcpyHostToDevice));
    evaluate_boys_sequence<<<1, 128>>>(d_order, d_x, d_value, 1, 0);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpy(value.data(), d_value, sizeof(double),
                              cudaMemcpyDeviceToHost));
    EXPECT_DOUBLE_EQ(value[0], 1.0 / 3.0);
    cudaFree(d_order);
    cudaFree(d_x);
    cudaFree(d_value);
}
