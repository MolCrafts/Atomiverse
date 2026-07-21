/// @file tests/cuda/pw/test_fft.cu
/// @brief Round-trip tests for cuda::pw::internal::FFTPlan3D (cuFFT). Mirrors
///        the CPU tests/cpu/pw/test_fft.cpp and adds a c2c round trip.
#include "atomiverse/cuda/pw/internal/fft.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cuda::pw::internal::FFTPlan3D;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}
} // namespace

// r2c -> c2r round trip recovers the input up to the unnormalized nr factor.
TEST(CudaPWFFT, R2CC2RRoundTrip) {
    int nx = 12, ny = 12, nz = 12;
    FFTPlan3D plan(nx, ny, nz);
    int nr = plan.nr();
    int nc = plan.nc();
    ASSERT_EQ(nr, nx * ny * nz);
    ASSERT_EQ(nc, (nx / 2 + 1) * ny * nz);

    std::vector<double> orig(nr);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < nr; ++i)
        orig[i] = dist(rng);

    auto* d_real = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_cplx = static_cast<cufftDoubleComplex*>(
        dmalloc(nc * sizeof(cufftDoubleComplex)));
    auto* d_back = static_cast<double*>(dmalloc(nr * sizeof(double)));

    ASSERT_EQ(cudaMemcpy(d_real, orig.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    plan.r2c(d_real, d_cplx);
    plan.c2r(d_cplx, d_back);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> back(nr);
    ASSERT_EQ(cudaMemcpy(back.data(), d_back, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    const double norm = 1.0 / static_cast<double>(nr);
    double max_err = 0;
    for (int i = 0; i < nr; ++i)
        max_err = std::max(max_err, std::abs(back[i] * norm - orig[i]));
    EXPECT_LT(max_err, 1e-10);

    cudaFree(d_real);
    cudaFree(d_cplx);
    cudaFree(d_back);
}

// c2c forward -> backward round trip recovers the input up to the nr factor.
TEST(CudaPWFFT, C2CRoundTrip) {
    int nx = 8, ny = 10, nz = 6;
    FFTPlan3D plan(nx, ny, nz);
    int nr = plan.nr();

    std::vector<cufftDoubleComplex> orig(nr);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < nr; ++i) {
        orig[i].x = dist(rng);
        orig[i].y = dist(rng);
    }

    auto* d = static_cast<cufftDoubleComplex*>(
        dmalloc(nr * sizeof(cufftDoubleComplex)));
    ASSERT_EQ(cudaMemcpy(d, orig.data(), nr * sizeof(cufftDoubleComplex),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    plan.c2c_fwd(d);
    plan.c2c_bwd(d);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> back(nr);
    ASSERT_EQ(cudaMemcpy(back.data(), d, nr * sizeof(cufftDoubleComplex),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    const double norm = 1.0 / static_cast<double>(nr);
    double max_err = 0;
    for (int i = 0; i < nr; ++i) {
        max_err = std::max(max_err, std::abs(back[i].x * norm - orig[i].x));
        max_err = std::max(max_err, std::abs(back[i].y * norm - orig[i].y));
    }
    EXPECT_LT(max_err, 1e-10);
    cudaFree(d);
}
