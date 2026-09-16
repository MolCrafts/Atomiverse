/// @file tests/cuda/planewave/test_poisson.cu
/// @brief Verifies cuda::pw::internal::poisson_hartree against the closed-form
///        CPU formula V_H(G) = 4π ρ(G)/|G|² (PW_Poisson::launch).
#include "atomiverse/cuda/planewave/internal/poisson.cuh"
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cuda::pw::internal::poisson_hartree;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

// Host reference identical to cpu::pw::PW_Poisson::launch.
std::complex<double> host_poisson(std::complex<double> rho, double g2) {
    constexpr double four_pi = 4.0 * 3.141592653589793;
    if (g2 < 1e-16)
        return {0.0, 0.0};
    return four_pi * rho / g2;
}
} // namespace

TEST(CudaPWPoisson, MatchesCpuFormula) {
    const int n_g = 4097; // crosses several blocks + a partial tail
    std::vector<cufftDoubleComplex> rho(n_g);
    std::vector<double> g2(n_g);
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::uniform_real_distribution<double> gdist(0.05, 50.0);
    for (int i = 0; i < n_g; ++i) {
        rho[i].x = dist(rng);
        rho[i].y = dist(rng);
        g2[i] = gdist(rng);
    }
    // Force a few G=0-like components to exercise the neutralizing branch.
    g2[0] = 0.0;
    g2[100] = 1e-20;
    g2[n_g - 1] = 0.0;

    auto* d_rho = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_g2 = static_cast<double*>(dmalloc(n_g * sizeof(double)));
    auto* d_vh = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));

    ASSERT_EQ(cudaMemcpy(d_rho, rho.data(), n_g * sizeof(cufftDoubleComplex),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_g2, g2.data(), n_g * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    poisson_hartree(d_rho, d_g2, d_vh, n_g);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> vh(n_g);
    ASSERT_EQ(cudaMemcpy(vh.data(), d_vh, n_g * sizeof(cufftDoubleComplex),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    double max_err = 0;
    for (int i = 0; i < n_g; ++i) {
        auto ref = host_poisson({rho[i].x, rho[i].y}, g2[i]);
        max_err = std::max(max_err, std::abs(vh[i].x - ref.real()));
        max_err = std::max(max_err, std::abs(vh[i].y - ref.imag()));
    }
    EXPECT_LT(max_err, 1e-12);

    cudaFree(d_rho);
    cudaFree(d_g2);
    cudaFree(d_vh);
}

// In-place aliasing (d_V_H_G == d_rho_G) must give the same result.
TEST(CudaPWPoisson, InPlaceAliasing) {
    const int n_g = 1000;
    std::vector<cufftDoubleComplex> rho(n_g);
    std::vector<double> g2(n_g);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::uniform_real_distribution<double> gdist(0.1, 10.0);
    for (int i = 0; i < n_g; ++i) {
        rho[i].x = dist(rng);
        rho[i].y = dist(rng);
        g2[i] = gdist(rng);
    }
    g2[0] = 0.0;

    auto* d_rho = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_g2 = static_cast<double*>(dmalloc(n_g * sizeof(double)));
    ASSERT_EQ(cudaMemcpy(d_rho, rho.data(), n_g * sizeof(cufftDoubleComplex),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_g2, g2.data(), n_g * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    poisson_hartree(d_rho, d_g2, d_rho, n_g); // alias output onto input
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> vh(n_g);
    ASSERT_EQ(cudaMemcpy(vh.data(), d_rho, n_g * sizeof(cufftDoubleComplex),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    double max_err = 0;
    for (int i = 0; i < n_g; ++i) {
        auto ref = host_poisson({rho[i].x, rho[i].y}, g2[i]);
        max_err = std::max(max_err, std::abs(vh[i].x - ref.real()));
        max_err = std::max(max_err, std::abs(vh[i].y - ref.imag()));
    }
    EXPECT_LT(max_err, 1e-12);

    cudaFree(d_rho);
    cudaFree(d_g2);
}
