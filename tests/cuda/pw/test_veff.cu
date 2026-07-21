/// @file tests/cuda/pw/test_veff.cu
/// @brief Verifies cuda::pw::internal::veff_assemble_g against the host
///        elementwise sum V_eff(G) = V_loc(G) + V_H(G) + V_xc(G).
#include "atomiverse/cuda/pw/internal/veff.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cuda::pw::internal::veff_assemble_g;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}
} // namespace

TEST(CudaPWVeff, MatchesHostSum) {
    const int n_g = 3333;
    std::vector<double> vloc(n_g);
    std::vector<cufftDoubleComplex> vh(n_g), vxc(n_g);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> d(-2, 2);
    for (int i = 0; i < n_g; ++i) {
        vloc[i] = d(rng);
        vh[i] = {d(rng), d(rng)};
        vxc[i] = {d(rng), d(rng)};
    }

    auto* d_vloc = static_cast<double*>(dmalloc(n_g * sizeof(double)));
    auto* d_vh = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_vxc = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_veff = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));

    cudaMemcpy(d_vloc, vloc.data(), n_g * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vh, vh.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vxc, vxc.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);

    veff_assemble_g(d_vloc, d_vh, d_vxc, d_veff, n_g);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> veff(n_g);
    cudaMemcpy(veff.data(), d_veff, n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    double max_err = 0;
    for (int i = 0; i < n_g; ++i) {
        double rx = vloc[i] + vh[i].x + vxc[i].x;
        double ry = vh[i].y + vxc[i].y;
        max_err = std::max(max_err, std::abs(veff[i].x - rx));
        max_err = std::max(max_err, std::abs(veff[i].y - ry));
    }
    EXPECT_LT(max_err, 1e-14);

    cudaFree(d_vloc);
    cudaFree(d_vh);
    cudaFree(d_vxc);
    cudaFree(d_veff);
}

// In-place aliasing onto the V_xc buffer.
TEST(CudaPWVeff, InPlaceAliasing) {
    const int n_g = 777;
    std::vector<double> vloc(n_g);
    std::vector<cufftDoubleComplex> vh(n_g), vxc(n_g);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> d(-1, 1);
    for (int i = 0; i < n_g; ++i) {
        vloc[i] = d(rng);
        vh[i] = {d(rng), d(rng)};
        vxc[i] = {d(rng), d(rng)};
    }
    auto* d_vloc = static_cast<double*>(dmalloc(n_g * sizeof(double)));
    auto* d_vh = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_vxc = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    cudaMemcpy(d_vloc, vloc.data(), n_g * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vh, vh.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vxc, vxc.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);

    veff_assemble_g(d_vloc, d_vh, d_vxc, d_vxc, n_g); // output aliases V_xc
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> veff(n_g);
    cudaMemcpy(veff.data(), d_vxc, n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    double max_err = 0;
    for (int i = 0; i < n_g; ++i) {
        double rx = vloc[i] + vh[i].x + vxc[i].x;
        double ry = vh[i].y + vxc[i].y;
        max_err = std::max(max_err, std::abs(veff[i].x - rx));
        max_err = std::max(max_err, std::abs(veff[i].y - ry));
    }
    EXPECT_LT(max_err, 1e-14);

    cudaFree(d_vloc);
    cudaFree(d_vh);
    cudaFree(d_vxc);
}
