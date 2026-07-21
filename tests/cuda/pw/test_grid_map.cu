/// @file tests/cuda/pw/test_grid_map.cu
/// @brief Verifies cuda::pw::internal scatter/gather between the sparse
///        G-vector list and the dense Hermitian half-complex FFT grid against
///        an identical host fold (the CPU PW_LDA / PW_Veff pad arithmetic).
#include "atomiverse/cuda/pw/internal/grid_map.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using ATV::cuda::pw::internal::gather_halfcomplex_to_g;
using ATV::cuda::pw::internal::scatter_g_to_halfcomplex;

namespace {
void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

// Host fold identical to the CPU pad loops and the device fold_index.
int host_fold(int h, int k, int l, int nx, int ny, int nz, bool& conj) {
    int kx = ((h % nx) + nx) % nx;
    int ky = ((k % ny) + ny) % ny;
    int kz = ((l % nz) + nz) % nz;
    conj = false;
    if (kx > nx / 2) {
        kx = nx - kx;
        ky = (ny - ky) % ny;
        kz = (nz - kz) % nz;
        conj = true;
    }
    return kx + (nx / 2 + 1) * (ky + ny * kz);
}

// Build a collision-free G list (one G per folded index) spanning fold and
// no-fold paths, with random complex values.
struct GList {
    std::vector<int> h, k, l;
    std::vector<cufftDoubleComplex> val;
};
GList make_glist(int nx, int ny, int nz, unsigned seed) {
    GList g;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::vector<char> used(static_cast<size_t>((nx / 2 + 1) * ny * nz), 0);
    for (int h = -nx / 2; h <= nx / 2; ++h)
        for (int k = -ny / 2; k <= ny / 2; ++k)
            for (int l = -nz / 2; l <= nz / 2; ++l) {
                bool conj;
                int idx = host_fold(h, k, l, nx, ny, nz, conj);
                if (used[idx])
                    continue; // keep first → no scatter collision
                used[idx] = 1;
                g.h.push_back(h);
                g.k.push_back(k);
                g.l.push_back(l);
                g.val.push_back({dist(rng), dist(rng)});
            }
    return g;
}
} // namespace

TEST(CudaPWGridMap, ScatterMatchesHostFold) {
    const int nx = 8, ny = 6, nz = 4;
    const int nc = (nx / 2 + 1) * ny * nz;
    GList g = make_glist(nx, ny, nz, 11);
    const int n_g = static_cast<int>(g.h.size());

    // Host scatter reference.
    std::vector<cufftDoubleComplex> ref(nc, {0.0, 0.0});
    for (int ig = 0; ig < n_g; ++ig) {
        bool conj;
        int idx = host_fold(g.h[ig], g.k[ig], g.l[ig], nx, ny, nz, conj);
        ref[idx].x = g.val[ig].x;
        ref[idx].y = conj ? -g.val[ig].y : g.val[ig].y;
    }

    auto* d_val = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_h = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_k = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_l = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_hc = static_cast<cufftDoubleComplex*>(
        dmalloc(nc * sizeof(cufftDoubleComplex)));

    cudaMemcpy(d_val, g.val.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, g.h.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, g.k.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, g.l.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);

    scatter_g_to_halfcomplex(d_val, d_h, d_k, d_l, n_g, d_hc, nx, ny, nz);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> hc(nc);
    cudaMemcpy(hc.data(), d_hc, nc * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    double max_err = 0;
    for (int i = 0; i < nc; ++i) {
        max_err = std::max(max_err, std::abs(hc[i].x - ref[i].x));
        max_err = std::max(max_err, std::abs(hc[i].y - ref[i].y));
    }
    EXPECT_LT(max_err, 1e-14);

    cudaFree(d_val);
    cudaFree(d_h);
    cudaFree(d_k);
    cudaFree(d_l);
    cudaFree(d_hc);
}

TEST(CudaPWGridMap, ScatterGatherRoundTrip) {
    const int nx = 10, ny = 8, nz = 6;
    const int nc = (nx / 2 + 1) * ny * nz;
    GList g = make_glist(nx, ny, nz, 29);
    const int n_g = static_cast<int>(g.h.size());

    auto* d_val = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_out = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_h = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_k = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_l = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_hc = static_cast<cufftDoubleComplex*>(
        dmalloc(nc * sizeof(cufftDoubleComplex)));

    cudaMemcpy(d_val, g.val.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, g.h.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, g.k.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, g.l.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);

    scatter_g_to_halfcomplex(d_val, d_h, d_k, d_l, n_g, d_hc, nx, ny, nz);
    gather_halfcomplex_to_g(d_hc, d_h, d_k, d_l, n_g, d_out, nx, ny, nz, 1.0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> out(n_g);
    cudaMemcpy(out.data(), d_out, n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    double max_err = 0;
    for (int ig = 0; ig < n_g; ++ig) {
        max_err = std::max(max_err, std::abs(out[ig].x - g.val[ig].x));
        max_err = std::max(max_err, std::abs(out[ig].y - g.val[ig].y));
    }
    EXPECT_LT(max_err, 1e-14);

    cudaFree(d_val);
    cudaFree(d_out);
    cudaFree(d_h);
    cudaFree(d_k);
    cudaFree(d_l);
    cudaFree(d_hc);
}

TEST(CudaPWGridMap, GatherAppliesScale) {
    const int nx = 8, ny = 8, nz = 8;
    const int nc = (nx / 2 + 1) * ny * nz;
    GList g = make_glist(nx, ny, nz, 5);
    const int n_g = static_cast<int>(g.h.size());
    const double scale = 1.0 / static_cast<double>(nx * ny * nz); // 1/nr style

    auto* d_val = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_out = static_cast<cufftDoubleComplex*>(
        dmalloc(n_g * sizeof(cufftDoubleComplex)));
    auto* d_h = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_k = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_l = static_cast<int*>(dmalloc(n_g * sizeof(int)));
    auto* d_hc = static_cast<cufftDoubleComplex*>(
        dmalloc(nc * sizeof(cufftDoubleComplex)));

    cudaMemcpy(d_val, g.val.data(), n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, g.h.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, g.k.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, g.l.data(), n_g * sizeof(int), cudaMemcpyHostToDevice);

    scatter_g_to_halfcomplex(d_val, d_h, d_k, d_l, n_g, d_hc, nx, ny, nz);
    gather_halfcomplex_to_g(d_hc, d_h, d_k, d_l, n_g, d_out, nx, ny, nz, scale);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<cufftDoubleComplex> out(n_g);
    cudaMemcpy(out.data(), d_out, n_g * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    double max_err = 0;
    for (int ig = 0; ig < n_g; ++ig) {
        max_err = std::max(max_err, std::abs(out[ig].x - g.val[ig].x * scale));
        max_err = std::max(max_err, std::abs(out[ig].y - g.val[ig].y * scale));
    }
    EXPECT_LT(max_err, 1e-15);

    cudaFree(d_val);
    cudaFree(d_out);
    cudaFree(d_h);
    cudaFree(d_k);
    cudaFree(d_l);
    cudaFree(d_hc);
}
