/// @file minimizer/cg.cu
/// @brief Polak-Ribière+ CG kernels: device-resident β + fused step.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/minimizer/cg.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace minimizer {

namespace {

/// @brief Polak-Ribière dot products: pr[0] += g·(g−g_prev),
///        pr[1] += g_prev·g_prev, with g = −F.
__global__ void pr_dots(const real_t* __restrict__ fx,
                        const real_t* __restrict__ fy,
                        const real_t* __restrict__ fz,
                        const work_t* __restrict__ g_prev,
                        work_t* __restrict__ pr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    work_t num = work_t(0), den = work_t(0);
    if (i < n) {
        const work_t g[3] = {-static_cast<work_t>(fx[i]),
                             -static_cast<work_t>(fy[i]),
                             -static_cast<work_t>(fz[i])};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const work_t gp = g_prev[c * n + i];
            num += g[c] * (g[c] - gp);
            den += gp * gp;
        }
    }

    extern __shared__ work_t warp_scratch[];
    work_t vals[2] = {num, den};
    work_t out[2];
    ATV::cuda::block_reduce_sum_n<2>(vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(&pr[0], out[0]);
        atomicAdd(&pr[1], out[1]);
    }
}

/// @brief β = max(0, num / den) if den > 0 else 0 — device-resident so
///        the step kernel chains off it without a host round-trip.
__global__ void pr_beta(work_t* __restrict__ pr) {
    const work_t num = pr[0], den = pr[1];
    pr[2] = (den > work_t(0)) ? fmax(work_t(0), num / den) : work_t(0);
}

/// @brief Fused CG step: d = −g + β·d_prev, save prev state, move, wrap.
///
/// Saves the PRE-step positions into x_prev (the convergence check
/// measures displacement against them) and the current gradient /
/// direction for the next iteration's β. PBC wrap matches the MD
/// integrator form (by-value Box, cached inv_l).
__global__ void
cg_step(real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
        const real_t* __restrict__ fx, const real_t* __restrict__ fy,
        const real_t* __restrict__ fz, work_t* __restrict__ x_prev,
        work_t* __restrict__ g_prev, work_t* __restrict__ d_prev,
        const work_t* __restrict__ pr, work_t alpha0, cuda::Box box, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const work_t beta = pr[2];
    const work_t g[3] = {-static_cast<work_t>(fx[i]),
                         -static_cast<work_t>(fy[i]),
                         -static_cast<work_t>(fz[i])};
    work_t pos[3] = {static_cast<work_t>(x[i]), static_cast<work_t>(y[i]),
                     static_cast<work_t>(z[i])};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const work_t d = -g[c] + beta * d_prev[c * n + i];
        x_prev[c * n + i] = pos[c];
        g_prev[c * n + i] = g[c];
        d_prev[c * n + i] = d;
        pos[c] += alpha0 * d;
    }

    const work_t Lx = work_t(box.cell[0]), Ly = work_t(box.cell[4]),
                 Lz = work_t(box.cell[8]);
    const work_t inv_Lx = work_t(box.inv_l[0]), inv_Ly = work_t(box.inv_l[1]),
                 inv_Lz = work_t(box.inv_l[2]);
    x[i] = static_cast<real_t>(pos[0] - Lx * floor(pos[0] * inv_Lx));
    y[i] = static_cast<real_t>(pos[1] - Ly * floor(pos[1] * inv_Ly));
    z[i] = static_cast<real_t>(pos[2] - Lz * floor(pos[2] * inv_Lz));
}

} // anonymous namespace

void CG::build(const BuildContext& ctx) {
    n_ = ctx.frame.n_atoms();
    const size_t bytes3n = static_cast<size_t>(3) * n_ * sizeof(work_t);
    cudaMalloc(&d_xprev_, bytes3n);
    cudaMalloc(&d_gprev_, bytes3n);
    cudaMalloc(&d_dprev_, bytes3n);
    cudaMalloc(&d_pr_, 3 * sizeof(work_t));
    // Zeroed g_prev/d_prev reproduce the CPU step-0 path: den = 0 → β = 0
    // → d = −g (steepest descent on the first moving step).
    cudaMemset(d_xprev_, 0, bytes3n);
    cudaMemset(d_gprev_, 0, bytes3n);
    cudaMemset(d_dprev_, 0, bytes3n);
    cudaMemset(d_pr_, 0, 3 * sizeof(work_t));

    step_ = 0;
    trace_.clear();
    conv_.build(n_);
}

void CG::launch(const LaunchContext& ctx) {
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        // Measure and publish. The Verdict is the loop's, through its handlers.
        conv_.measure(ctx, d_xprev_, step_, trace_);
        return;
    }

    // ── Search direction (Polak-Ribière+, GPU-pure) ──────────────────
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = n_;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    int shmem = cuda::warps_per_block(block) * sizeof(work_t);

    cudaMemsetAsync(d_pr_, 0, 2 * sizeof(work_t), stream);
    pr_dots<<<grid, block, shmem, stream>>>(frame.atoms.fx, frame.atoms.fy,
                                            frame.atoms.fz, d_gprev_, d_pr_, n);
    pr_beta<<<1, 1, 0, stream>>>(d_pr_);
    cg_step<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, d_xprev_, d_gprev_, d_dprev_, d_pr_,
        static_cast<work_t>(alpha0_), ctx.runtime.host_box, n);

    ++step_;
}

void CG::destroy(const DestroyContext& ctx) {
    cudaFree(d_xprev_);
    cudaFree(d_gprev_);
    cudaFree(d_dprev_);
    cudaFree(d_pr_);
    d_xprev_ = nullptr;
    d_gprev_ = nullptr;
    d_dprev_ = nullptr;
    d_pr_ = nullptr;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
