/// @file cuda/minimize/cg.cu
/// @brief Polak-Ribière+ CG kernels: device-resident β + fused step.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/minimize/cg.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace ATV {

using cuda::motion_t;
using cuda::real_t;

namespace cuda {
namespace minimizer {

namespace {

/// @brief Polak-Ribière dot products: pr[0] += g·(g−g_prev),
///        pr[1] += g_prev·g_prev, with g = −F.
__global__ void pr_dots(const real_t* __restrict__ fx,
                        const real_t* __restrict__ fy,
                        const real_t* __restrict__ fz,
                        const motion_t* __restrict__ g_prev,
                        motion_t* __restrict__ pr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    motion_t num = motion_t(0), den = motion_t(0);
    if (i < n) {
        const motion_t g[3] = {-static_cast<motion_t>(fx[i]),
                               -static_cast<motion_t>(fy[i]),
                               -static_cast<motion_t>(fz[i])};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const motion_t gp = g_prev[c * n + i];
            num += g[c] * (g[c] - gp);
            den += gp * gp;
        }
    }

    extern __shared__ motion_t warp_scratch[];
    motion_t vals[2] = {num, den};
    motion_t out[2];
    ATV::cuda::block_reduce_sum_n<2>(vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(&pr[0], out[0]);
        atomicAdd(&pr[1], out[1]);
    }
}

/// @brief β = max(0, num / den) if den > 0 else 0 — device-resident so
///        the step kernel chains off it without a host round-trip.
__global__ void pr_beta(motion_t* __restrict__ pr) {
    const motion_t num = pr[0], den = pr[1];
    pr[2] = (den > motion_t(0)) ? fmax(motion_t(0), num / den) : motion_t(0);
}

/// @brief Fused CG step: d = −g + β·d_prev, save prev state, move, wrap.
///
/// Saves the PRE-step positions into x_prev (the convergence check
/// measures displacement against them) and the current gradient /
/// direction for the next iteration's β. PBC wrap matches the MD
/// integrator form (by-value Box, cached inv_l).
__global__ void
cg_step(real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
        real_t* __restrict__ xcor, real_t* __restrict__ ycor,
        real_t* __restrict__ zcor, std::int8_t* __restrict__ image,
        const real_t* __restrict__ fx, const real_t* __restrict__ fy,
        const real_t* __restrict__ fz, motion_t* __restrict__ x_prev,
        motion_t* __restrict__ g_prev, motion_t* __restrict__ d_prev,
        const motion_t* __restrict__ pr, motion_t alpha0, cuda::MotionBox box,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const motion_t beta = pr[2];
    const motion_t g[3] = {-static_cast<motion_t>(fx[i]),
                           -static_cast<motion_t>(fy[i]),
                           -static_cast<motion_t>(fz[i])};
    motion_t pos[3] = {load_position(x, xcor, i), load_position(y, ycor, i),
                       load_position(z, zcor, i)};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const motion_t d = -g[c] + beta * d_prev[c * n + i];
        x_prev[c * n + i] = pos[c];
        g_prev[c * n + i] = g[c];
        d_prev[c * n + i] = d;
        pos[c] += alpha0 * d;
    }

    wrap_position(x, y, z, xcor, ycor, zcor, image, i, box, pos[0], pos[1],
                  pos[2]);
}

} // anonymous namespace

void CG::build(const BuildContext& ctx) {
    n_ = ctx.frame.n_atoms();
    if (n_ < 0 || n_ > std::numeric_limits<int>::max() / 3)
        throw std::invalid_argument(
            "cuda::minimizer::CG: atom count exceeds planar index capacity");
    const size_t bytes3n =
        static_cast<size_t>(3) * std::max(1, n_) * sizeof(motion_t);
    ATV_CUDA_CHECK(cudaMalloc(&d_xprev_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_gprev_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_dprev_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_pr_, 3 * sizeof(motion_t)));
    // Zeroed g_prev/d_prev reproduce the CPU step-0 path: den = 0 → β = 0
    // → d = −g (steepest descent on the first moving step).
    ATV_CUDA_CHECK(cudaMemset(d_xprev_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_gprev_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_dprev_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_pr_, 0, 3 * sizeof(motion_t)));

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
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    int shmem = cuda::warps_per_block(block) * sizeof(motion_t);

    ATV_CUDA_CHECK(cudaMemsetAsync(d_pr_, 0, 2 * sizeof(motion_t), stream));
    pr_dots<<<grid, block, shmem, stream>>>(frame.atoms.fx, frame.atoms.fy,
                                            frame.atoms.fz, d_gprev_, d_pr_, n);
    ATV_CUDA_CHECK(cudaGetLastError());
    pr_beta<<<1, 1, 0, stream>>>(d_pr_);
    ATV_CUDA_CHECK(cudaGetLastError());
    cg_step<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.x_correction,
        frame.atoms.y_correction, frame.atoms.z_correction, frame.atoms.image,
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, d_xprev_, d_gprev_,
        d_dprev_, d_pr_, static_cast<motion_t>(alpha0_),
        ctx.runtime.host_motion_box, n);
    ATV_CUDA_CHECK(cudaGetLastError());

    ++step_;
}

void CG::destroy(const DestroyContext& ctx) {
    ATV_CUDA_CHECK(cudaFree(d_xprev_));
    ATV_CUDA_CHECK(cudaFree(d_gprev_));
    ATV_CUDA_CHECK(cudaFree(d_dprev_));
    ATV_CUDA_CHECK(cudaFree(d_pr_));
    d_xprev_ = nullptr;
    d_gprev_ = nullptr;
    d_dprev_ = nullptr;
    d_pr_ = nullptr;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
