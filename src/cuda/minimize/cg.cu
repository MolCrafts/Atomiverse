/// @file cuda/minimize/cg.cu
/// @brief Polak-Ribière+ CG kernels: device-resident β + fused step.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/cg.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace minimizer {

namespace {

/// @brief Polak-Ribière dot products: pr[0] += g·(g−g_prev),
///        pr[1] += g_prev·g_prev, with g = −F decoded from fixed point.
///        Both reduce in acc_t.
__global__ void pr_dots(const fixed_t* __restrict__ fx,
                        const fixed_t* __restrict__ fy,
                        const fixed_t* __restrict__ fz,
                        const mixed_t* __restrict__ g_prev,
                        acc_t* __restrict__ pr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t num = acc_t(0), den = acc_t(0);
    if (i < n) {
        const mixed_t g[3] = {-decode_force<mixed_t>(fx[i]),
                              -decode_force<mixed_t>(fy[i]),
                              -decode_force<mixed_t>(fz[i])};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const acc_t gc = acc_t(g[c]);
            const acc_t gp = acc_t(g_prev[c * n + i]);
            num += gc * (gc - gp);
            den += gp * gp;
        }
    }

    ::cuda::std::array<acc_t, 2> vals = {num, den};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 2>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(vals, SumComponents<acc_t, 2>{});
    if (threadIdx.x == 0) {
        atomicAdd(&pr[0], out[0]);
        atomicAdd(&pr[1], out[1]);
    }
}

/// @brief β = max(0, num / den) if den > 0 else 0 — device-resident so
///        the step kernel chains off it without a host round-trip.
__global__ void pr_beta(acc_t* __restrict__ pr) {
    const acc_t num = pr[0], den = pr[1];
    pr[2] = (den > acc_t(0)) ? fmax(acc_t(0), num / den) : acc_t(0);
}

/// @brief Fused CG step: d = −g + β·d_prev, save prev state, move, wrap.
///
/// Saves the PRE-step positions (reconstructed primary + correction, mixed_t)
/// into x_prev (the convergence check measures displacement against them)
/// and the current gradient / direction for the next iteration's β. PBC
/// wrap matches the MD integrators (whole lattice vectors, FP64 Box).
/// A faulted force is never consumed.
__global__ void
cg_step(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
        pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
        const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
        const fixed_t* __restrict__ fz, mixed_t* __restrict__ x_prev,
        mixed_t* __restrict__ g_prev, mixed_t* __restrict__ d_prev,
        const acc_t* __restrict__ pr, mixed_t alpha0, ForceFault* fault,
        const std::int64_t* __restrict__ tag, long long step, cuda::Box box,
        int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const mixed_t beta = mixed_t(pr[2]);
    const mixed_t g[3] = {-decode_force<mixed_t>(fx[i]),
                          -decode_force<mixed_t>(fy[i]),
                          -decode_force<mixed_t>(fz[i])};
    mixed_t pos[3] = {load_position(x, xc, i), load_position(y, yc, i),
                      load_position(z, zc, i)};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const mixed_t d = -g[c] + beta * d_prev[c * n + i];
        x_prev[c * n + i] = pos[c];
        g_prev[c * n + i] = g[c];
        d_prev[c * n + i] = d;
        pos[c] += alpha0 * d;
    }

    box.wrap_position(pos[0], pos[1], pos[2]);
    store_position(x, xc, i, pos[0]);
    store_position(y, yc, i, pos[1]);
    store_position(z, zc, i, pos[2]);
}

} // anonymous namespace

void CG::build(const BuildContext& ctx) {
    n_ = ctx.frame.n_atoms();
    const size_t bytes3n = static_cast<size_t>(3) * n_ * sizeof(mixed_t);
    cudaMalloc(&d_xprev_, bytes3n);
    cudaMalloc(&d_gprev_, bytes3n);
    cudaMalloc(&d_dprev_, bytes3n);
    cudaMalloc(&d_pr_, 3 * sizeof(acc_t));
    // Zeroed g_prev/d_prev reproduce the CPU step-0 path: den = 0 → β = 0
    // → d = −g (steepest descent on the first moving step).
    cudaMemset(d_xprev_, 0, bytes3n);
    cudaMemset(d_gprev_, 0, bytes3n);
    cudaMemset(d_dprev_, 0, bytes3n);
    cudaMemset(d_pr_, 0, 3 * sizeof(acc_t));

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
    constexpr int shmem = 0;

    cudaMemsetAsync(d_pr_, 0, 2 * sizeof(acc_t), stream);
    pr_dots<<<grid, block, shmem, stream>>>(frame.atoms.fx, frame.atoms.fy,
                                            frame.atoms.fz, d_gprev_, d_pr_, n);
    pr_beta<<<1, 1, 0, stream>>>(d_pr_);
    const auto& a = frame.atoms;
    cg_step<<<grid, block, 0, stream>>>(
        a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.fx, a.fy, a.fz, d_xprev_,
        d_gprev_, d_dprev_, d_pr_, static_cast<mixed_t>(alpha0_),
        frame.force_fault, a.tag, ctx.step, ctx.runtime.host_box, n);

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
