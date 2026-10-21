/// @file cuda/minimize/lbfgs.cu
/// @brief L-BFGS kernels: device-resident two-loop recursion +
///        multi-criteria convergence via the shared ConvCheck.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
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

/// @brief Pack frame positions + gradient (g = −F) into planar motion_t
///        buffers for the history / two-loop vector algebra.
__global__ void
pack_xg(const real_t* __restrict__ x, const real_t* __restrict__ y,
        const real_t* __restrict__ z, const real_t* __restrict__ xcor,
        const real_t* __restrict__ ycor, const real_t* __restrict__ zcor,
        const real_t* __restrict__ fx, const real_t* __restrict__ fy,
        const real_t* __restrict__ fz, motion_t* __restrict__ xp,
        motion_t* __restrict__ gp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    xp[i] = load_position(x, xcor, i);
    xp[n + i] = load_position(y, ycor, i);
    xp[2 * n + i] = load_position(z, zcor, i);
    gp[i] = -static_cast<motion_t>(fx[i]);
    gp[n + i] = -static_cast<motion_t>(fy[i]);
    gp[2 * n + i] = -static_cast<motion_t>(fz[i]);
}

/// @brief Candidate history pair: s = x − x_prev (minimum-imaged — the
///        step kernel wraps positions), y = g − g_prev, ys += y·s.
///
/// Writes into the caller-chosen ring slot; the host registers the slot
/// only if the curvature condition ys > 1e-10 holds, so a rejected
/// candidate is simply overwritten by the next one.
__global__ void
sy_dot(const motion_t* __restrict__ xc, const motion_t* __restrict__ gc,
       const motion_t* __restrict__ x_prev, const motion_t* __restrict__ g_prev,
       motion_t* __restrict__ s_slot, motion_t* __restrict__ y_slot,
       cuda::MotionBox box, motion_t* __restrict__ ys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    motion_t my_ys = motion_t(0);
    if (i < n) {
        motion_t sx = xc[i] - x_prev[i];
        motion_t sy = xc[n + i] - x_prev[n + i];
        motion_t sz = xc[2 * n + i] - x_prev[2 * n + i];
        box.min_image(sx, sy, sz);
        const motion_t s[3] = {sx, sy, sz};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const motion_t yv = gc[c * n + i] - g_prev[c * n + i];
            s_slot[c * n + i] = s[c];
            y_slot[c * n + i] = yv;
            my_ys += yv * s[c];
        }
    }

    extern __shared__ motion_t warp_scratch[];
    motion_t sum = ATV::cuda::block_reduce_sum(my_ys, warp_scratch);
    if (threadIdx.x == 0)
        atomicAdd(ys, sum);
}

/// @brief Generic 3N dot product: out += a·b.
__global__ void dot3n(const motion_t* __restrict__ a,
                      const motion_t* __restrict__ b,
                      motion_t* __restrict__ out, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    motion_t v = (j < n3) ? a[j] * b[j] : motion_t(0);

    extern __shared__ motion_t warp_scratch[];
    motion_t sum = ATV::cuda::block_reduce_sum(v, warp_scratch);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

/// @brief Two-loop backward pass step: a_k = ρ_k · (s_k·q) (already in
///        *dot), q −= a_k · y_k; thread 0 parks a_k for the forward pass.
__global__ void axpy_backward(motion_t* __restrict__ q,
                              const motion_t* __restrict__ y_k,
                              const motion_t* __restrict__ dot, motion_t rho,
                              motion_t* __restrict__ alpha_k, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const motion_t a = rho * (*dot);
    if (j == 0)
        *alpha_k = a;
    if (j < n3)
        q[j] -= a * y_k[j];
}

/// @brief Initial Hessian scaling r = γ·q with γ = 1 / (ρ_newest · y·y)
///        (γ = 1 when the curvature norm underflows), in place.
__global__ void gamma_scale(motion_t* __restrict__ q,
                            const motion_t* __restrict__ yy,
                            motion_t rho_newest, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n3)
        return;
    const motion_t y2 = *yy;
    const motion_t gamma =
        (y2 > motion_t(1e-30)) ? motion_t(1) / (rho_newest * y2) : motion_t(1);
    q[j] *= gamma;
}

/// @brief Two-loop forward pass step: r += s_k · (a_k − ρ_k · (y_k·r)).
__global__ void axpy_forward(motion_t* __restrict__ r,
                             const motion_t* __restrict__ s_k,
                             const motion_t* __restrict__ alpha_k,
                             const motion_t* __restrict__ dot, motion_t rho,
                             int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n3)
        return;
    const motion_t b = rho * (*dot);
    r[j] += s_k[j] * (*alpha_k - b);
}

/// @brief Max-|·| reduction over a 3N buffer (step-length cap input).
__global__ void max_abs3n(const motion_t* __restrict__ a,
                          motion_t* __restrict__ out, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    motion_t v = (j < n3) ? fabs(a[j]) : motion_t(0);

    extern __shared__ motion_t warp_scratch[];
    motion_t m = ATV::cuda::block_reduce_max(v, warp_scratch);
    if (threadIdx.x == 0)
        ATV::cuda::atomic_max_nonneg(out, m);
}

/// @brief Apply the L-BFGS step: direction = −r, α capped so no atom
///        moves more than 0.1 Angstrom; saves prev state, moves, wraps.
///
/// Thread (0,0) parks the applied α in ls[0] for the line search (read
/// back on the host during the next convergence-check sync).
__global__ void apply_step(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ xcor, real_t* __restrict__ ycor,
    real_t* __restrict__ zcor, std::int8_t* __restrict__ image,
    std::int8_t* __restrict__ image_prev, const motion_t* __restrict__ xc,
    const motion_t* __restrict__ gc, const motion_t* __restrict__ r,
    motion_t* __restrict__ dir, motion_t* __restrict__ x_prev,
    motion_t* __restrict__ g_prev, const motion_t* __restrict__ dir_max,
    motion_t alpha0, motion_t* __restrict__ ls, cuda::MotionBox box, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const motion_t dmax = fmax(*dir_max, motion_t(1e-20));
    const motion_t alpha = fmin(alpha0, motion_t(0.1) / dmax);
    if (i == 0)
        ls[0] = alpha;

    motion_t pos[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        image_prev[3 * i + c] = image[3 * i + c];
        const motion_t d = -r[c * n + i];
        dir[c * n + i] = d;
        x_prev[c * n + i] = xc[c * n + i];
        g_prev[c * n + i] = gc[c * n + i];
        pos[c] = xc[c * n + i] + alpha * d;
    }
    wrap_position(x, y, z, xcor, ycor, zcor, image, i, box, pos[0], pos[1],
                  pos[2]);
}

/// @brief Armijo rollback: re-take the trial from x_prev with the
///        contracted α along the SAME stored direction, then wrap.
__global__ void rollback(real_t* __restrict__ x, real_t* __restrict__ y,
                         real_t* __restrict__ z, real_t* __restrict__ xcor,
                         real_t* __restrict__ ycor, real_t* __restrict__ zcor,
                         std::int8_t* __restrict__ image,
                         const std::int8_t* __restrict__ image_prev,
                         const motion_t* __restrict__ x_prev,
                         const motion_t* __restrict__ dir, motion_t alpha,
                         cuda::MotionBox box, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t pos[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        image[3 * i + c] = image_prev[3 * i + c];
        pos[c] = x_prev[c * n + i] + alpha * dir[c * n + i];
    }
    wrap_position(x, y, z, xcor, ycor, zcor, image, i, box, pos[0], pos[1],
                  pos[2]);
}

} // anonymous namespace

void LBFGS::build(const BuildContext& ctx) {
    n_ = ctx.frame.n_atoms();
    if (n_ < 0 || n_ > std::numeric_limits<int>::max() / 3 || memory_ <= 0)
        throw std::invalid_argument(
            "cuda::minimizer::LBFGS: invalid atom count or memory capacity");
    n3_ = 3 * n_;
    const size_t bytes3n =
        static_cast<size_t>(std::max(1, n3_)) * sizeof(motion_t);

    if (static_cast<size_t>(memory_)
        > std::numeric_limits<size_t>::max() / bytes3n)
        throw std::overflow_error(
            "cuda::minimizer::LBFGS: history allocation size overflow");

    ATV_CUDA_CHECK(cudaMalloc(&d_x_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_g_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_xprev_, bytes3n));
    ATV_CUDA_CHECK(
        cudaMalloc(&d_image_prev_, static_cast<size_t>(std::max(1, n3_))
                                       * sizeof(std::int8_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_gprev_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_q_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_dir_, bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_s_, static_cast<size_t>(memory_) * bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_y_, static_cast<size_t>(memory_) * bytes3n));
    ATV_CUDA_CHECK(cudaMalloc(&d_alpha_, memory_ * sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_dot_, 2 * sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_ys_, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_ls_, 2 * sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMemset(d_xprev_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_gprev_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_dir_, 0, bytes3n));
    ATV_CUDA_CHECK(cudaMemset(d_ls_, 0, 2 * sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMallocHost(&h_ys_, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMallocHost(&h_ls_, 2 * sizeof(motion_t)));

    rho_.assign(memory_, 0.0);
    n_stored_ = 0;
    oldest_ = 0;
    step_ = 0;
    has_prev_ = false;
    ls_had_trial_ = false;
    ls_fetch_pending_ = false;
    ls_alpha_ = 0;
    ls_gd_ = 0;
    trace_.clear();
    conv_.build(n_);
}

void LBFGS::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);

    // ── Convergence check phase ──────────────────────────────────────
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        // Batch the line-search scalars produced by this iteration's
        // search_direction onto the ConvCheck sync — zero extra host blocks.
        if (ls_fetch_pending_)
            ATV_CUDA_CHECK(cudaMemcpyAsync(h_ls_, d_ls_, 2 * sizeof(motion_t),
                                           cudaMemcpyDeviceToHost, stream));
        // Measure and publish. The Verdict is the loop's, through its handlers.
        conv_.measure(ctx, d_xprev_, step_, trace_);
        if (ls_fetch_pending_) {
            ls_alpha_ = static_cast<double>(h_ls_[0]);
            ls_gd_ = static_cast<double>(h_ls_[1]);
            ls_fetch_pending_ = false;
        }
        return;
    }

    // ── Search direction phase ───────────────────────────────────────
    auto& frame = ctx.frame;
    const int n = n_;
    if (n <= 0)
        return;
    const int n3 = n3_;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    int grid3 = cuda::grid_size(n3);
    int shmem = cuda::warps_per_block(block) * sizeof(motion_t);

    pack_xg<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.x_correction,
        frame.atoms.y_correction, frame.atoms.z_correction, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, d_x_, d_g_, n);

    ATV_CUDA_CHECK(cudaGetLastError());

    // ── Armijo rejection & rollback (host decision on cached scalars) ─
    //
    // Mirrors the CPU flow: the trial taken during the previous
    // search_direction was evaluated by the force phase; its energy
    // arrived on the host through the previous ConvCheck sync. On
    // rejection, roll back to x_prev with the contracted α and return —
    // history update + new direction happen on the next accepted trial.
    if (ls_enabled_ && ls_had_trial_) {
        const double armijo_rhs = ls_c1_ * ls_alpha_ * ls_gd_;
        const double dE = conv_.last_energy() - ls_origin_energy_;
        if (!(dE <= armijo_rhs) && ls_alpha_ > ls_min_alpha_) {
            ls_alpha_ *= ls_contract_;
            rollback<<<grid, block, 0, stream>>>(
                frame.atoms.x, frame.atoms.y, frame.atoms.z,
                frame.atoms.x_correction, frame.atoms.y_correction,
                frame.atoms.z_correction, frame.atoms.image, d_image_prev_,
                d_xprev_, d_dir_, static_cast<motion_t>(ls_alpha_),
                ctx.runtime.host_motion_box, n);
            ATV_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    // ── History update (one 8-byte D2H — see file header) ────────────
    if (has_prev_) {
        const int idx = (n_stored_ < memory_) ? n_stored_ : oldest_;
        ATV_CUDA_CHECK(cudaMemsetAsync(d_ys_, 0, sizeof(motion_t), stream));
        sy_dot<<<grid, block, shmem, stream>>>(
            d_x_, d_g_, d_xprev_, d_gprev_,
            d_s_ + static_cast<size_t>(idx) * n3,
            d_y_ + static_cast<size_t>(idx) * n3, ctx.runtime.host_motion_box,
            d_ys_, n);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaMemcpyAsync(h_ys_, d_ys_, sizeof(motion_t),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        const double ys = static_cast<double>(*h_ys_);
        // Curvature condition gates the host ring bookkeeping (and with
        // it the number of two-loop kernel launches below).
        if (ys > 1e-10) {
            rho_[idx] = 1.0 / ys;
            if (n_stored_ < memory_)
                ++n_stored_;
            else
                oldest_ = (oldest_ + 1) % memory_;
        }
    }

    // ── Two-loop recursion (GPU-pure, device-scalar chained) ─────────
    ATV_CUDA_CHECK(cudaMemcpyAsync(d_q_, d_g_,
                                   static_cast<size_t>(n3) * sizeof(motion_t),
                                   cudaMemcpyDeviceToDevice, stream));

    for (int k = n_stored_ - 1; k >= 0; --k) {
        const int idx = (oldest_ + k) % memory_;
        ATV_CUDA_CHECK(cudaMemsetAsync(d_dot_, 0, sizeof(motion_t), stream));
        dot3n<<<grid3, block, shmem, stream>>>(
            d_s_ + static_cast<size_t>(idx) * n3, d_q_, d_dot_, n3);
        ATV_CUDA_CHECK(cudaGetLastError());
        axpy_backward<<<grid3, block, 0, stream>>>(
            d_q_, d_y_ + static_cast<size_t>(idx) * n3, d_dot_,
            static_cast<motion_t>(rho_[idx]), d_alpha_ + k, n3);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    if (n_stored_ > 0) {
        const int newest = (oldest_ + n_stored_ - 1) % memory_;
        ATV_CUDA_CHECK(cudaMemsetAsync(d_dot_, 0, sizeof(motion_t), stream));
        dot3n<<<grid3, block, shmem, stream>>>(
            d_y_ + static_cast<size_t>(newest) * n3,
            d_y_ + static_cast<size_t>(newest) * n3, d_dot_, n3);
        ATV_CUDA_CHECK(cudaGetLastError());
        gamma_scale<<<grid3, block, 0, stream>>>(
            d_q_, d_dot_, static_cast<motion_t>(rho_[newest]), n3);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    for (int k = 0; k < n_stored_; ++k) {
        const int idx = (oldest_ + k) % memory_;
        ATV_CUDA_CHECK(cudaMemsetAsync(d_dot_, 0, sizeof(motion_t), stream));
        dot3n<<<grid3, block, shmem, stream>>>(
            d_y_ + static_cast<size_t>(idx) * n3, d_q_, d_dot_, n3);
        ATV_CUDA_CHECK(cudaGetLastError());
        axpy_forward<<<grid3, block, 0, stream>>>(
            d_q_, d_s_ + static_cast<size_t>(idx) * n3, d_alpha_ + k, d_dot_,
            static_cast<motion_t>(rho_[idx]), n3);
        ATV_CUDA_CHECK(cudaGetLastError());
    }

    // ── Step application (α capped at 0.1 Angstrom max displacement) ─
    ATV_CUDA_CHECK(cudaMemsetAsync(d_dot_ + 1, 0, sizeof(motion_t), stream));
    max_abs3n<<<grid3, block, shmem, stream>>>(d_q_, d_dot_ + 1, n3);
    ATV_CUDA_CHECK(cudaGetLastError());
    apply_step<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.x_correction,
        frame.atoms.y_correction, frame.atoms.z_correction, frame.atoms.image,
        d_image_prev_, d_x_, d_g_, d_q_, d_dir_, d_xprev_, d_gprev_, d_dot_ + 1,
        static_cast<motion_t>(alpha0_), d_ls_, ctx.runtime.host_motion_box, n);
    ATV_CUDA_CHECK(cudaGetLastError());

    if (ls_enabled_) {
        // g·d at the trial origin, fetched with the next ConvCheck sync.
        ATV_CUDA_CHECK(cudaMemsetAsync(d_ls_ + 1, 0, sizeof(motion_t), stream));
        dot3n<<<grid3, block, shmem, stream>>>(d_g_, d_dir_, d_ls_ + 1, n3);
        ATV_CUDA_CHECK(cudaGetLastError());
        ls_origin_energy_ = conv_.last_energy();
        ls_had_trial_ = true;
        ls_fetch_pending_ = true;
    }

    has_prev_ = true;
    ++step_;
}

void LBFGS::destroy(const DestroyContext& ctx) {
    ATV_CUDA_CHECK(cudaFree(d_x_));
    ATV_CUDA_CHECK(cudaFree(d_g_));
    ATV_CUDA_CHECK(cudaFree(d_xprev_));
    ATV_CUDA_CHECK(cudaFree(d_image_prev_));
    d_image_prev_ = nullptr;
    ATV_CUDA_CHECK(cudaFree(d_gprev_));
    ATV_CUDA_CHECK(cudaFree(d_q_));
    ATV_CUDA_CHECK(cudaFree(d_dir_));
    ATV_CUDA_CHECK(cudaFree(d_s_));
    ATV_CUDA_CHECK(cudaFree(d_y_));
    ATV_CUDA_CHECK(cudaFree(d_alpha_));
    ATV_CUDA_CHECK(cudaFree(d_dot_));
    ATV_CUDA_CHECK(cudaFree(d_ys_));
    ATV_CUDA_CHECK(cudaFree(d_ls_));
    ATV_CUDA_CHECK(cudaFreeHost(h_ys_));
    ATV_CUDA_CHECK(cudaFreeHost(h_ls_));
    d_x_ = d_g_ = d_xprev_ = d_gprev_ = d_q_ = d_dir_ = nullptr;
    d_s_ = d_y_ = d_alpha_ = d_dot_ = d_ys_ = d_ls_ = nullptr;
    h_ys_ = h_ls_ = nullptr;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
