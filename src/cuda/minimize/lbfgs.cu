/// @file cuda/minimize/lbfgs.cu
/// @brief L-BFGS kernels: device-resident two-loop recursion +
///        multi-criteria convergence via the shared ConvCheck.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ATV {
namespace cuda {
namespace minimizer {

namespace {

/// @brief PBC wrap (whole lattice vectors, FP64 cell) + split store shared
///        by the step/rollback kernels; writes primary and correction columns
///        and counts the removed lattice translation into `image` (atom i's
///        three flags; null when the Frame carries none).
__device__ __forceinline__ void
wrap_store(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
           pos_t* __restrict__ xc, pos_t* __restrict__ yc,
           pos_t* __restrict__ zc, ImageInt* image, int i, const mixed_t pos[3],
           const cuda::Box& box) {
    mixed_t px = pos[0], py = pos[1], pz = pos[2];
    box.wrap_position(px, py, pz, image);
    store_position(x, xc, i, px);
    store_position(y, yc, i, py);
    store_position(z, zc, i, pz);
}

/// @brief Pack reconstructed frame positions + decoded gradient (g = −F)
///        into planar mixed_t buffers for the history / two-loop algebra.
__global__ void
pack_xg(const pos_t* __restrict__ x, const pos_t* __restrict__ y,
        const pos_t* __restrict__ z, const pos_t* __restrict__ xc,
        const pos_t* __restrict__ yc, const pos_t* __restrict__ zc,
        const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
        const fixed_t* __restrict__ fz, mixed_t* __restrict__ xp,
        mixed_t* __restrict__ gp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    xp[i] = load_position(x, xc, i);
    xp[n + i] = load_position(y, yc, i);
    xp[2 * n + i] = load_position(z, zc, i);
    gp[i] = -decode_force<mixed_t>(fx[i]);
    gp[n + i] = -decode_force<mixed_t>(fy[i]);
    gp[2 * n + i] = -decode_force<mixed_t>(fz[i]);
}

/// @brief Candidate history pair: s = x − x_prev (minimum-imaged — the
///        step kernel wraps positions), y = g − g_prev, ys += y·s (acc_t).
///
/// Writes into the caller-chosen ring slot; the host registers the slot
/// only if the curvature condition ys > 1e-10 holds, so a rejected
/// candidate is simply overwritten by the next one.
__global__ void
sy_dot(const mixed_t* __restrict__ xc, const mixed_t* __restrict__ gc,
       const mixed_t* __restrict__ x_prev, const mixed_t* __restrict__ g_prev,
       mixed_t* __restrict__ s_slot, mixed_t* __restrict__ y_slot,
       cuda::Box box, acc_t* __restrict__ ys, int n) {
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t my_ys = acc_t(0);
    if (i < n) {
        mixed_t sx = xc[i] - x_prev[i];
        mixed_t sy = xc[n + i] - x_prev[n + i];
        mixed_t sz = xc[2 * n + i] - x_prev[2 * n + i];
        box.min_image(sx, sy, sz);
        const mixed_t s[3] = {sx, sy, sz};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const mixed_t yv = gc[c * n + i] - g_prev[c * n + i];
            s_slot[c * n + i] = s[c];
            y_slot[c * n + i] = yv;
            my_ys += acc_t(yv) * acc_t(s[c]);
        }
    }

    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    acc_t sum = Reduce(reduction).Sum(my_ys);
    if (threadIdx.x == 0)
        atomicAdd(ys, sum);
}

/// @brief Generic 3N dot product: out += a·b, reduced in acc_t.
__global__ void dot3n(const mixed_t* __restrict__ a,
                      const mixed_t* __restrict__ b, acc_t* __restrict__ out,
                      int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t v = (j < n3) ? acc_t(a[j]) * acc_t(b[j]) : acc_t(0);

    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    acc_t sum = Reduce(reduction).Sum(v);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

/// @brief Two-loop backward pass step: a_k = ρ_k · (s_k·q) (already in
///        *dot), q −= a_k · y_k; thread 0 parks a_k for the forward pass.
__global__ void axpy_backward(mixed_t* __restrict__ q,
                              const mixed_t* __restrict__ y_k,
                              const acc_t* __restrict__ dot, mixed_t rho,
                              mixed_t* __restrict__ alpha_k, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const mixed_t a = rho * mixed_t(*dot);
    if (j == 0)
        *alpha_k = a;
    if (j < n3)
        q[j] -= a * y_k[j];
}

/// @brief Initial Hessian scaling r = γ·q with γ = 1 / (ρ_newest · y·y)
///        (γ = 1 when the curvature norm underflows), in place.
__global__ void gamma_scale(mixed_t* __restrict__ q,
                            const acc_t* __restrict__ yy, mixed_t rho_newest,
                            int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n3)
        return;
    const acc_t y2 = *yy;
    const mixed_t gamma = (y2 > acc_t(1e-30))
                              ? mixed_t(acc_t(1) / (acc_t(rho_newest) * y2))
                              : mixed_t(1);
    q[j] *= gamma;
}

/// @brief Two-loop forward pass step: r += s_k · (a_k − ρ_k · (y_k·r)).
__global__ void axpy_forward(mixed_t* __restrict__ r,
                             const mixed_t* __restrict__ s_k,
                             const mixed_t* __restrict__ alpha_k,
                             const acc_t* __restrict__ dot, mixed_t rho,
                             int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n3)
        return;
    const mixed_t b = rho * mixed_t(*dot);
    r[j] += s_k[j] * (*alpha_k - b);
}

/// @brief Max-|·| reduction over a 3N buffer (step-length cap input).
__global__ void max_abs3n(const mixed_t* __restrict__ a,
                          acc_t* __restrict__ out, int n3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t v = (j < n3) ? fabs(acc_t(a[j])) : acc_t(0);

    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    acc_t m = Reduce(reduction).Reduce(v, ::cuda::maximum<>{});
    if (threadIdx.x == 0)
        ATV::cuda::atomic_max_nonneg(out, m);
}

/// @brief Apply the L-BFGS step: direction = −r, α capped so no atom
///        moves more than 0.1 Angstrom; saves prev state, moves, wraps.
///
/// Thread (0,0) parks the applied α in ls[0] for the line search (read
/// back on the host during the next convergence-check sync). The direction
/// was built from the decoded forces of this Frame, so a faulted force is
/// never consumed: the step (and its history save) is skipped.
__global__ void
apply_step(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
           pos_t* __restrict__ xcorr, pos_t* __restrict__ ycorr,
           pos_t* __restrict__ zcorr, const mixed_t* __restrict__ xc,
           const mixed_t* __restrict__ gc, const mixed_t* __restrict__ r,
           mixed_t* __restrict__ dir, mixed_t* __restrict__ x_prev,
           mixed_t* __restrict__ g_prev, ImageInt* __restrict__ image,
           ImageInt* __restrict__ image_prev, const acc_t* __restrict__ dir_max,
           mixed_t alpha0, acc_t* __restrict__ ls, ForceFault* fault,
           const std::int64_t* __restrict__ tag, long long step, cuda::Box box,
           int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const mixed_t dmax = mixed_t(fmax(*dir_max, acc_t(1e-20)));
    const mixed_t alpha = fmin(alpha0, mixed_t(0.1) / dmax);
    if (i == 0)
        ls[0] = acc_t(alpha);

    mixed_t pos[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const mixed_t d = -r[c * n + i];
        dir[c * n + i] = d;
        x_prev[c * n + i] = xc[c * n + i];
        g_prev[c * n + i] = gc[c * n + i];
        pos[c] = xc[c * n + i] + alpha * d;
    }
    ImageInt* row = nullptr;
    if (image != nullptr) {
        row = image + 3 * i;
#pragma unroll
        for (int c = 0; c < 3; ++c)
            image_prev[3 * i + c] = row[c];
    }
    wrap_store(x, y, z, xcorr, ycorr, zcorr, row, i, pos, box);
}

/// @brief Armijo rollback: re-take the trial from x_prev with the
///        contracted α along the SAME stored direction, then wrap.
///
/// x_prev is the reconstructed coordinate, so the split store restores the
/// primary and correction columns together; the image flags restart from
/// their x_prev values, so the rejected trial's crossings are undone.
__global__ void
rollback(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
         pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
         ImageInt* __restrict__ image, const ImageInt* __restrict__ image_prev,
         const mixed_t* __restrict__ x_prev, const mixed_t* __restrict__ dir,
         mixed_t alpha, cuda::Box box, int n) {
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    mixed_t pos[3];
#pragma unroll
    for (int c = 0; c < 3; ++c)
        pos[c] = x_prev[c * n + i] + alpha * dir[c * n + i];
    ImageInt* row = nullptr;
    if (image != nullptr) {
        row = image + 3 * i;
#pragma unroll
        for (int c = 0; c < 3; ++c)
            row[c] = image_prev[3 * i + c];
    }
    wrap_store(x, y, z, xc, yc, zc, row, i, pos, box);
}

} // anonymous namespace

void LBFGS::build(const BuildContext& ctx) {
    if (memory_ <= 0 || max_ls_ <= 0 || !std::isfinite(alpha0_) || alpha0_ <= 0
        || (ls_enabled_
            && (!std::isfinite(ls_c1_) || ls_c1_ <= 0 || ls_c1_ >= 1
                || !std::isfinite(ls_contract_) || ls_contract_ <= 0
                || ls_contract_ >= 1 || !std::isfinite(ls_min_alpha_)
                || ls_min_alpha_ <= 0)))
        throw std::invalid_argument("CUDA LBFGS requires finite positive "
                                    "controls and Armijo c1/contract in (0,1)");
    n_ = ctx.frame.n_atoms();
    n3_ = 3 * n_;
    const size_t bytes3n = static_cast<size_t>(n3_) * sizeof(mixed_t);

    cudaMalloc(&d_x_, bytes3n);
    cudaMalloc(&d_g_, bytes3n);
    cudaMalloc(&d_xprev_, bytes3n);
    cudaMalloc(&d_gprev_, bytes3n);
    cudaMalloc(&d_q_, bytes3n);
    cudaMalloc(&d_dir_, bytes3n);
    cudaMalloc(&d_s_, static_cast<size_t>(memory_) * bytes3n);
    cudaMalloc(&d_y_, static_cast<size_t>(memory_) * bytes3n);
    cudaMalloc(&d_alpha_, memory_ * sizeof(mixed_t));
    cudaMalloc(&d_image_prev_, static_cast<size_t>(n3_) * sizeof(ImageInt));
    cudaMemset(d_image_prev_, 0, static_cast<size_t>(n3_) * sizeof(ImageInt));
    cudaMalloc(&d_dot_, 2 * sizeof(acc_t));
    cudaMalloc(&d_ys_, sizeof(acc_t));
    cudaMalloc(&d_ls_, 3 * sizeof(acc_t));
    cudaMemset(d_xprev_, 0, bytes3n);
    cudaMemset(d_gprev_, 0, bytes3n);
    cudaMemset(d_dir_, 0, bytes3n);
    cudaMemset(d_ls_, 0, 3 * sizeof(acc_t));
    cudaMallocHost(&h_ys_, sizeof(acc_t));
    cudaMallocHost(&h_ls_, 3 * sizeof(acc_t));

    rho_.assign(memory_, 0.0);
    n_stored_ = 0;
    oldest_ = 0;
    step_ = 0;
    has_prev_ = false;
    ls_alpha_ = 0;
    ls_gd_ = 0;
    trace_.clear();
    conv_.build(n_);
}

void LBFGS::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);

    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        conv_.measure(ctx, d_xprev_, step_, trace_);
        return;
    }
    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        if (!ls_enabled_)
            return;
        if (!ctx.runtime.opt.reeval_forces)
            throw std::logic_error(
                "CUDA LBFGS line search requires Minimize force replay");
        struct RestoreTrial {
            cuda::Context& context;
            bool saved;
            ~RestoreTrial() { context.exec.trial_evaluation = saved; }
        } guard{ctx.runtime, ctx.runtime.exec.trial_evaluation};
        ctx.runtime.exec.trial_evaluation = true;
        auto& a = ctx.frame.atoms;
        auto set_trial = [&](double alpha) {
            rollback<<<cuda::grid_size(n_), cuda::kDefaultBlock, 0, stream>>>(
                a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.image,
                d_image_prev_, d_xprev_, d_dir_, static_cast<mixed_t>(alpha),
                ctx.runtime.host_box, n_);
        };
        try {
            ATV_CUDA_CHECK(cudaMemcpyAsync(h_ls_, d_ls_, 3 * sizeof(acc_t),
                                           cudaMemcpyDeviceToHost, stream));
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
            ls_alpha_ = static_cast<double>(h_ls_[0]);
            ls_gd_ = static_cast<double>(h_ls_[1]);
            ls_origin_energy_ = static_cast<double>(h_ls_[2]);
            if (!std::isfinite(ls_origin_energy_) || !std::isfinite(ls_gd_))
                throw std::runtime_error(
                    "CUDA LBFGS nonfinite line-search origin");
            if (!(ls_gd_ < 0)) {
                set_trial(0);
                ATV_CUDA_CHECK(cudaGetLastError());
                return;
            }
            for (int attempt = 0; attempt < max_ls_; ++attempt) {
                ctx.runtime.opt.reeval_forces(true);
                ATV_CUDA_CHECK(cudaMemcpyAsync(
                    h_ls_, &ctx.frame.scalars->energy, sizeof(double),
                    cudaMemcpyDeviceToHost, stream));
                ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
                const double energy = static_cast<double>(h_ls_[0]);
                if (!std::isfinite(energy))
                    throw std::runtime_error(
                        "CUDA LBFGS nonfinite trial energy");
                if (energy - ls_origin_energy_ <= ls_c1_ * ls_alpha_ * ls_gd_)
                    return; // The normal force phase evaluates this accepted point.
                if (ls_alpha_ <= ls_min_alpha_ || attempt + 1 == max_ls_)
                    break;
                ls_alpha_ = std::max(ls_min_alpha_, ls_alpha_ * ls_contract_);
                set_trial(ls_alpha_);
                ATV_CUDA_CHECK(cudaGetLastError());
            }
            throw std::runtime_error("CUDA LBFGS line search exhausted its "
                                     "trial budget or alpha floor");
        } catch (...) {
            // No failed candidate is returned. The outer driver discards device
            // results and the evaluation-history transaction on this exception.
            set_trial(0);
            cudaStreamSynchronize(stream);
            throw;
        }
    }

    // ── Search direction phase ───────────────────────────────────────
    auto& frame = ctx.frame;
    const int n = n_;
    const int n3 = n3_;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    int grid3 = cuda::grid_size(n3);
    int shmem = 0;

    const auto& a = frame.atoms;
    pack_xg<<<grid, block, 0, stream>>>(a.x, a.y, a.z, a.x_corr, a.y_corr,
                                        a.z_corr, a.fx, a.fy, a.fz, d_x_, d_g_,
                                        n);

    // ── History update (one 8-byte D2H — see file header) ────────────
    if (has_prev_) {
        const int idx = (n_stored_ < memory_) ? n_stored_ : oldest_;
        cudaMemsetAsync(d_ys_, 0, sizeof(acc_t), stream);
        sy_dot<<<grid, block, shmem, stream>>>(
            d_x_, d_g_, d_xprev_, d_gprev_,
            d_s_ + static_cast<size_t>(idx) * n3,
            d_y_ + static_cast<size_t>(idx) * n3, ctx.runtime.host_box, d_ys_,
            n);
        cudaMemcpyAsync(h_ys_, d_ys_, sizeof(acc_t), cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
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
    cudaMemcpyAsync(d_q_, d_g_, static_cast<size_t>(n3) * sizeof(mixed_t),
                    cudaMemcpyDeviceToDevice, stream);

    for (int k = n_stored_ - 1; k >= 0; --k) {
        const int idx = (oldest_ + k) % memory_;
        cudaMemsetAsync(d_dot_, 0, sizeof(acc_t), stream);
        dot3n<<<grid3, block, shmem, stream>>>(
            d_s_ + static_cast<size_t>(idx) * n3, d_q_, d_dot_, n3);
        axpy_backward<<<grid3, block, 0, stream>>>(
            d_q_, d_y_ + static_cast<size_t>(idx) * n3, d_dot_,
            static_cast<mixed_t>(rho_[idx]), d_alpha_ + k, n3);
    }

    if (n_stored_ > 0) {
        const int newest = (oldest_ + n_stored_ - 1) % memory_;
        cudaMemsetAsync(d_dot_, 0, sizeof(acc_t), stream);
        dot3n<<<grid3, block, shmem, stream>>>(
            d_y_ + static_cast<size_t>(newest) * n3,
            d_y_ + static_cast<size_t>(newest) * n3, d_dot_, n3);
        gamma_scale<<<grid3, block, 0, stream>>>(
            d_q_, d_dot_, static_cast<mixed_t>(rho_[newest]), n3);
    }

    for (int k = 0; k < n_stored_; ++k) {
        const int idx = (oldest_ + k) % memory_;
        cudaMemsetAsync(d_dot_, 0, sizeof(acc_t), stream);
        dot3n<<<grid3, block, shmem, stream>>>(
            d_y_ + static_cast<size_t>(idx) * n3, d_q_, d_dot_, n3);
        axpy_forward<<<grid3, block, 0, stream>>>(
            d_q_, d_s_ + static_cast<size_t>(idx) * n3, d_alpha_ + k, d_dot_,
            static_cast<mixed_t>(rho_[idx]), n3);
    }

    // ── Step application (α capped at 0.1 Angstrom max displacement) ─
    cudaMemsetAsync(d_dot_ + 1, 0, sizeof(acc_t), stream);
    max_abs3n<<<grid3, block, shmem, stream>>>(d_q_, d_dot_ + 1, n3);
    apply_step<<<grid, block, 0, stream>>>(
        a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, d_x_, d_g_, d_q_, d_dir_,
        d_xprev_, d_gprev_, a.image, d_image_prev_, d_dot_ + 1,
        static_cast<mixed_t>(alpha0_), d_ls_, frame.force_fault, a.tag,
        ctx.step, ctx.runtime.host_box, n);

    if (ls_enabled_) {
        // Origin scalars consumed by the following bounded line-search phase.
        cudaMemsetAsync(d_ls_ + 1, 0, sizeof(acc_t), stream);
        dot3n<<<grid3, block, shmem, stream>>>(d_g_, d_dir_, d_ls_ + 1, n3);
        // Preserve the actual origin energy on device, including the first
        // step after initial-force preparation. Fetch it with the trial metrics.
        cudaMemcpyAsync(d_ls_ + 2, &frame.scalars->energy, sizeof(acc_t),
                        cudaMemcpyDeviceToDevice, stream);
    }

    has_prev_ = true;
    ++step_;
}

void LBFGS::destroy(const DestroyContext& ctx) {
    cudaFree(d_x_);
    cudaFree(d_g_);
    cudaFree(d_xprev_);
    cudaFree(d_gprev_);
    cudaFree(d_q_);
    cudaFree(d_dir_);
    cudaFree(d_s_);
    cudaFree(d_y_);
    cudaFree(d_alpha_);
    cudaFree(d_image_prev_);
    d_image_prev_ = nullptr;
    cudaFree(d_dot_);
    cudaFree(d_ys_);
    cudaFree(d_ls_);
    cudaFreeHost(h_ys_);
    cudaFreeHost(h_ls_);
    d_x_ = d_g_ = d_xprev_ = d_gprev_ = d_q_ = d_dir_ = nullptr;
    d_s_ = d_y_ = d_alpha_ = nullptr;
    d_dot_ = d_ys_ = d_ls_ = nullptr;
    h_ys_ = h_ls_ = nullptr;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
