/// @file cuda/minimize/lbfgs.cu
/// @brief L-BFGS kernels: device-resident two-loop recursion +
///        multi-criteria convergence via the shared ConvCheck.

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/graph_ring.cuh"
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

/// Device ring of correction pairs: ring position p (0 = oldest) holds
/// buffer `slot[(oldest + p) % memory]` of the (memory + 1)-buffer s/y
/// arrays; `spare` is the staging buffer the next candidate pair is written
/// to, swapped into the ring only when it passes the curvature test.
struct Ring {
    int* state; ///< {n_stored, oldest, spare}.
    int* slot;  ///< Buffer of each ring position [memory].
    acc_t* rho; ///< 1 / (y·s) per buffer [memory + 1].
    acc_t* yy;  ///< y·y per buffer [memory + 1].
    int memory;
    __device__ int stored() const { return state[0]; }
    __device__ int buffer(int p) const { return slot[(state[1] + p) % memory]; }
};

/// Reduction slots of one iteration (zeroed by pack_xg).
enum DotSlot : int { kYs = 0, kYy = 1, kDirMax = 2, kFirstPass = 3 };

/// Sum `v` over the block into `out` (one atomic per block).
__device__ __forceinline__ void block_sum_into(acc_t v, acc_t* out) {
    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const acc_t sum = Reduce(reduction).Sum(v);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

/// Max |v| over the block into `out`.
__device__ __forceinline__ void block_max_into(acc_t v, acc_t* out) {
    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const acc_t m = Reduce(reduction).Reduce(v, ::cuda::maximum<>{});
    if (threadIdx.x == 0)
        ATV::cuda::atomic_max_nonneg(out, m);
}

/// @brief Pack reconstructed frame positions + decoded gradient (g = −F)
///        into planar mixed_t buffers; the first thread clears the
///        iteration's reduction slots and parks the origin energy.
__global__ void
pack_xg(const pos_t* __restrict__ x, const pos_t* __restrict__ y,
        const pos_t* __restrict__ z, const pos_t* __restrict__ xc,
        const pos_t* __restrict__ yc, const pos_t* __restrict__ zc,
        const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
        const fixed_t* __restrict__ fz, mixed_t* __restrict__ xp,
        mixed_t* __restrict__ gp, acc_t* __restrict__ dots, int n_dots,
        acc_t* __restrict__ ls, const acc_t* __restrict__ energy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        for (int k = 0; k < n_dots; ++k)
            dots[k] = acc_t(0);
        ls[1] = acc_t(0);
        ls[2] = *energy; // line-search origin energy
    }
    if (i >= n)
        return;
    xp[i] = load_position(x, xc, i);
    xp[n + i] = load_position(y, yc, i);
    xp[2 * n + i] = load_position(z, zc, i);
    gp[i] = -decode_force<mixed_t>(fx[i]);
    gp[n + i] = -decode_force<mixed_t>(fy[i]);
    gp[2 * n + i] = -decode_force<mixed_t>(fz[i]);
}

/// @brief Candidate pair s = x − x_prev (min image), y = g − g_prev into the
///        spare buffers; y·s and y·y in one block collective.
__global__ void
sy_dot(const mixed_t* __restrict__ xc, const mixed_t* __restrict__ gc,
       const mixed_t* __restrict__ x_prev, const mixed_t* __restrict__ g_prev,
       mixed_t* __restrict__ s_all, mixed_t* __restrict__ y_all, Ring ring,
       cuda::Box box, acc_t* __restrict__ dots, int n) {
    box = box.current();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t n3 = 3 * static_cast<std::size_t>(n);
    mixed_t* s_slot = s_all + ring.state[2] * n3;
    mixed_t* y_slot = y_all + ring.state[2] * n3;
    ::cuda::std::array<acc_t, 2> mine = {acc_t(0), acc_t(0)};
    if (i < n) {
        mixed_t sx = xc[i] - x_prev[i];
        mixed_t sy = xc[n + i] - x_prev[n + i];
        mixed_t sz = xc[2 * n + i] - x_prev[2 * n + i];
        box.min_image(sx, sy, sz);
        const mixed_t sv[3] = {sx, sy, sz};
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const mixed_t yv = gc[c * n + i] - g_prev[c * n + i];
            s_slot[c * n + i] = sv[c];
            y_slot[c * n + i] = yv;
            mine[0] += acc_t(yv) * acc_t(sv[c]);
            mine[1] += acc_t(yv) * acc_t(yv);
        }
    }
    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 2>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(mine, SumComponents<acc_t, 2>{});
    if (threadIdx.x == 0) {
        atomicAdd(&dots[kYs], out[0]);
        atomicAdd(&dots[kYy], out[1]);
    }
}

/// @brief Curvature test on the device: an accepted spare pair joins the
///        ring (evicting the oldest when full, whose buffer becomes the
///        spare); a rejected one leaves the stored history intact.
__global__ void ring_accept(Ring ring, const acc_t* __restrict__ dots) {
    const acc_t ys = dots[kYs];
    if (!(ys > acc_t(1e-10)))
        return;
    int& n_stored = ring.state[0];
    int& oldest = ring.state[1];
    int& spare = ring.state[2];
    const int fresh = spare;
    ring.rho[fresh] = acc_t(1) / ys;
    ring.yy[fresh] = dots[kYy];
    if (n_stored < ring.memory) {
        ring.slot[(oldest + n_stored) % ring.memory] = fresh;
        ++n_stored;
        spare = n_stored; // buffers 0..n_stored-1 are in use
    } else {
        spare = ring.slot[oldest];
        ring.slot[oldest] = fresh;
        oldest = (oldest + 1) % ring.memory;
    }
}

/// @brief q = g, plus the first backward inner product s_{n−1}·g (or, with
///        no stored pair, the step cap max|g|).
__global__ void q_init(const mixed_t* __restrict__ g, mixed_t* __restrict__ q,
                       const mixed_t* __restrict__ s_all, Ring ring,
                       acc_t* __restrict__ dots, int n3) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int stored = ring.stored();
    const mixed_t gj = j < n3 ? g[j] : mixed_t(0);
    if (j < n3)
        q[j] = gj;
    if (stored > 0) {
        const mixed_t* s = s_all + std::size_t(ring.buffer(stored - 1)) * n3;
        block_sum_into(j < n3 ? acc_t(s[j]) * acc_t(gj) : acc_t(0),
                       &dots[kFirstPass + stored - 1]);
    } else {
        block_max_into(fabs(acc_t(gj)), &dots[kDirMax]);
    }
}

/// @brief Backward pass k: a_k = ρ_k (s_k·q), q −= a_k y_k, fused with the
///        next inner product s_{k−1}·q (or y_0·q after the last pass).
__global__ void backward_pass(mixed_t* __restrict__ q,
                              const mixed_t* __restrict__ s_all,
                              const mixed_t* __restrict__ y_all, Ring ring,
                              mixed_t* __restrict__ alpha, acc_t* dots, int k,
                              int n3) {
    if (k >= ring.stored())
        return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = ring.buffer(k);
    const mixed_t a = mixed_t(ring.rho[b] * dots[kFirstPass + k]);
    if (j == 0)
        alpha[k] = a;
    const mixed_t* yk = y_all + std::size_t(b) * n3;
    mixed_t qj = 0;
    if (j < n3) {
        qj = q[j] - a * yk[j];
        q[j] = qj;
    }
    // Next: s_{k-1}·q into its backward slot, or y_0·q into forward slot 0.
    const mixed_t* next = k > 0 ? s_all + std::size_t(ring.buffer(k - 1)) * n3
                                : y_all + std::size_t(ring.buffer(0)) * n3;
    acc_t* out =
        k > 0 ? &dots[kFirstPass + k - 1] : &dots[kFirstPass + ring.memory];
    block_sum_into(j < n3 ? acc_t(next[j]) * acc_t(qj) : acc_t(0), out);
}

/// @brief Forward pass k: r += s_k (a_k − ρ_k y_k·r), with the initial
///        Hessian scaling r = γ q folded into pass 0 (γ = s·y / y·y of the
///        newest pair, 1 when y·y underflows), fused with the next inner
///        product y_{k+1}·r — or, after the last pass, the step cap max|r|.
__global__ void forward_pass(mixed_t* __restrict__ q,
                             const mixed_t* __restrict__ s_all,
                             const mixed_t* __restrict__ y_all, Ring ring,
                             const mixed_t* __restrict__ alpha, acc_t* dots,
                             int k, int n3) {
    const int stored = ring.stored();
    if (k >= stored)
        return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = ring.buffer(k);
    mixed_t gamma = 1;
    if (k == 0) {
        const int newest = ring.buffer(stored - 1);
        const acc_t y2 = ring.yy[newest];
        if (y2 > acc_t(1e-30))
            gamma = mixed_t(acc_t(1) / (ring.rho[newest] * y2));
    }
    const acc_t yr =
        dots[kFirstPass + ring.memory + k] * (k == 0 ? acc_t(gamma) : acc_t(1));
    const mixed_t coef = alpha[k] - mixed_t(ring.rho[b] * yr);
    const mixed_t* sk = s_all + std::size_t(b) * n3;
    mixed_t rj = 0;
    if (j < n3) {
        rj = (k == 0 ? gamma * q[j] : q[j]) + sk[j] * coef;
        q[j] = rj;
    }
    if (k + 1 < stored) {
        const mixed_t* yn = y_all + std::size_t(ring.buffer(k + 1)) * n3;
        block_sum_into(j < n3 ? acc_t(yn[j]) * acc_t(rj) : acc_t(0),
                       &dots[kFirstPass + ring.memory + k + 1]);
    } else {
        block_max_into(fabs(acc_t(rj)), &dots[kDirMax]);
    }
}

/// @brief Apply the L-BFGS step: direction = −r, α capped so no atom
///        moves more than 0.1 Angstrom; saves prev state, moves, wraps.
///
/// Thread (0,0) parks the applied α in ls[0]; every block adds its g·d to
/// ls[1] for the line search. The direction was built from the decoded
/// forces of this Frame, so a faulted force is never consumed: the step
/// (and its history save) is skipped.
__global__ void
apply_step(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
           pos_t* __restrict__ xcorr, pos_t* __restrict__ ycorr,
           pos_t* __restrict__ zcorr, const mixed_t* __restrict__ xc,
           const mixed_t* __restrict__ gc, const mixed_t* __restrict__ r,
           mixed_t* __restrict__ dir, mixed_t* __restrict__ x_prev,
           mixed_t* __restrict__ g_prev, ImageInt* __restrict__ image,
           ImageInt* __restrict__ image_prev, const acc_t* __restrict__ dir_max,
           mixed_t alpha0, acc_t* __restrict__ ls, bool line_search,
           ForceFault* fault, const std::int64_t* __restrict__ tag,
           const long long* step_ptr, long long step, cuda::Box box, int n) {
    if (!forces_consumable(fault, tag, step_ptr, step))
        return;
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const mixed_t dmax = mixed_t(fmax(*dir_max, acc_t(1e-20)));
    const mixed_t alpha = fmin(alpha0, mixed_t(0.1) / dmax);
    if (i == 0)
        ls[0] = acc_t(alpha);

    acc_t gd = 0;
    if (i < n) {
        mixed_t pos[3];
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const mixed_t d = -r[c * n + i];
            dir[c * n + i] = d;
            x_prev[c * n + i] = xc[c * n + i];
            g_prev[c * n + i] = gc[c * n + i];
            pos[c] = xc[c * n + i] + alpha * d;
            gd += acc_t(gc[c * n + i]) * acc_t(d);
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
    if (line_search)
        block_sum_into(gd, &ls[1]);
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
    // memory + 1 pair buffers: the ring plus the staging spare.
    cudaMalloc(&d_s_, static_cast<size_t>(memory_ + 1) * bytes3n);
    cudaMalloc(&d_y_, static_cast<size_t>(memory_ + 1) * bytes3n);
    cudaMalloc(&d_alpha_, memory_ * sizeof(mixed_t));
    cudaMalloc(&d_image_prev_, static_cast<size_t>(n3_) * sizeof(ImageInt));
    cudaMemset(d_image_prev_, 0, static_cast<size_t>(n3_) * sizeof(ImageInt));
    n_dots_ = kFirstPass + 2 * memory_;
    cudaMalloc(&d_dot_, n_dots_ * sizeof(acc_t));
    cudaMalloc(&d_ring_state_, (3 + memory_) * sizeof(int));
    cudaMemset(d_ring_state_, 0, (3 + memory_) * sizeof(int));
    cudaMalloc(&d_rho_, 2 * (memory_ + 1) * sizeof(acc_t));
    cudaMemset(d_rho_, 0, 2 * (memory_ + 1) * sizeof(acc_t));
    cudaMalloc(&d_ls_, 3 * sizeof(acc_t));
    cudaMemset(d_xprev_, 0, bytes3n);
    cudaMemset(d_gprev_, 0, bytes3n);
    cudaMemset(d_dir_, 0, bytes3n);
    cudaMemset(d_ls_, 0, 3 * sizeof(acc_t));
    cudaMallocHost(&h_ls_, 3 * sizeof(acc_t));
    for (auto& g : graphs_)
        g.reset();

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
    // One fixed launch sequence per history state (the device decides the
    // curvature test and the stored count), replayed as a graph: no
    // device-to-host exchange in the iteration.
    auto& frame = ctx.frame;
    const int n = n_;
    const int n3 = n3_;
    const int block = cuda::kDefaultBlock;
    const int grid = cuda::grid_size(n);
    const int grid3 = cuda::grid_size(n3);
    const auto& a = frame.atoms;
    const Ring ring{d_ring_state_, d_ring_state_ + 3, d_rho_,
                    d_rho_ + memory_ + 1, memory_};
    const bool with_pair = has_prev_;
    auto enqueue = [&] {
        pack_xg<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.fx, a.fy, a.fz, d_x_,
            d_g_, d_dot_, n_dots_, d_ls_, &frame.scalars->energy, n);
        if (with_pair) {
            sy_dot<<<grid, block, 0, stream>>>(d_x_, d_g_, d_xprev_, d_gprev_,
                                               d_s_, d_y_, ring,
                                               ctx.runtime.host_box, d_dot_, n);
            ring_accept<<<1, 1, 0, stream>>>(ring, d_dot_);
        }
        q_init<<<grid3, block, 0, stream>>>(d_g_, d_q_, d_s_, ring, d_dot_, n3);
        for (int k = memory_ - 1; k >= 0; --k)
            backward_pass<<<grid3, block, 0, stream>>>(d_q_, d_s_, d_y_, ring,
                                                       d_alpha_, d_dot_, k, n3);
        for (int k = 0; k < memory_; ++k)
            forward_pass<<<grid3, block, 0, stream>>>(d_q_, d_s_, d_y_, ring,
                                                      d_alpha_, d_dot_, k, n3);
        apply_step<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, d_x_, d_g_, d_q_,
            d_dir_, d_xprev_, d_gprev_, a.image, d_image_prev_,
            d_dot_ + kDirMax, static_cast<mixed_t>(alpha0_), d_ls_, ls_enabled_,
            frame.force_fault, a.tag, ctx.runtime.device_step,
            static_cast<long long>(ctx.step), ctx.runtime.host_box, n);
    };
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (stream != nullptr)
        ATV_CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    // The legacy default stream cannot be captured (component tests drive
    // the optimizer on it); an enclosing capture records the kernels itself.
    if (stream == nullptr || capturing != cudaStreamCaptureStatusNone) {
        enqueue();
    } else {
        GraphRing& ring_graph = graphs_[with_pair ? 1 : 0];
        if (!ring_graph.ready())
            ring_graph.build([&]() -> cudaGraph_t {
                cudaGraph_t graph = nullptr;
                ATV_CUDA_CHECK(cudaStreamBeginCapture(
                    stream, cudaStreamCaptureModeThreadLocal));
                try {
                    enqueue();
                } catch (...) {
                    cudaStreamEndCapture(stream, &graph);
                    if (graph)
                        cudaGraphDestroy(graph);
                    throw;
                }
                ATV_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                return graph;
            });
        ATV_CUDA_CHECK(ring_graph.launch(stream));
    }
    ATV_CUDA_CHECK(cudaGetLastError());

    has_prev_ = true;
    ++step_;
}

void LBFGS::destroy(const DestroyContext& ctx) {
    for (auto& g : graphs_)
        g.reset();
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
    cudaFree(d_ring_state_);
    cudaFree(d_rho_);
    cudaFree(d_ls_);
    cudaFreeHost(h_ls_);
    d_x_ = d_g_ = d_xprev_ = d_gprev_ = d_q_ = d_dir_ = nullptr;
    d_s_ = d_y_ = d_alpha_ = nullptr;
    d_dot_ = d_rho_ = d_ls_ = nullptr;
    d_ring_state_ = nullptr;
    h_ls_ = nullptr;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
