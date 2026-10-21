/// @file cuda/dynamics/nose_hoover_npt.cu
/// @brief CUDA MTK NPT integrator implementation. Mirrors the CPU
///        sibling cpu::md::NoseHooverNPT line for line — see that file
///        for the canonical algorithm documentation block.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

namespace {

/// @brief Smooth `(e^x - 1)/x` with a Taylor branch around 0 to avoid
///        catastrophic cancellation. Identical to the CPU `expm1c`.
__host__ inline double expm1c(double x) {
    if (std::fabs(x) < 1e-4) {
        // 1 + x/2 + x²/6 — accurate to O(x³)
        return 1.0 + x * (0.5 + x * (1.0 / 6.0));
    }
    return std::expm1(x) / x;
}

constexpr double kEpsDtMaxFraction = 0.005;

// ── Per-atom kernels ────────────────────────────────────────────────

__global__ void scale_vel_uniform_k(motion_t* __restrict__ vx,
                                    motion_t* __restrict__ vy,
                                    motion_t* __restrict__ vz, motion_t scale,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    vx[i] = static_cast<motion_t>(static_cast<motion_t>(vx[i]) * scale);
    vy[i] = static_cast<motion_t>(static_cast<motion_t>(vy[i]) * scale);
    vz[i] = static_cast<motion_t>(static_cast<motion_t>(vz[i]) * scale);
}

__global__ void scale_vel_per_axis_k(motion_t* __restrict__ vx,
                                     motion_t* __restrict__ vy,
                                     motion_t* __restrict__ vz, motion_t sx,
                                     motion_t sy, motion_t sz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    vx[i] = static_cast<motion_t>(static_cast<motion_t>(vx[i]) * sx);
    vy[i] = static_cast<motion_t>(static_cast<motion_t>(vy[i]) * sy);
    vz[i] = static_cast<motion_t>(static_cast<motion_t>(vz[i]) * sz);
}

__global__ void
kick_v_k(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
         motion_t* __restrict__ vz, const real_t* __restrict__ fx,
         const real_t* __restrict__ fy, const real_t* __restrict__ fz,
         const motion_t* __restrict__ mass, motion_t half_dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t dtfm =
        half_dt * ATV::units::force_to_accel / static_cast<motion_t>(mass[i]);
    vx[i] = static_cast<motion_t>(static_cast<motion_t>(vx[i])
                                  + dtfm * static_cast<motion_t>(fx[i]));
    vy[i] = static_cast<motion_t>(static_cast<motion_t>(vy[i])
                                  + dtfm * static_cast<motion_t>(fy[i]));
    vz[i] = static_cast<motion_t>(static_cast<motion_t>(vz[i])
                                  + dtfm * static_cast<motion_t>(fz[i]));
}

__global__ void
propagate_x_k(real_t* __restrict__ x, real_t* __restrict__ y,
              real_t* __restrict__ z, real_t* __restrict__ xc,
              real_t* __restrict__ yc, real_t* __restrict__ zc,
              std::int8_t* __restrict__ image, const motion_t* __restrict__ vx,
              const motion_t* __restrict__ vy, const motion_t* __restrict__ vz,
              motion_t ax, motion_t ay, motion_t az, motion_t dx, motion_t dy,
              motion_t dz, motion_t dt, prec_t* __restrict__ box,
              cuda::MotionBox new_box, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Closed-form per-axis MTK position propagator (matches CPU
        // propagate_x_and_box).
        wrap_position(x, y, z, xc, yc, zc, image, i, new_box,
                      load_position(x, xc, i) * ax + vx[i] * dt * dx,
                      load_position(y, yc, i) * ay + vy[i] * dt * dy,
                      load_position(z, zc, i) * az + vz[i] * dt * dz);
    }
    // Thread 0 of block 0 writes the new box diagonal to device, so
    // downstream consumers reading raw `frame.box` see the updated
    // dimensions in the same step.
    if (i == 0) {
        box[0] = static_cast<prec_t>(new_box.cell[0]);
        box[4] = static_cast<prec_t>(new_box.cell[4]);
        box[8] = static_cast<prec_t>(new_box.cell[8]);
    }
}

// ── KE reductions ───────────────────────────────────────────────────
//
// reduce_ke_axes_k writes the first 4 slots of the consolidated
// NPT scratch buffer (`State::d_npt_scratch_`, size kNptScratchSize):
//   d_npt_scratch_[0] = Σ m·(vx² + vy² + vz²)   (global ke pre-factor)
//   d_npt_scratch_[1] = Σ m·vx²                  (per-axis x)
//   d_npt_scratch_[2] = Σ m·vy²
//   d_npt_scratch_[3] = Σ m·vz²
// Slots [4..6] hold the stress[0..2] mirror; [7] the energy mirror —
// both filled by `gather_npt_scalars_k` so a single async D2H pulls
// all NPT scalars in one defer-stream round-trip.
// Caller multiplies the KE slots by 0.5·ATV::units::mass_v2_to_energy to get eV.

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void reduce_ke_axes_k(
    const motion_t* __restrict__ vx, const motion_t* __restrict__ vy,
    const motion_t* __restrict__ vz, const motion_t* __restrict__ mass,
    double* __restrict__ buf, int n) {
    extern __shared__ double warp_scratch[]; // per-warp staging cells
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double mx = 0.0, my = 0.0, mz = 0.0;
    if (i < n) {
        double m = static_cast<double>(mass[i]);
        double vxi = static_cast<double>(vx[i]);
        double vyi = static_cast<double>(vy[i]);
        double vzi = static_cast<double>(vz[i]);
        mx = m * vxi * vxi;
        my = m * vyi * vyi;
        mz = m * vzi * vzi;
    }
    // Warp-shuffle reduction via the shared helper. Drops shared-mem
    // from `3 * blockDim * 8` = ~6 KiB/block to (blockDim/32) * 8 ≈
    // 64 bytes/block — opens up occupancy on the 228 KiB/SM budget
    // (cuda-md-review-cleanup-05-reductions-and-atomics).
    double my_vals[3] = {mx, my, mz};
    double out[3];
    ATV::cuda::block_reduce_sum_n<3>(my_vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        double bx = out[0], by = out[1], bz = out[2];
        atomicAdd(&buf[1], bx);
        atomicAdd(&buf[2], by);
        atomicAdd(&buf[3], bz);
        atomicAdd(&buf[0], bx + by + bz);
    }
}

/// @brief Mirror stress[0..2] + energy device scalars into the
///        8-double consolidated NPT scratch buffer (slots [4..7]).
///
/// Pairs with `reduce_ke_axes_k` (which fills [0..3]). One thread is
/// sufficient — the four mirrored values are tiny and contiguous —
/// and pairing the two kernels behind a single async D2H pulls all
/// NPT scalars in one defer-stream round-trip
/// (cuda-md-review-cleanup-03-launch-purity).
__global__ void gather_npt_scalars_k(const double* __restrict__ stress,
                                     const double* __restrict__ energy,
                                     double* __restrict__ scratch /*8*/) {
    if (threadIdx.x == 0) {
        scratch[4] = stress[0];
        scratch[5] = stress[1];
        scratch[6] = stress[2];
        scratch[7] = *energy;
    }
}

/// @brief Snapshot box diagonal (box[0], box[4], box[8]) into a
///        3-double scratch buffer for async readback.
///
/// Used by `propagate_x_and_box_` to refresh the host-cached box
/// dimensions after `cuda::md::Deform` (order -100, before NPT at
/// order 0) may have rescaled the box in-place. Single thread is
/// sufficient — three loads + three stores
/// (cuda-md-review-cleanup-03-launch-purity).
__global__ void gather_box_diag_k(const prec_t* __restrict__ box,
                                  double* __restrict__ scratch /*3*/) {
    if (threadIdx.x == 0) {
        scratch[0] = static_cast<double>(box[0]);
        scratch[1] = static_cast<double>(box[4]);
        scratch[2] = static_cast<double>(box[8]);
    }
}

} // anonymous namespace

// ── ctor + lifecycle ────────────────────────────────────────────────

NoseHooverNPT::NoseHooverNPT(double T_target, double tau_T, double P_target,
                             double tau_P, double dt, double drag, int mtchain,
                             int mpchain, std::array<bool, 3> couple_axis,
                             double T_stop)
    : T_target_(T_target), T_stop_(T_stop < 0 ? T_target : T_stop),
      tau_T_(tau_T), P_target_(P_target), tau_P_(tau_P), dt_(dt), drag_(drag),
      mtchain_(mtchain), mpchain_(mpchain), couple_axis_(couple_axis),
      T_live_(T_target) {
    pdim_ = 0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            ++pdim_;
}

void NoseHooverNPT::build(const BuildContext& ctx) {
    n_atoms_ = ctx.frame.n_atoms();
    n_dof_ = 3 * n_atoms_ - 3;
    kT_ = ATV::units::kB_ev_per_k * T_target_;

    Q_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    Q_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_t_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_t_dot_.assign(static_cast<std::size_t>(mtchain_), 0.0);
    eta_b_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eta_b_dot_.assign(static_cast<std::size_t>(mpchain_), 0.0);
    eps_dot_.fill(0.0);

    const double tau_T2 = tau_T_ * tau_T_;
    const double tau_P2 = tau_P_ * tau_P_;
    Q_t_[0] = static_cast<double>(n_dof_) * kT_ * tau_T2;
    for (int k = 1; k < mtchain_; ++k)
        Q_t_[k] = kT_ * tau_T2;
    Q_b_[0] = kT_ * tau_P2;
    for (int k = 1; k < mpchain_; ++k)
        Q_b_[k] = kT_ * tau_P2;
    W_ = (static_cast<double>(n_dof_) + 3.0) * kT_ * tau_P2;

    tdrag_factor_ = 1.0 - drag_ * dt_ / tau_T_;
    pdrag_factor_ = 1.0 - drag_ * dt_ / tau_P_;

    // Velocities can be initialized by an earlier component after build().
    // Sample them at this component's first integration launch.
    KE_global_ = 0.0;
    kinetic_initialized_ = false;

    // Cache box diagonal (orthorhombic only — same restriction as
    // cuda::md::Barostat / VelocityVerlet wrap).
    prec_t host_box[9];
    ATV_CUDA_CHECK(cudaMemcpy(host_box, ctx.frame.box, 9 * sizeof(prec_t),
                              cudaMemcpyDeviceToHost));

    // Enforce the restriction instead of documenting it. Only the diagonal
    // is read below and only the diagonal is written back, so a tilted cell
    // would have its off-diagonals silently dropped on the first barostat
    // step — and the CPU twin, which guards, would then disagree with CUDA
    // on identical input with nothing to point at.
    for (int i = 0; i < 9; ++i) {
        if (i == 0 || i == 4 || i == 8)
            continue;
        if (host_box[i] != prec_t(0))
            throw std::runtime_error(
                "cuda::md::NoseHooverNPT: triclinic cells are not supported "
                "(couple_axis is per diagonal axis; off-diagonals would be "
                "dropped). Use an orthorhombic cell, or run this on the CPU "
                "backend.");
    }

    box_lx_ = static_cast<double>(host_box[0]);
    box_ly_ = static_cast<double>(host_box[4]);
    box_lz_ = static_cast<double>(host_box[8]);

    // Consolidated NPT scratch: 8 doubles
    //   [0..3] = KE_axes reduction output (existing reduce_ke_axes_k)
    //   [4..6] = stress[0..2] mirror (filled by gather_npt_scalars_k)
    //   [7]    = energy mirror
    double* d_npt = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&d_npt, kNptScratchSize * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemset(d_npt, 0, kNptScratchSize * sizeof(double)));
    d_npt_scratch_ = d_npt;

    // Box-diagonal device scratch (3 doubles): refreshed by Deform-aware
    // path in propagate_x_and_box_.
    double* d_box = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&d_box, kBoxDiagSize * sizeof(double)));
    d_box_diag_ = d_box;

    // Pinned host scratch + events. Routed through defer_stream so the
    // main compute stream is never drained for KE / stress / box reads
    // (cuda-md-review-cleanup-03-launch-purity).
    ATV_CUDA_CHECK(
        cudaMallocHost(&h_npt_scratch_, kNptScratchSize * sizeof(double)));
    for (int i = 0; i < kNptScratchSize; ++i)
        h_npt_scratch_[i] = 0.0;
    ATV_CUDA_CHECK(
        cudaMallocHost(&h_box_diag_pinned_, kBoxDiagSize * sizeof(double)));
    h_box_diag_pinned_[0] = box_lx_;
    h_box_diag_pinned_[1] = box_ly_;
    h_box_diag_pinned_[2] = box_lz_;
    ATV_CUDA_CHECK(
        cudaMallocHost(&h_kinetic_conserved_pinned_,
                       kKineticConservedPinnedSize * sizeof(double)));
    h_kinetic_conserved_pinned_[0] = KE_global_;
    h_kinetic_conserved_pinned_[1] = 0.0;

    cudaEvent_t ev_npt = nullptr;
    ATV_CUDA_CHECK(cudaEventCreateWithFlags(&ev_npt, cudaEventDisableTiming));
    npt_scratch_event_ = ev_npt;
    cudaEvent_t ev_box = nullptr;
    ATV_CUDA_CHECK(cudaEventCreateWithFlags(&ev_box, cudaEventDisableTiming));
    box_diag_event_ = ev_box;
}

void NoseHooverNPT::destroy(const DestroyContext& /*ctx*/) {
    if (d_npt_scratch_) {
        ATV_CUDA_CHECK(cudaFree(d_npt_scratch_));
        d_npt_scratch_ = nullptr;
    }
    if (d_box_diag_) {
        ATV_CUDA_CHECK(cudaFree(d_box_diag_));
        d_box_diag_ = nullptr;
    }
    if (h_npt_scratch_) {
        ATV_CUDA_CHECK(cudaFreeHost(h_npt_scratch_));
        h_npt_scratch_ = nullptr;
    }
    if (h_box_diag_pinned_) {
        ATV_CUDA_CHECK(cudaFreeHost(h_box_diag_pinned_));
        h_box_diag_pinned_ = nullptr;
    }
    if (h_kinetic_conserved_pinned_) {
        ATV_CUDA_CHECK(cudaFreeHost(h_kinetic_conserved_pinned_));
        h_kinetic_conserved_pinned_ = nullptr;
    }
    if (npt_scratch_event_) {
        ATV_CUDA_CHECK(
            cudaEventDestroy(static_cast<cudaEvent_t>(npt_scratch_event_)));
        npt_scratch_event_ = nullptr;
    }
    if (box_diag_event_) {
        ATV_CUDA_CHECK(
            cudaEventDestroy(static_cast<cudaEvent_t>(box_diag_event_)));
        box_diag_event_ = nullptr;
    }
}

// ── Consolidated NPT scalar readback ────────────────────────────────
//
// Replaces the old per-call `cudaStreamSynchronize(stream) +
// cudaMemcpy(D2H)` patterns in compute_global_KE_ / update_eps_dot_ /
// extended_hamiltonian_ (CPU sibling has a separate compute_axis_KE_;
// CUDA folds the per-axis path into refresh_npt_scalars_). Issues one
// reduction
// kernel + one gather kernel + one async 8-double D2H on
// `defer_stream` (ordered after main_stream via event), then a host
// wait on the data-ready event. The main compute stream is never
// drained — link 03 launch-purity contract
// (cuda-md-review-cleanup-03-launch-purity).
//
// All four host-side scalars (ke_total, ke_axes[3], stress_diag[3],
// E_pot) are populated in one round-trip; subsequent reads within the
// same NPT phase reuse the host cache without re-issuing the kernel.
void NoseHooverNPT::refresh_npt_scalars_(const LaunchContext& ctx) {
    auto main_stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto defer_stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    auto* d_npt = static_cast<double*>(d_npt_scratch_);

    // Order defer_stream after all pending main_stream work (the
    // velocity / stress / energy values are produced by main_stream
    // kernels — Virial, ZeroEnergy, integrator kicks).
    auto event = static_cast<cudaEvent_t>(npt_scratch_event_);
    ATV_CUDA_CHECK(cudaEventRecord(event, main_stream));
    ATV_CUDA_CHECK(cudaStreamWaitEvent(defer_stream, event, 0));

    // Zero the 4 KE slots + the 4 mirror slots (mirror slots are
    // overwritten by gather_npt_scalars_k but cheap to clear).
    ATV_CUDA_CHECK(cudaMemsetAsync(d_npt, 0, kNptScratchSize * sizeof(double),
                                   defer_stream));

    // KE reduction → d_npt[0..3]
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    size_t smem = cuda::warps_per_block(block) * sizeof(double);
    reduce_ke_axes_k<<<grid, block, smem, defer_stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass, d_npt, n);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Mirror stress[0..2] + energy → d_npt[4..7]
    gather_npt_scalars_k<<<1, 1, 0, defer_stream>>>(&f.scalars->stress[0],
                                                    &f.scalars->energy, d_npt);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Async D2H to pinned + event record. Host-block on event.
    ATV_CUDA_CHECK(cudaMemcpyAsync(h_npt_scratch_, d_npt,
                                   kNptScratchSize * sizeof(double),
                                   cudaMemcpyDeviceToHost, defer_stream));
    ATV_CUDA_CHECK(cudaEventRecord(event, defer_stream));
    ATV_CUDA_CHECK(
        cudaEventSynchronize(event)); // host-thread wait; no stream sync
}

double NoseHooverNPT::compute_global_KE_(const LaunchContext& ctx) {
    refresh_npt_scalars_(ctx);
    return 0.5 * ATV::units::mass_v2_to_energy * h_npt_scratch_[0];
}

void NoseHooverNPT::refresh_box_diag_(const LaunchContext& ctx) {
    auto main_stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto defer_stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    auto& f = ctx.frame;
    auto* d_box = static_cast<double*>(d_box_diag_);

    // Order defer_stream after Deform's box mutation (which runs on
    // main_stream at order -100).
    auto event = static_cast<cudaEvent_t>(box_diag_event_);
    ATV_CUDA_CHECK(cudaEventRecord(event, main_stream));
    ATV_CUDA_CHECK(cudaStreamWaitEvent(defer_stream, event, 0));

    // Single-thread gather + async D2H + event-sync.
    gather_box_diag_k<<<1, 1, 0, defer_stream>>>(f.box, d_box);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpyAsync(h_box_diag_pinned_, d_box,
                                   kBoxDiagSize * sizeof(double),
                                   cudaMemcpyDeviceToHost, defer_stream));
    ATV_CUDA_CHECK(cudaEventRecord(event, defer_stream));
    ATV_CUDA_CHECK(
        cudaEventSynchronize(event)); // host-thread wait; no stream sync
}

// ── Temperature ramp ────────────────────────────────────────────────

void NoseHooverNPT::compute_temp_target_(int step) {
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        if (delta < 0.0)
            delta = 0.0;
        if (delta > 1.0)
            delta = 1.0;
    }
    T_live_ = T_target_ + delta * (T_stop_ - T_target_);
    kT_ = ATV::units::kB_ev_per_k * T_live_;
}

// ── NHC chain helpers ───────────────────────────────────────────────

void NoseHooverNPT::nhc_atoms_half_(const LaunchContext& ctx, double half_dt) {
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    const int M = mtchain_;

    auto G_t = [&](int k) -> double {
        if (k == 0)
            return (2.0 * KE_global_ - n_dof_ * kT_) / Q_t_[0];
        return (Q_t_[k - 1] * eta_t_dot_[k - 1] * eta_t_dot_[k - 1] - kT_)
               / Q_t_[k];
    };

    // Backward pass with per-link drag.
    for (int k = M - 1; k >= 0; --k) {
        double expfac =
            (k < M - 1) ? std::exp(-eta_t_dot_[k + 1] * eighth_dt) : 1.0;
        eta_t_dot_[k] *= expfac;
        eta_t_dot_[k] += G_t(k) * quarter_dt;
        eta_t_dot_[k] *= tdrag_factor_;
        eta_t_dot_[k] *= expfac;
    }

    // Scale atom velocities by the innermost thermostat momentum (GPU kernel).
    double scale = std::exp(-eta_t_dot_[0] * half_dt);
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    scale_vel_uniform_k<<<grid, block, 0, stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, static_cast<motion_t>(scale), n);
    ATV_CUDA_CHECK(cudaGetLastError());
    KE_global_ *= scale * scale;

    for (int k = 0; k < M; ++k)
        eta_t_[k] += eta_t_dot_[k] * half_dt;

    // Forward pass, no drag.
    for (int k = 0; k < M; ++k) {
        double expfac =
            (k < M - 1) ? std::exp(-eta_t_dot_[k + 1] * eighth_dt) : 1.0;
        eta_t_dot_[k] *= expfac;
        eta_t_dot_[k] += G_t(k) * quarter_dt;
        eta_t_dot_[k] *= expfac;
    }
}

void NoseHooverNPT::nhc_baro_half_(double half_dt) {
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    const int M = mpchain_;
    const double W_axis = W_ / 3.0;

    auto baro_2ke = [&]() {
        double s = 0.0;
        for (int a = 0; a < 3; ++a)
            if (couple_axis_[a])
                s += W_axis * eps_dot_[a] * eps_dot_[a];
        return s;
    };
    auto G_b = [&](int k) -> double {
        if (k == 0)
            return (baro_2ke() - kT_) / Q_b_[0];
        return (Q_b_[k - 1] * eta_b_dot_[k - 1] * eta_b_dot_[k - 1] - kT_)
               / Q_b_[k];
    };

    // Backward pass with per-link drag.
    for (int k = M - 1; k >= 0; --k) {
        double expfac =
            (k < M - 1) ? std::exp(-eta_b_dot_[k + 1] * eighth_dt) : 1.0;
        eta_b_dot_[k] *= expfac;
        eta_b_dot_[k] += G_b(k) * quarter_dt;
        eta_b_dot_[k] *= pdrag_factor_;
        eta_b_dot_[k] *= expfac;
    }

    double baro_scale = std::exp(-eta_b_dot_[0] * half_dt);
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            eps_dot_[a] *= baro_scale;

    for (int k = 0; k < M; ++k)
        eta_b_[k] += eta_b_dot_[k] * half_dt;

    // Forward pass, no drag.
    for (int k = 0; k < M; ++k) {
        double expfac =
            (k < M - 1) ? std::exp(-eta_b_dot_[k + 1] * eighth_dt) : 1.0;
        eta_b_dot_[k] *= expfac;
        eta_b_dot_[k] += G_b(k) * quarter_dt;
        eta_b_dot_[k] *= expfac;
    }
}

// ── Pressure update ─────────────────────────────────────────────────

void NoseHooverNPT::update_eps_dot_(const LaunchContext& ctx, double half_dt) {
    double V = box_lx_ * box_ly_ * box_lz_;
    if (V <= 0.0)
        return;

    // Consolidated readback fills h_npt_scratch_[0..7] in one pass —
    // ke_axes + ke_total + stress[0..2] + energy — via defer_stream +
    // event-sync (no main_stream drain).
    refresh_npt_scalars_(ctx);
    std::array<double, 3> ke_axis = {
        0.5 * ATV::units::mass_v2_to_energy * h_npt_scratch_[1],
        0.5 * ATV::units::mass_v2_to_energy * h_npt_scratch_[2],
        0.5 * ATV::units::mass_v2_to_energy * h_npt_scratch_[3]};
    double stress_diag[3] = {h_npt_scratch_[4], h_npt_scratch_[5],
                             h_npt_scratch_[6]};

    double diag[3] = {stress_diag[0] + 2.0 * ke_axis[0],
                      stress_diag[1] + 2.0 * ke_axis[1],
                      stress_diag[2] + 2.0 * ke_axis[2]};
    if (pdim_ == 3) {
        double ave = (1.0 / 3.0) * (diag[0] + diag[1] + diag[2]);
        diag[0] = diag[1] = diag[2] = ave;
    }

    double mtk_active_ke = 0.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            mtk_active_ke += ke_axis[a];
    const int N_atoms = (n_dof_ + 3) / 3;
    const double mtk_term =
        (pdim_ > 0)
            ? 6.0 * mtk_active_ke / (static_cast<double>(pdim_) * N_atoms)
            : 0.0;

    for (int a = 0; a < 3; ++a) {
        if (!couple_axis_[a])
            continue;
        double P_axis = diag[a] / V;
        double G_eps_a = (3.0 * V * (P_axis - P_target_) + mtk_term) / W_;
        eps_dot_[a] += G_eps_a * half_dt;
    }
}

// ── Velocity scale by baro ──────────────────────────────────────────

void NoseHooverNPT::scale_v_by_baro_(const LaunchContext& ctx, double half_dt) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;

    double sum_eps_dot = 0.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            sum_eps_dot += eps_dot_[a];
    const int N_atoms = (n_dof_ + 3) / 3;
    double mtk_term2 =
        (pdim_ > 0) ? sum_eps_dot / (static_cast<double>(pdim_) * N_atoms)
                    : 0.0;

    double sx = std::exp(-(eps_dot_[0] + mtk_term2) * half_dt);
    double sy = std::exp(-(eps_dot_[1] + mtk_term2) * half_dt);
    double sz = std::exp(-(eps_dot_[2] + mtk_term2) * half_dt);

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    if (sx == sy && sy == sz) {
        scale_vel_uniform_k<<<grid, block, 0, stream>>>(
            f.atoms.vx, f.atoms.vy, f.atoms.vz, static_cast<motion_t>(sx), n);
        ATV_CUDA_CHECK(cudaGetLastError());
        KE_global_ *= sx * sx;
    } else {
        scale_vel_per_axis_k<<<grid, block, 0, stream>>>(
            f.atoms.vx, f.atoms.vy, f.atoms.vz, static_cast<motion_t>(sx),
            static_cast<motion_t>(sy), static_cast<motion_t>(sz), n);
        ATV_CUDA_CHECK(cudaGetLastError());
        KE_global_ = compute_global_KE_(ctx);
    }
}

// ── Force kick ──────────────────────────────────────────────────────

void NoseHooverNPT::kick_v_by_force_(const LaunchContext& ctx, double half_dt) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    kick_v_k<<<grid, block, 0, stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.atoms.mass, static_cast<motion_t>(half_dt), n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Drift + box scale ───────────────────────────────────────────────

void NoseHooverNPT::propagate_x_and_box_(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;

    double eps_dt[3], a_arr[3], drift[3];
    for (int d = 0; d < 3; ++d) {
        eps_dt[d] = std::max(-kEpsDtMaxFraction,
                             std::min(eps_dot_[d] * dt_, kEpsDtMaxFraction));
        a_arr[d] = std::exp(eps_dt[d]);
        drift[d] = expm1c(eps_dt[d] * 0.5);
    }

    // Refresh cached box from device via defer-stream + event-sync —
    // a cooperative component bound earlier in `initial_integrate`
    // (e.g. `cuda::md::Deform`, which runs at order -100 vs this
    // integrator's order 0) may have rescaled f.box[0,4,8] since the
    // last NPT step. Without this read the cached values would
    // overwrite Deform's modification: stage-5 tension showed box_x
    // frozen at the initial 64.665 Å and strain stuck at 0 instead of
    // ramping to 1.7 over 172 000 steps.
    //
    // Routed through defer_stream so the main compute stream is not
    // drained; cudaEventSynchronize on the host thread blocks only
    // until the small 3-double copy completes
    // (cuda-md-review-cleanup-03-launch-purity).
    refresh_box_diag_(ctx);
    box_lx_ = h_box_diag_pinned_[0];
    box_ly_ = h_box_diag_pinned_[1];
    box_lz_ = h_box_diag_pinned_[2];

    double lx_new = box_lx_ * a_arr[0];
    double ly_new = box_ly_ * a_arr[1];
    double lz_new = box_lz_ * a_arr[2];

    // Build the new cuda::Box on host so the kernel can read cached
    // inv_l without a per-thread FP64 division, and so peer components
    // (bonded force kernels in the same / next step) see the updated
    // box via `ctx.runtime.host_box`.
    // Build the new orthorhombic Box for the kernel + host_box
    // mirror. Triclinic NPT not yet supported (couple_axis is per
    // diagonal axis; off-diagonals stay zero).
    cuda::MotionBox new_box = ctx.runtime.host_motion_box;
    new_box.cell[0] = static_cast<motion_t>(lx_new);
    new_box.cell[4] = static_cast<motion_t>(ly_new);
    new_box.cell[8] = static_cast<motion_t>(lz_new);
    new_box.refresh();

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    propagate_x_k<<<grid, block, 0, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.x_correction,
        f.atoms.y_correction, f.atoms.z_correction, f.atoms.image, f.atoms.vx,
        f.atoms.vy, f.atoms.vz, static_cast<motion_t>(a_arr[0]),
        static_cast<motion_t>(a_arr[1]), static_cast<motion_t>(a_arr[2]),
        static_cast<motion_t>(drift[0]), static_cast<motion_t>(drift[1]),
        static_cast<motion_t>(drift[2]), static_cast<motion_t>(dt_), f.box,
        new_box, n);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Mutate the host-side Context Box so peer components see the
    // new dimensions on the next launch().
    ctx.runtime.host_motion_box = new_box;
    ctx.runtime.host_box = box_cast<real_t>(new_box);

    box_lx_ = lx_new;
    box_ly_ = ly_new;
    box_lz_ = lz_new;
}

// ── Conserved MTK Hamiltonian ───────────────────────────────────────

double
NoseHooverNPT::extended_hamiltonian_(const LaunchContext& /*ctx*/) const {
    // E_pot already pulled into `h_npt_scratch_[7]` by the most recent
    // refresh_npt_scalars_ call (compute_global_KE_ in end_of_step P8).
    // KE is host-side (`KE_global_` analytically maintained through the
    // step), no D2H needed. The previous synchronous cudaMemcpy of
    // energy + kinetic from device is eliminated
    // (cuda-md-review-cleanup-03-launch-purity).
    double E_pot = h_npt_scratch_[7];
    double KE = KE_global_;
    double H = KE + E_pot;
    const double W_axis = W_ / 3.0;
    for (int a = 0; a < 3; ++a)
        if (couple_axis_[a])
            H += 0.5 * W_axis * eps_dot_[a] * eps_dot_[a];
    for (int k = 0; k < mtchain_; ++k)
        H += 0.5 * Q_t_[k] * eta_t_dot_[k] * eta_t_dot_[k];
    for (int k = 0; k < mpchain_; ++k)
        H += 0.5 * Q_b_[k] * eta_b_dot_[k] * eta_b_dot_[k];
    H += static_cast<double>(n_dof_) * kT_ * eta_t_[0];
    for (int k = 1; k < mtchain_; ++k)
        H += kT_ * eta_t_[k];
    for (int k = 0; k < mpchain_; ++k)
        H += kT_ * eta_b_[k];
    H += P_target_ * (box_lx_ * box_ly_ * box_lz_);
    return H;
}

// ── launch() dispatch ───────────────────────────────────────────────

void NoseHooverNPT::launch(const LaunchContext& ctx) {
    if (n_atoms_ <= 0)
        return;
    const double half_dt = 0.5 * dt_;

    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        if (!kinetic_initialized_) {
            KE_global_ = compute_global_KE_(ctx);
            kinetic_initialized_ = true;
        }
        compute_temp_target_(ctx.step);
        const double tau_T2 = tau_T_ * tau_T_;
        Q_t_[0] = static_cast<double>(n_dof_) * kT_ * tau_T2;
        for (int k = 1; k < mtchain_; ++k)
            Q_t_[k] = kT_ * tau_T2;

        nhc_baro_half_(half_dt);        // P1
        nhc_atoms_half_(ctx, half_dt);  // P2
        update_eps_dot_(ctx, half_dt);  // P3 (uses prev step's stress)
        scale_v_by_baro_(ctx, half_dt); // P4
        kick_v_by_force_(ctx, half_dt); // P5
        propagate_x_and_box_(ctx);      // P6

    } else if (ctx.at(ATV::md::Phase::final_integrate)) {
        kick_v_by_force_(ctx, half_dt); // P7

    } else if (ctx.at(ATV::md::Phase::end_of_step)) {
        // Reduce has run (single-GPU: no-op, but the order matches CPU).
        KE_global_ = compute_global_KE_(ctx);

        scale_v_by_baro_(ctx, half_dt); // P8
        update_eps_dot_(ctx, half_dt);  // P9
        nhc_atoms_half_(ctx, half_dt);  // P10
        nhc_baro_half_(half_dt);        // P11

        // Rewrite frame.scalars.{kinetic, conserved} so downstream
        // observers see post-NPT values. Source must be pinned host
        // memory so the async copy can outlive this lexical scope —
        // a stack-local `double H` would have its storage reused as
        // soon as launch() returns, racing the queued D2H copy.
        // h_kinetic_conserved_pinned_[0]=kinetic, [1]=conserved.
        auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
        h_kinetic_conserved_pinned_[0] = KE_global_;
        h_kinetic_conserved_pinned_[1] = extended_hamiltonian_(ctx);
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            &ctx.frame.scalars->kinetic, &h_kinetic_conserved_pinned_[0],
            sizeof(double), cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            &ctx.frame.scalars->conserved, &h_kinetic_conserved_pinned_[1],
            sizeof(double), cudaMemcpyHostToDevice, stream));
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
