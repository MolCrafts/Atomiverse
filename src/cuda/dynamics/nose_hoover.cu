/// @file cuda/dynamics/nose_hoover.cu
/// @brief NoseHoover -- single-chain Nosé-Hoover thermostat.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {

namespace cuda {
namespace md {

struct NoseHoover::State {
    mixed_t* d_xi = nullptr;               // thermostat variable [1/fs]
    acc_t* d_ke_temp = nullptr;            // temporary KE [eV]
    mixed_t* d_scale = nullptr;            // velocity scale factor [1 element]
    unsigned int* d_block_count = nullptr; // last-block detection counter
    mixed_t Q = 0;                         // thermostat mass
    int ndof = 0;
};

NoseHoover::NoseHoover(double T_target, double tau, double dt)
    : T_target_(T_target), tau_(tau), dt_(dt),
      state_(std::make_unique<State>()) {
    if (!(T_target > 0.0) || !std::isfinite(T_target) || !(tau > 0.0)
        || !std::isfinite(tau) || !(dt > 0.0) || !std::isfinite(dt))
        throw std::invalid_argument(
            "cuda::md::NoseHoover: T_target [K], tau [fs] and dt [fs] must be "
            "finite and > 0");
}

void NoseHoover::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument(
            "cuda::md::NoseHoover: timestep must be finite and > 0 fs, got "
            + std::to_string(dt_fs));
    dt_ = dt_fs;
}

NoseHoover::~NoseHoover() = default;
NoseHoover::NoseHoover(NoseHoover&&) noexcept = default;
NoseHoover& NoseHoover::operator=(NoseHoover&&) noexcept = default;

std::vector<DefaultBinding> NoseHoover::default_bindings() const {
    return {ATV::make_binding(ATV::md::Phase::initial_integrate, 100),
            ATV::make_binding(ATV::md::Phase::final_integrate, 100)};
}

namespace {

/// @brief Fused KE reduction + thermostat update.
///
/// Reduces per-atom kinetic energy via shared-memory block reduction and
/// atomicAdd. The last block to finish (detected via atomicInc counter)
/// updates the thermostat variable xi and computes the velocity scale
/// factor, eliminating the separate <<<1,1>>> kernel launch.
__global__ void reduce_ke_and_update(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    acc_t* __restrict__ ke_out, int n, acc_t kMvv2e_val,
    mixed_t* __restrict__ xi, mixed_t* __restrict__ scale, mixed_t Q,
    mixed_t ndof_kbT, mixed_t half_dt, unsigned int* __restrict__ block_count,
    const ForceFault* fault) {
    // A faulted run keeps its thermostat state at the faulting step.
    if (state_frozen(fault))
        return;
    using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t val = 0;
    if (idx < n) {
        const acc_t m = mass[idx];
        const acc_t vx_w = vx[idx], vy_w = vy[idx], vz_w = vz[idx];
        val = acc_t(0.5) * kMvv2e_val * m
              * (vx_w * vx_w + vy_w * vy_w + vz_w * vz_w);
    }
    const acc_t sum = Reduce(reduction).Sum(val);
    if (threadIdx.x == 0)
        atomicAdd(ke_out, sum);

    // Last-block pattern: ensure atomicAdd is globally visible, then detect
    // the last block to finish. Thread 0 of the last block updates xi/scale.
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        unsigned int old = atomicInc(block_count, gridDim.x - 1);
        is_last = (old == gridDim.x - 1);
    }
    __syncthreads();

    if (is_last && threadIdx.x == 0) {
        const mixed_t ke_val = mixed_t(*ke_out);
        *xi += half_dt * (mixed_t(2) * ke_val - ndof_kbT) / Q;
        *scale = exp(-half_dt * (*xi));
    }
}

__global__ void scale_vel(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                          mixed_t* __restrict__ vz,
                          const mixed_t* __restrict__ scale, int n,
                          const ForceFault* fault) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || state_frozen(fault))
        return;
    const mixed_t s = *scale;
    vx[idx] *= s;
    vy[idx] *= s;
    vz[idx] *= s;
}

} // anonymous namespace

void NoseHoover::build(const BuildContext& ctx) {
    state_->ndof = ctx.runtime.md.n_dof_global;
    state_->Q = static_cast<mixed_t>(state_->ndof * ATV::units::md_boltz
                                     * T_target_ * tau_ * tau_);

    ATV_CUDA_CHECK(cudaMalloc(&state_->d_xi, sizeof(mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_ke_temp, sizeof(acc_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_scale, sizeof(mixed_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_block_count, sizeof(unsigned int)));
    ATV_CUDA_CHECK(cudaMemset(state_->d_xi, 0, sizeof(mixed_t)));
    ATV_CUDA_CHECK(cudaMemset(state_->d_block_count, 0, sizeof(unsigned int)));

    // Initialize scale to 1.0
    const mixed_t one = mixed_t(1);
    ATV_CUDA_CHECK(cudaMemcpy(state_->d_scale, &one, sizeof(mixed_t),
                              cudaMemcpyHostToDevice));
}

void NoseHoover::launch(const LaunchContext& ctx) {
    // No thermal bath (e.g. one atom after COM removal): keep velocities and
    // xi, as cpu::md::NoseHoover does, instead of dividing by Q = 0. The DOF
    // count is fixed at build (identity edits rebuild), so this branches on
    // a build-time host value, never on device state.
    if (state_->ndof <= 0)
        return;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    const mixed_t ndof_kbT =
        static_cast<mixed_t>(state_->ndof * ATV::units::md_boltz * T_target_);
    const mixed_t half_dt = mixed_t(0.5 * dt_);

    // Fused: reduce KE + update xi/scale (last block computes scale)
    cudaMemsetAsync(state_->d_ke_temp, 0, sizeof(acc_t), stream);
    reduce_ke_and_update<<<grid, block, 0, stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass, state_->d_ke_temp, n,
        ATV::units::md_mvv2e, state_->d_xi, state_->d_scale, state_->Q,
        ndof_kbT, half_dt, state_->d_block_count, f.force_fault);

    // Scale velocities (reads scale written by the last block above)
    scale_vel<<<grid, block, 0, stream>>>(f.atoms.vx, f.atoms.vy, f.atoms.vz,
                                          state_->d_scale, n, f.force_fault);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NoseHoover::destroy(const DestroyContext& ctx) {
    cudaFree(state_->d_xi);
    cudaFree(state_->d_ke_temp);
    cudaFree(state_->d_scale);
    cudaFree(state_->d_block_count);
    *state_ = {};
}

} // namespace md
} // namespace cuda
} // namespace ATV
