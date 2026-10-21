/// @file cuda/dynamics/nose_hoover.cu
/// @brief NoseHoover -- single-chain Nosé-Hoover thermostat.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

struct NoseHoover::State {
    motion_t* d_xi = nullptr;              // thermostat variable [1/fs]
    motion_t* d_ke_temp = nullptr;         // temporary KE [eV]
    motion_t* d_scale = nullptr;           // velocity scale factor [1 element]
    unsigned int* d_block_count = nullptr; // last-block detection counter
    motion_t Q = 0;                        // thermostat mass
    int ndof = 0;
};

NoseHoover::NoseHoover(double T_target, double tau, double dt)
    : T_target_(T_target), tau_(tau), dt_(dt),
      state_(std::make_unique<State>()) {}

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
    const motion_t* __restrict__ vx, const motion_t* __restrict__ vy,
    const motion_t* __restrict__ vz, const motion_t* __restrict__ mass,
    motion_t* __restrict__ ke_out, int n, motion_t kMvv2e_val,
    motion_t* __restrict__ xi, motion_t* __restrict__ scale, motion_t Q,
    motion_t ndof_kbT, motion_t half_dt,
    unsigned int* __restrict__ block_count) {
    extern __shared__ motion_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    motion_t val = 0;
    if (idx < n) {
        motion_t m = static_cast<motion_t>(mass[idx]);
        motion_t vx_w = static_cast<motion_t>(vx[idx]);
        motion_t vy_w = static_cast<motion_t>(vy[idx]);
        motion_t vz_w = static_cast<motion_t>(vz[idx]);
        val = motion_t(0.5) * kMvv2e_val * m
              * (vx_w * vx_w + vy_w * vy_w + vz_w * vz_w);
    }
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(ke_out, sdata[0]);

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
        motion_t ke_val = *ke_out;
        *xi += half_dt * (motion_t(2) * ke_val - ndof_kbT) / Q;
        *scale = exp(-half_dt * (*xi));
    }
}

/// @brief Scale all velocities by the precomputed factor.
__global__ void scale_vel(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
                          motion_t* __restrict__ vz,
                          const motion_t* __restrict__ scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    motion_t s = *scale;
    vx[idx] = static_cast<motion_t>(static_cast<motion_t>(vx[idx]) * s);
    vy[idx] = static_cast<motion_t>(static_cast<motion_t>(vy[idx]) * s);
    vz[idx] = static_cast<motion_t>(static_cast<motion_t>(vz[idx]) * s);
}

} // anonymous namespace

void NoseHoover::build(const BuildContext& ctx) {
    int n = ctx.frame.n_atoms();
    if (n <= 0)
        return;
    state_->ndof = 3 * n - 3;
    state_->Q = static_cast<motion_t>(state_->ndof) * ATV::units::kB_ev_per_k
                * T_target_ * tau_ * tau_;

    ATV_CUDA_CHECK(cudaMalloc(&state_->d_xi, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_ke_temp, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_scale, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&state_->d_block_count, sizeof(unsigned int)));
    ATV_CUDA_CHECK(cudaMemset(state_->d_xi, 0, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMemset(state_->d_block_count, 0, sizeof(unsigned int)));

    // Initialize scale to 1.0
    motion_t one = motion_t(1);
    ATV_CUDA_CHECK(cudaMemcpy(state_->d_scale, &one, sizeof(motion_t),
                              cudaMemcpyHostToDevice));
}

void NoseHoover::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    motion_t ndof_kbT = static_cast<motion_t>(state_->ndof)
                        * ATV::units::kB_ev_per_k * T_target_;
    motion_t half_dt = motion_t(0.5) * dt_;

    // Fused: reduce KE + update xi/scale (last block computes scale)
    ATV_CUDA_CHECK(
        cudaMemsetAsync(state_->d_ke_temp, 0, sizeof(motion_t), stream));
    reduce_ke_and_update<<<grid, block, block * sizeof(motion_t), stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass, state_->d_ke_temp, n,
        ATV::units::mass_v2_to_energy, state_->d_xi, state_->d_scale, state_->Q,
        ndof_kbT, half_dt, state_->d_block_count);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Scale velocities (reads scale written by the last block above)
    scale_vel<<<grid, block, 0, stream>>>(f.atoms.vx, f.atoms.vy, f.atoms.vz,
                                          state_->d_scale, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NoseHoover::destroy(const DestroyContext& ctx) {
    ATV_CUDA_CHECK(cudaFree(state_->d_xi));
    ATV_CUDA_CHECK(cudaFree(state_->d_ke_temp));
    ATV_CUDA_CHECK(cudaFree(state_->d_scale));
    ATV_CUDA_CHECK(cudaFree(state_->d_block_count));
    *state_ = {};
}

} // namespace md
} // namespace cuda
} // namespace ATV
