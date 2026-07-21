/// @file md/nose_hoover.cu
/// @brief NoseHoover -- single-chain Nosé-Hoover thermostat.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/md/nose_hoover.cuh"
#include "atomiverse/facade/stage.h"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

struct NoseHoover::State {
    work_t* d_xi = nullptr;                // thermostat variable [1/fs]
    work_t* d_ke_temp = nullptr;           // temporary KE [eV]
    work_t* d_scale = nullptr;             // velocity scale factor [1 element]
    unsigned int* d_block_count = nullptr; // last-block detection counter
    work_t Q = 0;                          // thermostat mass
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
    const real_t* __restrict__ vx, const real_t* __restrict__ vy,
    const real_t* __restrict__ vz, const real_t* __restrict__ mass,
    work_t* __restrict__ ke_out, int n, work_t kMvv2e_val,
    work_t* __restrict__ xi, work_t* __restrict__ scale, work_t Q,
    work_t ndof_kbT, work_t half_dt, unsigned int* __restrict__ block_count) {
    extern __shared__ work_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    work_t val = 0;
    if (idx < n) {
        work_t m = static_cast<work_t>(mass[idx]);
        work_t vx_w = static_cast<work_t>(vx[idx]);
        work_t vy_w = static_cast<work_t>(vy[idx]);
        work_t vz_w = static_cast<work_t>(vz[idx]);
        val = work_t(0.5) * kMvv2e_val * m
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
        work_t ke_val = *ke_out;
        *xi += half_dt * (work_t(2) * ke_val - ndof_kbT) / Q;
        *scale = exp(-half_dt * (*xi));
    }
}

/// @brief Scale all velocities by the precomputed factor.
__global__ void scale_vel(real_t* __restrict__ vx, real_t* __restrict__ vy,
                          real_t* __restrict__ vz,
                          const work_t* __restrict__ scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    work_t s = *scale;
    vx[idx] = static_cast<real_t>(static_cast<work_t>(vx[idx]) * s);
    vy[idx] = static_cast<real_t>(static_cast<work_t>(vy[idx]) * s);
    vz[idx] = static_cast<real_t>(static_cast<work_t>(vz[idx]) * s);
}

} // anonymous namespace

void NoseHoover::build(const BuildContext& ctx) {
    int n = ctx.frame.n_atoms();
    state_->ndof = 3 * n - 3;
    state_->Q = static_cast<work_t>(state_->ndof) * kBoltzmann * T_target_
                * tau_ * tau_;

    cudaMalloc(&state_->d_xi, sizeof(work_t));
    cudaMalloc(&state_->d_ke_temp, sizeof(work_t));
    cudaMalloc(&state_->d_scale, sizeof(work_t));
    cudaMalloc(&state_->d_block_count, sizeof(unsigned int));
    cudaMemset(state_->d_xi, 0, sizeof(work_t));
    cudaMemset(state_->d_block_count, 0, sizeof(unsigned int));

    // Initialize scale to 1.0
    work_t one = work_t(1);
    cudaMemcpy(state_->d_scale, &one, sizeof(work_t), cudaMemcpyHostToDevice);
}

void NoseHoover::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    work_t ndof_kbT =
        static_cast<work_t>(state_->ndof) * kBoltzmann * T_target_;
    work_t half_dt = work_t(0.5) * dt_;

    // Fused: reduce KE + update xi/scale (last block computes scale)
    cudaMemsetAsync(state_->d_ke_temp, 0, sizeof(work_t), stream);
    reduce_ke_and_update<<<grid, block, block * sizeof(work_t), stream>>>(
        f.atoms.vx, f.atoms.vy, f.atoms.vz, f.atoms.mass, state_->d_ke_temp, n,
        kMvv2e, state_->d_xi, state_->d_scale, state_->Q, ndof_kbT, half_dt,
        state_->d_block_count);

    // Scale velocities (reads scale written by the last block above)
    scale_vel<<<grid, block, 0, stream>>>(f.atoms.vx, f.atoms.vy, f.atoms.vz,
                                          state_->d_scale, n);
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
