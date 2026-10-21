/// @file cuda/dynamics/barostat.cu
/// @brief Barostat -- Berendsen isotropic pressure coupling.

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/barostat.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

struct Barostat::State {
    prec_t* host_cell = nullptr; // pinned 9-scalar geometry publication
    int n_atoms = 0;
};

Barostat::Barostat(double P_target, double tau_p, double dt)
    : P_target_(P_target), tau_p_(tau_p), dt_(dt),
      state_(std::make_unique<State>()) {}

Barostat::~Barostat() = default;
Barostat::Barostat(Barostat&&) noexcept = default;
Barostat& Barostat::operator=(Barostat&&) noexcept = default;

std::vector<DefaultBinding> Barostat::default_bindings() const {
    // end_of_step @ -45, matching cpu::md::Barostat and LAMMPS
    // `fix press/berendsen` (an end-of-step fix). Runs after KineticEnergy
    // (end_of_step @ -100) so the instantaneous pressure the barostat reads
    // (virial + current kinetic) is complete — at final_integrate the kinetic
    // for this step is not yet accumulated, giving a stale pressure.
    return {ATV::make_binding(ATV::md::Phase::end_of_step, -45)};
}

namespace {

// Compute μ from pressure and target, then scale box + positions
__global__ void
berendsen_scale_k(real_t* __restrict__ x, real_t* __restrict__ y,
                  real_t* __restrict__ z, real_t* __restrict__ xc,
                  real_t* __restrict__ yc, real_t* __restrict__ zc,
                  prec_t* __restrict__ box, const double* __restrict__ stress,
                  motion_t P_target, motion_t beta_dt_tau, int n) {
    // Isotropic pressure = -(σ_xx + σ_yy + σ_zz) / 3
    // Compute P and mu once per block via shared memory
    __shared__ motion_t s_mu;
    if (threadIdx.x == 0) {
        motion_t P = -(static_cast<motion_t>(stress[0])
                       + static_cast<motion_t>(stress[1])
                       + static_cast<motion_t>(stress[2]))
                     / motion_t(3);
        s_mu = cbrt(motion_t(1) - beta_dt_tau * (P_target - P));
    }
    __syncthreads();
    motion_t mu = s_mu;

    // Thread 0 of block 0 rescales box
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        box[0] = static_cast<prec_t>(static_cast<motion_t>(box[0]) * mu);
        box[4] = static_cast<prec_t>(static_cast<motion_t>(box[4]) * mu);
        box[8] = static_cast<prec_t>(static_cast<motion_t>(box[8]) * mu);
    }

    // All threads rescale positions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    store_position(x, xc, idx, load_position(x, xc, idx) * mu);
    store_position(y, yc, idx, load_position(y, yc, idx) * mu);
    store_position(z, zc, idx, load_position(z, zc, idx) * mu);
}

} // anonymous namespace

void Barostat::build(const BuildContext& ctx) {
    state_->n_atoms = ctx.frame.n_atoms();
    ATV_CUDA_CHECK(cudaMallocHost(&state_->host_cell, 9 * sizeof(prec_t)));
}

void Barostat::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    // β is the isothermal compressibility; for Berendsen we use β*dt/τ_p as a single parameter.
    // Typical β for solids ~ 1e-4 GPa^-1. We absorb it into the coupling.
    motion_t beta_dt_tau = dt_ / tau_p_;

    berendsen_scale_k<<<grid, block, 0, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.x_correction,
        f.atoms.y_correction, f.atoms.z_correction, f.box, f.scalars->stress,
        P_target_, beta_dt_tau, n);
    ATV_CUDA_CHECK(cudaGetLastError());
    // Barostat has host-published geometry, like NPT: this small scalar
    // synchronization keeps all subsequent force/integration caches current.
    ATV_CUDA_CHECK(cudaMemcpyAsync(state_->host_cell, f.box, 9 * sizeof(prec_t),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int k = 0; k < 9; ++k)
        ctx.runtime.host_motion_box.cell[k] =
            static_cast<motion_t>(state_->host_cell[k]);
    ctx.runtime.host_motion_box.refresh();
    ctx.runtime.host_box = box_cast<real_t>(ctx.runtime.host_motion_box);
}

void Barostat::destroy(const DestroyContext&) {
    if (state_->host_cell)
        ATV_CUDA_CHECK(cudaFreeHost(state_->host_cell));
    *state_ = {};
}

} // namespace md
} // namespace cuda
} // namespace ATV
