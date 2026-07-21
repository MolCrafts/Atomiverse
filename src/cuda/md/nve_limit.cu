/// @file md/nve_limit.cu
/// @brief CUDA NVE/limit integrator kernels.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/md/nve_limit.cuh"
#include "atomiverse/facade/stage.h"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

namespace {

/// @brief Apply Euclidean speed cap: if `vsq > vmax_sq`, rescale
/// velocity by `vmax / sqrt(vsq)`. Matches LAMMPS
/// `fix_nve_limit.cpp::initial_integrate` lines 105–112.
__device__ inline void apply_cap(work_t& vx, work_t& vy, work_t& vz,
                                 work_t vmax, work_t vmax_sq) {
    work_t vsq = vx * vx + vy * vy + vz * vz;
    if (vsq > vmax_sq) {
        work_t s = vmax / sqrt(vsq);
        vx *= s;
        vy *= s;
        vz *= s;
    }
}

__global__ void initial_integrate_limit(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ vx, real_t* __restrict__ vy, real_t* __restrict__ vz,
    const real_t* __restrict__ fx, const real_t* __restrict__ fy,
    const real_t* __restrict__ fz, const real_t* __restrict__ mass,
    cuda::Box box, work_t dt, work_t vmax, work_t vmax_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work_t inv_m = work_t(1) / static_cast<work_t>(mass[i]);
    work_t hdt = work_t(0.5) * dt;
    work_t vxi = static_cast<work_t>(vx[i])
                 + hdt * static_cast<work_t>(fx[i]) * inv_m * kFtm2v;
    work_t vyi = static_cast<work_t>(vy[i])
                 + hdt * static_cast<work_t>(fy[i]) * inv_m * kFtm2v;
    work_t vzi = static_cast<work_t>(vz[i])
                 + hdt * static_cast<work_t>(fz[i]) * inv_m * kFtm2v;
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = static_cast<real_t>(vxi);
    vy[i] = static_cast<real_t>(vyi);
    vz[i] = static_cast<real_t>(vzi);
    x[i] = static_cast<real_t>(static_cast<work_t>(x[i]) + dt * vxi);
    y[i] = static_cast<real_t>(static_cast<work_t>(y[i]) + dt * vyi);
    z[i] = static_cast<real_t>(static_cast<work_t>(z[i]) + dt * vzi);
    // Orthorhombic PBC wrap via cached inv_l on the by-value Box.
    work_t Lx = work_t(box.cell[0]), Ly = work_t(box.cell[4]),
           Lz = work_t(box.cell[8]);
    work_t inv_Lx = work_t(box.inv_l[0]), inv_Ly = work_t(box.inv_l[1]),
           inv_Lz = work_t(box.inv_l[2]);
    x[i] -= static_cast<real_t>(Lx * floor(static_cast<work_t>(x[i]) * inv_Lx));
    y[i] -= static_cast<real_t>(Ly * floor(static_cast<work_t>(y[i]) * inv_Ly));
    z[i] -= static_cast<real_t>(Lz * floor(static_cast<work_t>(z[i]) * inv_Lz));
}

__global__ void final_integrate_limit(
    real_t* __restrict__ vx, real_t* __restrict__ vy, real_t* __restrict__ vz,
    const real_t* __restrict__ fx, const real_t* __restrict__ fy,
    const real_t* __restrict__ fz, const real_t* __restrict__ mass, work_t dt,
    work_t vmax, work_t vmax_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work_t inv_m = work_t(1) / static_cast<work_t>(mass[i]);
    work_t hdt = work_t(0.5) * dt;
    work_t vxi = static_cast<work_t>(vx[i])
                 + hdt * static_cast<work_t>(fx[i]) * inv_m * kFtm2v;
    work_t vyi = static_cast<work_t>(vy[i])
                 + hdt * static_cast<work_t>(fy[i]) * inv_m * kFtm2v;
    work_t vzi = static_cast<work_t>(vz[i])
                 + hdt * static_cast<work_t>(fz[i]) * inv_m * kFtm2v;
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = static_cast<real_t>(vxi);
    vy[i] = static_cast<real_t>(vyi);
    vz[i] = static_cast<real_t>(vzi);
}

} // anonymous namespace

void launch_nve_limit_initial(real_t* x, real_t* y, real_t* z, real_t* vx,
                              real_t* vy, real_t* vz, const real_t* fx,
                              const real_t* fy, const real_t* fz,
                              const real_t* mass, cuda::Box box, work_t dt,
                              work_t xmax, int n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    work_t vmax = xmax / dt;
    work_t vmax_sq = vmax * vmax;
    initial_integrate_limit<<<grid, block, 0, stream>>>(
        x, y, z, vx, vy, vz, fx, fy, fz, mass, box, dt, vmax, vmax_sq, n);
}

void launch_nve_limit_final(real_t* vx, real_t* vy, real_t* vz,
                            const real_t* fx, const real_t* fy,
                            const real_t* fz, const real_t* mass, work_t dt,
                            work_t xmax, int n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    work_t vmax = xmax / dt;
    work_t vmax_sq = vmax * vmax;
    final_integrate_limit<<<grid, block, 0, stream>>>(
        vx, vy, vz, fx, fy, fz, mass, dt, vmax, vmax_sq, n);
}

void NVELimit::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    work_t vmax = static_cast<work_t>(xmax_) / static_cast<work_t>(dt_);
    work_t vmax_sq = vmax * vmax;
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_integrate_limit<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.vx,
            frame.atoms.vy, frame.atoms.vz, frame.atoms.fx, frame.atoms.fy,
            frame.atoms.fz, frame.atoms.mass, ctx.runtime.host_box, dt_, vmax,
            vmax_sq, n);
    } else {
        final_integrate_limit<<<grid, block, 0, stream>>>(
            frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
            frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, dt_, vmax,
            vmax_sq, n);
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
