/// @file cuda/dynamics/nve_limit.cu
/// @brief CUDA NVE/limit integrator kernels.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nve_limit.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

namespace {

/// @brief Apply Euclidean speed cap: if `vsq > vmax_sq`, rescale
/// velocity by `vmax / sqrt(vsq)`. Matches LAMMPS
/// `fix_nve_limit.cpp::initial_integrate` lines 105–112.
__device__ inline void apply_cap(motion_t& vx, motion_t& vy, motion_t& vz,
                                 motion_t vmax, motion_t vmax_sq) {
    motion_t vsq = vx * vx + vy * vy + vz * vz;
    if (vsq > vmax_sq) {
        motion_t s = vmax / sqrt(vsq);
        vx *= s;
        vy *= s;
        vz *= s;
    }
}

__global__ void initial_integrate_limit(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ xc, real_t* __restrict__ yc, real_t* __restrict__ zc,
    std::int8_t* __restrict__ image, motion_t* __restrict__ vx,
    motion_t* __restrict__ vy, motion_t* __restrict__ vz,
    const real_t* __restrict__ fx, const real_t* __restrict__ fy,
    const real_t* __restrict__ fz, const motion_t* __restrict__ mass,
    cuda::MotionBox box, motion_t dt, motion_t vmax, motion_t vmax_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t inv_m = motion_t(1) / static_cast<motion_t>(mass[i]);
    motion_t hdt = motion_t(0.5) * dt;
    motion_t vxi = static_cast<motion_t>(vx[i])
                   + hdt * static_cast<motion_t>(fx[i]) * inv_m
                         * ATV::units::force_to_accel;
    motion_t vyi = static_cast<motion_t>(vy[i])
                   + hdt * static_cast<motion_t>(fy[i]) * inv_m
                         * ATV::units::force_to_accel;
    motion_t vzi = static_cast<motion_t>(vz[i])
                   + hdt * static_cast<motion_t>(fz[i]) * inv_m
                         * ATV::units::force_to_accel;
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = static_cast<motion_t>(vxi);
    vy[i] = static_cast<motion_t>(vyi);
    vz[i] = static_cast<motion_t>(vzi);
    wrap_position(
        x, y, z, xc, yc, zc, image, i, box, load_position(x, xc, i) + dt * vxi,
        load_position(y, yc, i) + dt * vyi, load_position(z, zc, i) + dt * vzi);
}

__global__ void
final_integrate_limit(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
                      motion_t* __restrict__ vz, const real_t* __restrict__ fx,
                      const real_t* __restrict__ fy,
                      const real_t* __restrict__ fz,
                      const motion_t* __restrict__ mass, motion_t dt,
                      motion_t vmax, motion_t vmax_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t inv_m = motion_t(1) / static_cast<motion_t>(mass[i]);
    motion_t hdt = motion_t(0.5) * dt;
    motion_t vxi = static_cast<motion_t>(vx[i])
                   + hdt * static_cast<motion_t>(fx[i]) * inv_m
                         * ATV::units::force_to_accel;
    motion_t vyi = static_cast<motion_t>(vy[i])
                   + hdt * static_cast<motion_t>(fy[i]) * inv_m
                         * ATV::units::force_to_accel;
    motion_t vzi = static_cast<motion_t>(vz[i])
                   + hdt * static_cast<motion_t>(fz[i]) * inv_m
                         * ATV::units::force_to_accel;
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = static_cast<motion_t>(vxi);
    vy[i] = static_cast<motion_t>(vyi);
    vz[i] = static_cast<motion_t>(vzi);
}

} // anonymous namespace

void launch_nve_limit_initial(real_t* x, real_t* y, real_t* z, real_t* xc,
                              real_t* yc, real_t* zc, std::int8_t* image,
                              motion_t* vx, motion_t* vy, motion_t* vz,
                              const real_t* fx, const real_t* fy,
                              const real_t* fz, const motion_t* mass,
                              cuda::MotionBox box, motion_t dt, motion_t xmax,
                              int n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    motion_t vmax = xmax / dt;
    motion_t vmax_sq = vmax * vmax;
    initial_integrate_limit<<<grid, block, 0, stream>>>(
        x, y, z, xc, yc, zc, image, vx, vy, vz, fx, fy, fz, mass, box, dt, vmax,
        vmax_sq, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void launch_nve_limit_final(motion_t* vx, motion_t* vy, motion_t* vz,
                            const real_t* fx, const real_t* fy,
                            const real_t* fz, const motion_t* mass, motion_t dt,
                            motion_t xmax, int n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    motion_t vmax = xmax / dt;
    motion_t vmax_sq = vmax * vmax;
    final_integrate_limit<<<grid, block, 0, stream>>>(
        vx, vy, vz, fx, fy, fz, mass, dt, vmax, vmax_sq, n);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void NVELimit::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = frame.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    motion_t vmax = static_cast<motion_t>(xmax_) / static_cast<motion_t>(dt_);
    motion_t vmax_sq = vmax * vmax;
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_integrate_limit<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z,
            frame.atoms.x_correction, frame.atoms.y_correction,
            frame.atoms.z_correction, frame.atoms.image, frame.atoms.vx,
            frame.atoms.vy, frame.atoms.vz, frame.atoms.fx, frame.atoms.fy,
            frame.atoms.fz, frame.atoms.mass, ctx.runtime.host_motion_box, dt_,
            vmax, vmax_sq, n);
        ATV_CUDA_CHECK(cudaGetLastError());
    } else {
        final_integrate_limit<<<grid, block, 0, stream>>>(
            frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
            frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, dt_, vmax,
            vmax_sq, n);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
