/// @file cuda/dynamics/velocity_verlet.cu

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
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

__global__ void initial_integrate(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ xc, real_t* __restrict__ yc, real_t* __restrict__ zc,
    std::int8_t* __restrict__ image, motion_t* __restrict__ vx,
    motion_t* __restrict__ vy, motion_t* __restrict__ vz,
    const real_t* __restrict__ fx, const real_t* __restrict__ fy,
    const real_t* __restrict__ fz, const motion_t* __restrict__ mass,
    cuda::MotionBox box, motion_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t inv_m = motion_t(1) / static_cast<motion_t>(mass[i]);
    motion_t hdt = motion_t(0.5) * dt;
    motion_t ax =
        static_cast<motion_t>(fx[i]) * inv_m * ATV::units::force_to_accel;
    motion_t ay =
        static_cast<motion_t>(fy[i]) * inv_m * ATV::units::force_to_accel;
    motion_t az =
        static_cast<motion_t>(fz[i]) * inv_m * ATV::units::force_to_accel;
    vx[i] += hdt * ax;
    vy[i] += hdt * ay;
    vz[i] += hdt * az;
    wrap_position(x, y, z, xc, yc, zc, image, i, box,
                  load_position(x, xc, i) + dt * vx[i],
                  load_position(y, yc, i) + dt * vy[i],
                  load_position(z, zc, i) + dt * vz[i]);
}

__global__ void
final_integrate(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
                motion_t* __restrict__ vz, const real_t* __restrict__ fx,
                const real_t* __restrict__ fy, const real_t* __restrict__ fz,
                const motion_t* __restrict__ mass, motion_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    motion_t inv_m = motion_t(1) / static_cast<motion_t>(mass[i]);
    motion_t hdt = motion_t(0.5) * dt;
    vx[i] +=
        hdt * static_cast<motion_t>(fx[i]) * inv_m * ATV::units::force_to_accel;
    vy[i] +=
        hdt * static_cast<motion_t>(fy[i]) * inv_m * ATV::units::force_to_accel;
    vz[i] +=
        hdt * static_cast<motion_t>(fz[i]) * inv_m * ATV::units::force_to_accel;
}

} // anonymous namespace

void VelocityVerlet::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_integrate<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z,
            frame.atoms.x_correction, frame.atoms.y_correction,
            frame.atoms.z_correction, frame.atoms.image, frame.atoms.vx,
            frame.atoms.vy, frame.atoms.vz, frame.atoms.fx, frame.atoms.fy,
            frame.atoms.fz, frame.atoms.mass, ctx.runtime.host_motion_box, dt_,
            n);
        ATV_CUDA_CHECK(cudaGetLastError());
    } else {
        final_integrate<<<grid, block, 0, stream>>>(
            frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
            frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, dt_, n);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
