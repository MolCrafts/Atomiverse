/// @file cuda/dynamics/velocity_verlet.cu

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace md {

namespace {

// Integration runs in mixed_t: coordinates are reconstructed from the stored
// primary + correction pair, advanced, wrapped against the FP64 cell and
// split again. Forces are decoded from fixed point; a faulted force is never
// consumed (the first refusing launch stamps step and tag).
__global__ void initial_integrate(
    pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
    pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
    mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
    mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
    const fixed_t* __restrict__ fy, const fixed_t* __restrict__ fz,
    const mixed_t* __restrict__ mass, ForceFault* fault,
    const std::int64_t* __restrict__ tag, const long long* step_ptr,
    long long step, cuda::Box box, mixed_t dt, int n) {
    if (!forces_consumable(fault, tag, step_ptr, step))
        return;
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const mixed_t scale =
        mixed_t(0.5) * dt * mixed_t(ATV::units::force_to_accel) / mass[i];
    const mixed_t ux = vx[i] + scale * decode_force<mixed_t>(fx[i]);
    const mixed_t uy = vy[i] + scale * decode_force<mixed_t>(fy[i]);
    const mixed_t uz = vz[i] + scale * decode_force<mixed_t>(fz[i]);
    vx[i] = ux;
    vy[i] = uy;
    vz[i] = uz;
    mixed_t px = load_position(x, xc, i) + dt * ux;
    mixed_t py = load_position(y, yc, i) + dt * uy;
    mixed_t pz = load_position(z, zc, i) + dt * uz;
    // PBC wrap by whole lattice vectors (cached inverse cell, no division).
    box.wrap_position(px, py, pz);
    store_position(x, xc, i, px);
    store_position(y, yc, i, py);
    store_position(z, zc, i, pz);
}

__global__ void
final_integrate(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
                const fixed_t* __restrict__ fy, const fixed_t* __restrict__ fz,
                const mixed_t* __restrict__ mass, ForceFault* fault,
                const std::int64_t* __restrict__ tag, const long long* step_ptr,
                long long step, mixed_t dt, int n) {
    if (!forces_consumable(fault, tag, step_ptr, step))
        return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const mixed_t scale =
        mixed_t(0.5) * dt * mixed_t(ATV::units::force_to_accel) / mass[i];
    vx[i] += scale * decode_force<mixed_t>(fx[i]);
    vy[i] += scale * decode_force<mixed_t>(fy[i]);
    vz[i] += scale * decode_force<mixed_t>(fz[i]);
}

} // anonymous namespace

void VelocityVerlet::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    const auto& a = frame.atoms;
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_integrate<<<grid, block, 0, stream>>>(
            a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.vx, a.vy, a.vz, a.fx,
            a.fy, a.fz, a.mass, frame.force_fault, a.tag,
            ctx.runtime.device_step, ctx.step, ctx.runtime.host_box,
            mixed_t(dt_), n);
    } else {
        final_integrate<<<grid, block, 0, stream>>>(
            a.vx, a.vy, a.vz, a.fx, a.fy, a.fz, a.mass, frame.force_fault,
            a.tag, ctx.runtime.device_step, ctx.step, mixed_t(dt_), n);
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
