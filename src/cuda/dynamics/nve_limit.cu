/// @file cuda/dynamics/nve_limit.cu
/// @brief CUDA NVE/limit integrator kernels.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/nve_limit.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace md {

namespace {

__device__ inline void apply_cap(mixed_t& vx, mixed_t& vy, mixed_t& vz,
                                 mixed_t vmax, mixed_t vmax_sq) {
    mixed_t vsq = vx * vx + vy * vy + vz * vz;
    if (vsq > vmax_sq) {
        mixed_t s = vmax / sqrt(vsq);
        vx *= s;
        vy *= s;
        vz *= s;
    }
}

__global__ void initial_integrate_limit(
    pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
    pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
    mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
    mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
    const fixed_t* __restrict__ fy, const fixed_t* __restrict__ fz,
    const mixed_t* __restrict__ mass, ForceFault* fault,
    const std::int64_t* __restrict__ tag, long long step, cuda::Box box,
    mixed_t dt, mixed_t vmax, mixed_t vmax_sq, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const mixed_t scale =
        mixed_t(0.5) * dt * mixed_t(ATV::units::force_to_accel) / mass[i];
    mixed_t vxi = vx[i] + scale * decode_force<mixed_t>(fx[i]);
    mixed_t vyi = vy[i] + scale * decode_force<mixed_t>(fy[i]);
    mixed_t vzi = vz[i] + scale * decode_force<mixed_t>(fz[i]);
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
    mixed_t px = load_position(x, xc, i) + dt * vxi;
    mixed_t py = load_position(y, yc, i) + dt * vyi;
    mixed_t pz = load_position(z, zc, i) + dt * vzi;
    // PBC wrap by whole lattice vectors (cached inverse cell, no division).
    box.wrap_position(px, py, pz);
    store_position(x, xc, i, px);
    store_position(y, yc, i, py);
    store_position(z, zc, i, pz);
}

__global__ void
final_integrate_limit(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                      mixed_t* __restrict__ vz, const fixed_t* __restrict__ fx,
                      const fixed_t* __restrict__ fy,
                      const fixed_t* __restrict__ fz,
                      const mixed_t* __restrict__ mass, ForceFault* fault,
                      const std::int64_t* __restrict__ tag, long long step,
                      mixed_t dt, mixed_t vmax, mixed_t vmax_sq, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const mixed_t scale =
        mixed_t(0.5) * dt * mixed_t(ATV::units::force_to_accel) / mass[i];
    mixed_t vxi = vx[i] + scale * decode_force<mixed_t>(fx[i]);
    mixed_t vyi = vy[i] + scale * decode_force<mixed_t>(fy[i]);
    mixed_t vzi = vz[i] + scale * decode_force<mixed_t>(fz[i]);
    apply_cap(vxi, vyi, vzi, vmax, vmax_sq);
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}

} // anonymous namespace

void launch_nve_limit_initial(Frame& frame, int n, const cuda::Box& box,
                              mixed_t dt, mixed_t xmax, long long step,
                              cudaStream_t stream) {
    if (n <= 0)
        return;
    const mixed_t vmax = xmax / dt;
    auto& a = frame.atoms;
    initial_integrate_limit<<<cuda::grid_size(n), cuda::kDefaultBlock, 0,
                              stream>>>(a.x, a.y, a.z, a.x_corr, a.y_corr,
                                        a.z_corr, a.vx, a.vy, a.vz, a.fx, a.fy,
                                        a.fz, a.mass, frame.force_fault, a.tag,
                                        step, box, dt, vmax, vmax * vmax, n);
}

void launch_nve_limit_final(Frame& frame, int n, mixed_t dt, mixed_t xmax,
                            long long step, cudaStream_t stream) {
    if (n <= 0)
        return;
    const mixed_t vmax = xmax / dt;
    auto& a = frame.atoms;
    final_integrate_limit<<<cuda::grid_size(n), cuda::kDefaultBlock, 0,
                            stream>>>(a.vx, a.vy, a.vz, a.fx, a.fy, a.fz,
                                      a.mass, frame.force_fault, a.tag, step,
                                      dt, vmax, vmax * vmax, n);
}

void NVELimit::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const int n = ctx.frame.n_atoms();
    if (ctx.at(ATV::md::Phase::initial_integrate))
        launch_nve_limit_initial(ctx.frame, n, ctx.runtime.host_box,
                                 mixed_t(dt_), mixed_t(xmax_), ctx.step,
                                 stream);
    else
        launch_nve_limit_final(ctx.frame, n, mixed_t(dt_), mixed_t(xmax_),
                               ctx.step, stream);
}

} // namespace md
} // namespace cuda
} // namespace ATV
