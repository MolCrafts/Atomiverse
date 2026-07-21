/// @file md/velocity_verlet.cu

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/facade/stage.h"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

namespace {

__global__ void initial_integrate(
    real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
    real_t* __restrict__ vx, real_t* __restrict__ vy, real_t* __restrict__ vz,
    const real_t* __restrict__ fx, const real_t* __restrict__ fy,
    const real_t* __restrict__ fz, const real_t* __restrict__ mass,
    cuda::Box box, work_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work_t inv_m = work_t(1) / static_cast<work_t>(mass[i]);
    work_t hdt = work_t(0.5) * dt;
    work_t ax = static_cast<work_t>(fx[i]) * inv_m * kFtm2v;
    work_t ay = static_cast<work_t>(fy[i]) * inv_m * kFtm2v;
    work_t az = static_cast<work_t>(fz[i]) * inv_m * kFtm2v;
    vx[i] += hdt * ax;
    vy[i] += hdt * ay;
    vz[i] += hdt * az;
    x[i] += dt * vx[i];
    y[i] += dt * vy[i];
    z[i] += dt * vz[i];
    // PBC wrap via cached inv_l on the by-value Box — no per-thread
    // 1/Lx FP64 division, no widely-spaced box[0]/box[4]/box[8] loads
    // (cuda-md-review-cleanup-04-box-by-value-pbc).
    work_t Lx = work_t(box.cell[0]), Ly = work_t(box.cell[4]),
           Lz = work_t(box.cell[8]);
    work_t inv_Lx = work_t(box.inv_l[0]), inv_Ly = work_t(box.inv_l[1]),
           inv_Lz = work_t(box.inv_l[2]);
    x[i] -= Lx * floor(x[i] * inv_Lx);
    y[i] -= Ly * floor(y[i] * inv_Ly);
    z[i] -= Lz * floor(z[i] * inv_Lz);
}

__global__ void
final_integrate(real_t* __restrict__ vx, real_t* __restrict__ vy,
                real_t* __restrict__ vz, const real_t* __restrict__ fx,
                const real_t* __restrict__ fy, const real_t* __restrict__ fz,
                const real_t* __restrict__ mass, work_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    work_t inv_m = work_t(1) / static_cast<work_t>(mass[i]);
    work_t hdt = work_t(0.5) * dt;
    vx[i] += hdt * static_cast<work_t>(fx[i]) * inv_m * kFtm2v;
    vy[i] += hdt * static_cast<work_t>(fy[i]) * inv_m * kFtm2v;
    vz[i] += hdt * static_cast<work_t>(fz[i]) * inv_m * kFtm2v;
}

} // anonymous namespace

void VelocityVerlet::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        initial_integrate<<<grid, block, 0, stream>>>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.vx,
            frame.atoms.vy, frame.atoms.vz, frame.atoms.fx, frame.atoms.fy,
            frame.atoms.fz, frame.atoms.mass, ctx.runtime.host_box, dt_, n);
    } else {
        final_integrate<<<grid, block, 0, stream>>>(
            frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
            frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, dt_, n);
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
