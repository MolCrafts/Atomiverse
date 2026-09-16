/// @file cuda/minimize/sd.cu
/// @brief cuda::minimizer::SteepestDescent kernel: x += alpha * f / m * ATV::units::force_to_accel.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace minimizer {

namespace {

/// @brief Steepest descent position update.
///
/// Moves each atom along the force direction:
///   x[i] += alpha * fx[i] / mass[i] * ATV::units::force_to_accel
///
/// Same unit conversion as VelocityVerlet. At step 0, forces are zero
/// (from initialization), so positions remain unchanged.
__global__ void sd_step(real_t* __restrict__ x, real_t* __restrict__ y,
                        real_t* __restrict__ z, const real_t* __restrict__ fx,
                        const real_t* __restrict__ fy,
                        const real_t* __restrict__ fz,
                        const real_t* __restrict__ mass, cuda::Box box,
                        work_t alpha, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    work_t inv_m = work_t(1) / static_cast<work_t>(mass[i]);
    x[i] = static_cast<real_t>(static_cast<work_t>(x[i])
                               + alpha * static_cast<work_t>(fx[i]) * inv_m
                                     * ATV::units::force_to_accel);
    y[i] = static_cast<real_t>(static_cast<work_t>(y[i])
                               + alpha * static_cast<work_t>(fy[i]) * inv_m
                                     * ATV::units::force_to_accel);
    z[i] = static_cast<real_t>(static_cast<work_t>(z[i])
                               + alpha * static_cast<work_t>(fz[i]) * inv_m
                                     * ATV::units::force_to_accel);
    // PBC wrap via cached inv_l on the by-value Box — same form as the
    // MD integrators (cuda-md-review-cleanup-04-box-by-value-pbc).
    work_t Lx = work_t(box.cell[0]), Ly = work_t(box.cell[4]),
           Lz = work_t(box.cell[8]);
    work_t inv_Lx = work_t(box.inv_l[0]), inv_Ly = work_t(box.inv_l[1]),
           inv_Lz = work_t(box.inv_l[2]);
    work_t x_w = static_cast<work_t>(x[i]);
    work_t y_w = static_cast<work_t>(y[i]);
    work_t z_w = static_cast<work_t>(z[i]);
    x[i] = static_cast<real_t>(x_w - Lx * floor(x_w * inv_Lx));
    y[i] = static_cast<real_t>(y_w - Ly * floor(y_w * inv_Ly));
    z[i] = static_cast<real_t>(z_w - Lz * floor(z_w * inv_Lz));
}

} // anonymous namespace

void SteepestDescent::build(const BuildContext& ctx) {
    step_ = 0;
    trace_.clear();
    conv_.build(ctx.frame.n_atoms());
}

void SteepestDescent::launch(const LaunchContext& ctx) {
    // ── Convergence-metric publish phase ─────────────────────────────
    //
    // No previous geometry is kept (the step is applied in place), so no displacement
    // is measured: `x_prev = nullptr` tells ConvCheck to publish
    // `has_displacement = false`, and the handler leaves the displacement criteria out
    // of its test rather than judging a zero it was never given.
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        conv_.measure(ctx, /*x_prev=*/nullptr, step_, trace_);
        return;
    }

    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    sd_step<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, ctx.runtime.host_box,
        alpha_, n);
    ++step_;
}

void SteepestDescent::destroy(const DestroyContext& ctx) {
    (void)ctx;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
