/// @file cuda/minimize/sd.cu
/// @brief cuda::minimizer::SteepestDescent kernel: x += alpha * f / m * ATV::units::force_to_accel.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::real_t;

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
__global__ void
sd_step(real_t* __restrict__ x, real_t* __restrict__ y, real_t* __restrict__ z,
        real_t* __restrict__ xcor, real_t* __restrict__ ycor,
        real_t* __restrict__ zcor, std::int8_t* __restrict__ image,
        const real_t* __restrict__ fx, const real_t* __restrict__ fy,
        const real_t* __restrict__ fz, const motion_t* __restrict__ mass,
        cuda::MotionBox box, motion_t alpha, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    motion_t inv_m = motion_t(1) / static_cast<motion_t>(mass[i]);
    wrap_position(
        x, y, z, xcor, ycor, zcor, image, i, box,
        load_position(x, xcor, i)
            + alpha * motion_t(fx[i]) * inv_m * ATV::units::force_to_accel,
        load_position(y, ycor, i)
            + alpha * motion_t(fy[i]) * inv_m * ATV::units::force_to_accel,
        load_position(z, zcor, i)
            + alpha * motion_t(fz[i]) * inv_m * ATV::units::force_to_accel);
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
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    sd_step<<<grid, block, 0, stream>>>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, frame.atoms.x_correction,
        frame.atoms.y_correction, frame.atoms.z_correction, frame.atoms.image,
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.mass,
        ctx.runtime.host_motion_box, alpha_, n);
    ATV_CUDA_CHECK(cudaGetLastError());
    ++step_;
}

void SteepestDescent::destroy(const DestroyContext& ctx) {
    (void)ctx;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
