/// @file cuda/minimize/sd.cu
/// @brief cuda::minimizer::SteepestDescent kernel: x += alpha * f / m * ftm2v.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace minimizer {

namespace {

/// @brief Steepest descent position update.
///
/// Moves each atom along the force direction:
///   x[i] += alpha * fx[i] / mass[i] * ftm2v
///
/// Same unit conversion as VelocityVerlet. The backend prepares initial forces
/// before the first optimizer step. The update runs in
/// mixed_t on the reconstructed coordinate (primary + correction) with
/// decoded fixed-point forces; a faulted force is never consumed.
__global__ void
sd_step(pos_t* __restrict__ x, pos_t* __restrict__ y, pos_t* __restrict__ z,
        pos_t* __restrict__ xc, pos_t* __restrict__ yc, pos_t* __restrict__ zc,
        const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
        const fixed_t* __restrict__ fz, const mixed_t* __restrict__ mass,
        ImageInt* __restrict__ image, ForceFault* fault,
        const std::int64_t* __restrict__ tag, long long step, cuda::Box box,
        mixed_t alpha, mixed_t ftm2v, int n) {
    if (!forces_consumable(fault, tag, step))
        return;
    box = box.current();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const mixed_t scale = alpha * ftm2v / mass[i];
    mixed_t px = load_position(x, xc, i) + scale * decode_force<mixed_t>(fx[i]);
    mixed_t py = load_position(y, yc, i) + scale * decode_force<mixed_t>(fy[i]);
    mixed_t pz = load_position(z, zc, i) + scale * decode_force<mixed_t>(fz[i]);
    // PBC wrap by whole lattice vectors, counted into the image flags as in
    // the MD integrators (r + H·image follows the relaxation).
    box.wrap_position(px, py, pz, image ? image + 3 * i : nullptr);
    store_position(x, xc, i, px);
    store_position(y, yc, i, py);
    store_position(z, zc, i, pz);
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
    const auto& a = frame.atoms;
    int n = ctx.frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    sd_step<<<grid, block, 0, stream>>>(
        a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.fx, a.fy, a.fz, a.mass,
        a.image, frame.force_fault, a.tag, ctx.step, ctx.runtime.host_box,
        mixed_t(alpha_), mixed_t(ctx.runtime.units.ftm2v), n);
    ++step_;
}

void SteepestDescent::destroy(const DestroyContext& ctx) {
    (void)ctx;
    conv_.destroy();
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
