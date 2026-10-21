/// @file cuda/dynamics/deform.cu
/// @brief Deform -- uniaxial constant-strain-rate deformation.

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

Deform::Deform(int axis, double strain_rate, double dt)
    : axis_(axis), strain_rate_(strain_rate), dt_(dt) {}

std::vector<DefaultBinding> Deform::default_bindings() const {
    // initial_integrate @ +200, matching cpu::md::Deform: the affine box
    // remap runs AFTER the integrator's kick/drift (order 0), so the deformed
    // box wraps the just-updated positions consistently across backends.
    return {ATV::make_binding(ATV::md::Phase::initial_integrate, 200)};
}

void Deform::build(const BuildContext& ctx) {
    // Cache the initial box edge on the deform axis so launch() can
    // compute L_k = L0 · (1 + erate · k · dt). One D2H at build time.
    prec_t host_box[9];
    ATV_CUDA_CHECK(cudaMemcpy(host_box, ctx.frame.box, 9 * sizeof(prec_t),
                              cudaMemcpyDeviceToHost));
    L0_ = static_cast<double>(host_box[axis_ * 3 + axis_]);
    step_count_ = 0;
}

namespace {

__global__ void
deform_k(real_t* __restrict__ pos,
         real_t* __restrict__ correction, // x, y, or z depending on axis
         prec_t* __restrict__ box,        // 3x3 row-major
         int axis, motion_t scale_factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // Remap atom position proportionally
    store_position(pos, correction, idx,
                   load_position(pos, correction, idx) * scale_factor);

    // Thread 0 updates the box diagonal element
    if (idx == 0) {
        int box_idx = axis * 3 + axis;
        box[box_idx] = static_cast<prec_t>(static_cast<motion_t>(box[box_idx])
                                           * scale_factor);
    }
}

} // anonymous namespace

void Deform::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int n = f.n_atoms();
    if (n <= 0)
        return;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    // LAMMPS-equivalent engineering strain: box grows LINEARLY in time
    // (`fix deform x erate value` doc § "Units box remap x"). Compute the
    // target edge L_k = L0·(1 + erate·k·dt) and apply the per-step ratio
    // L_k / L_{k-1} so the box and atom positions land exactly on the
    // engineering-strain ramp. Previous `scale = 1 + erate·dt` compounded
    // to L0·exp(erate·t) — fine at small strain but ~2.05× too long at
    // the stage-5 full strain ε=1.7 (L/L0 = 5.57 vs LAMMPS 2.72).
    ++step_count_;
    const double L_prev = L0_ * (1.0 + strain_rate_ * dt_ * (step_count_ - 1));
    const double L_curr = L0_ * (1.0 + strain_rate_ * dt_ * step_count_);
    motion_t scale = static_cast<motion_t>(L_curr / L_prev);

    real_t* pos = nullptr;
    real_t* correction = nullptr;
    switch (axis_) {
    case 0:
        pos = f.atoms.x;
        correction = f.atoms.x_correction;
        break;
    case 1:
        pos = f.atoms.y;
        correction = f.atoms.y_correction;
        break;
    case 2:
        pos = f.atoms.z;
        correction = f.atoms.z_correction;
        break;
    }

    deform_k<<<grid, block, 0, stream>>>(pos, correction, f.box, axis_, scale,
                                         n);

    ATV_CUDA_CHECK(cudaGetLastError());
    ctx.runtime.host_motion_box.cell[axis_ * 3 + axis_] *= scale;
    ctx.runtime.host_motion_box.refresh();
    ctx.runtime.host_box = box_cast<real_t>(ctx.runtime.host_motion_box);
}

} // namespace md
} // namespace cuda
} // namespace ATV
