#include "atomiverse/core/dynamics_math.h"
/// @file cuda/dynamics/fix_langevin.cu
/// @brief CUDA Langevin thermostat kernel.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits>

namespace ATV {

namespace cuda {
namespace md {

namespace {

// Drag and noise are computed in integration precision and added to the
// fixed-point force like every other classical contribution.
__global__ void
apply_langevin(const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
               const mixed_t* __restrict__ vz, fixed_t* __restrict__ fx,
               fixed_t* __restrict__ fy, fixed_t* __restrict__ fz,
               const mixed_t* __restrict__ mass,
               const std::int64_t* __restrict__ tag, ForceFault* fault,
               std::uint64_t seed, std::uint64_t step, mixed_t T_live,
               mixed_t damp, mixed_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const mixed_t m = mass[i];
    // gfactor1 (drag): `-m / (damp · ftm2v)`. Force = gfactor1 · v.
    const mixed_t gfactor1 = -m / (damp * mixed_t(ATV::units::md_ftm2v));
    // gfactor2 (random): `sqrt(2 m kB T_live / (damp · dt · mvv2e)) / ftm2v`.
    const mixed_t gfactor2_inside =
        mixed_t(2) * m * mixed_t(ATV::units::md_boltz) * T_live
        / (damp * dt * mixed_t(ATV::units::md_mvv2e));
    const mixed_t gfactor2 =
        sqrt(gfactor2_inside) / mixed_t(ATV::units::md_ftm2v);

    // Two normal2_double calls consume eight 32-bit outputs per step.
    // Global tags select subsequences, so sorting/migration needs no RNG state.
    curandStatePhilox4_32_10_t state;
    // Keep a distinct 2^62-output window from initialization and Brownian.
    curand_init(seed, static_cast<unsigned long long>(tag[i]),
                (2ull << 62) + 8ull * step, &state);
    const double2 xy = curand_normal2_double(&state);
    const double2 zw = curand_normal2_double(&state);

    add_force_owned(fx, i, gfactor1 * vx[i] + gfactor2 * mixed_t(xy.x), fault);
    add_force_owned(fy, i, gfactor1 * vy[i] + gfactor2 * mixed_t(xy.y), fault);
    add_force_owned(fz, i, gfactor1 * vz[i] + gfactor2 * mixed_t(zw.x), fault);
}

} // anonymous namespace

void FixLangevin::launch(const LaunchContext& ctx) {
    // ── Host-side T_live update (matches CPU compute_target) ────────
    T_live_ =
        ATV::md::ramp_target(T_start_, T_stop_, ctx.step, nfirst_, nlast_);

    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);

    apply_langevin<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, frame.atoms.tag,
        frame.force_fault, seed_,
        ctx.initializing_forces ? std::numeric_limits<std::uint64_t>::max()
                                : static_cast<std::uint64_t>(ctx.step),
        static_cast<mixed_t>(T_live_), static_cast<mixed_t>(damp_),
        static_cast<mixed_t>(dt_), n);
}

} // namespace md
} // namespace cuda
} // namespace ATV
