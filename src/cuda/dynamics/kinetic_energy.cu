/// @file cuda/dynamics/kinetic_energy.cu
/// @brief KineticEnergy kernel: per-atom KE with warp-shuffle block
///        reduction. Also writes the kinetic stress tensor
///        Σ m·vα·vβ (eV units, no volume division) into
///        `frame.scalars->stress_kinetic[6]` so `cuda::Virial`
///        no longer needs its own kernel launch
///        (cuda-md-review-cleanup-05-reductions-and-atomics).

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/kinetic_energy.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

namespace {

/// @brief Per-atom KE + kinetic-stress 7-scalar packed reduction.
///
/// Each thread computes the 7-scalar contribution
/// `{ke, mvxx, mvyy, mvzz, mvxy, mvxz, mvyz}` for one atom (zeros for
/// out-of-range), then `cuda::block_reduce_sum_n` collapses them in a
/// single warp-shuffle pass; thread 0 atomic-adds the 7 partials to
/// the global accumulators. Replaces the legacy two-kernel pattern
/// (`kinetic_energy` + `kinetic_stress_k`, both retired by
/// `cuda-md-review-cleanup-05-reductions-and-atomics`) — the second
/// launch + its 12 KiB shared-mem sweep are eliminated.
///
/// Stress writes Σ m·vα·vβ (ATV::units::mass_v2_to_energy folded in, NO volume division);
/// the consumer (`cuda::md::NoseHooverNPT::update_eps_dot_` /
/// pressure observers) divides by V.
__global__
__launch_bounds__(ATV::cuda::kDefaultBlock, 4) void kinetic_energy_stress(
    const real_t* __restrict__ vx, const real_t* __restrict__ vy,
    const real_t* __restrict__ vz, const real_t* __restrict__ mass,
    prec_t* __restrict__ kinetic, prec_t* __restrict__ stress_kinetic, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    work_t my_ke = work_t(0);
    work_t my_s[6] = {};
    if (i < n) {
        work_t m = static_cast<work_t>(mass[i]);
        work_t vxi = static_cast<work_t>(vx[i]);
        work_t vyi = static_cast<work_t>(vy[i]);
        work_t vzi = static_cast<work_t>(vz[i]);
        work_t v2 = vxi * vxi + vyi * vyi + vzi * vzi;
        my_ke = work_t(0.5) * ATV::units::mass_v2_to_energy * m * v2;
        if (stress_kinetic != nullptr) {
            my_s[0] = m * vxi * vxi; // xx
            my_s[1] = m * vyi * vyi; // yy
            my_s[2] = m * vzi * vzi; // zz
            my_s[3] = m * vxi * vyi; // xy
            my_s[4] = m * vxi * vzi; // xz
            my_s[5] = m * vyi * vzi; // yz
        }
    }

    extern __shared__ work_t warp_scratch[]; // per-warp staging cells
    work_t vals[7] = {my_ke,   my_s[0], my_s[1], my_s[2],
                      my_s[3], my_s[4], my_s[5]};
    work_t out[7];
    ATV::cuda::block_reduce_sum_n<7>(vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(kinetic, static_cast<prec_t>(out[0]));
        if (stress_kinetic != nullptr) {
            const work_t mvv2e =
                static_cast<work_t>(ATV::units::mass_v2_to_energy);
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress_kinetic[c],
                          static_cast<prec_t>(out[c + 1] * mvv2e));
        }
    }
}

} // anonymous namespace

void KineticEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));
    int shmem = cuda::warps_per_block(block) * sizeof(work_t);
    // Fused KE + kinetic-stress reduction. Passes
    // `frame.scalars->stress_kinetic` so a co-bound `Virial`
    // doesn't need its own kernel launch — that component now
    // becomes a zero-kernel auto-injection that just declares the
    // capability (cuda-md-review-cleanup-05-reductions-and-atomics).
    kinetic_energy_stress<<<grid, block, shmem, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass,
        &frame.scalars->kinetic,
        frame.scalars != nullptr ? frame.scalars->stress_kinetic : nullptr, n);
}

} // namespace md
} // namespace cuda
} // namespace ATV
