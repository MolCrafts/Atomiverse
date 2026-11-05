/// @file cuda/dynamics/kinetic_energy.cu
/// @brief Fused CUB reduction of kinetic energy and six kinetic-stress components.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/kinetic_energy.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

namespace cuda {
namespace md {

namespace {

/// @brief Per-atom KE + kinetic-stress 7-scalar packed reduction.
///
/// Each thread contributes `{ke, mvxx, mvyy, mvzz, mvxy, mvxz, mvyz}`
/// (zeros for tail lanes). CUB reduces the independent components together;
/// thread 0 adds the block's results to the global accumulators.
///
/// Stress writes Σ m·vα·vβ (ATV::units::md_mvv2e folded in, NO volume division);
/// the consumer (`cuda::md::NoseHooverNPT::update_eps_dot_` /
/// pressure observers) divides by V.
__global__
__launch_bounds__(ATV::cuda::kDefaultBlock, 4) void kinetic_energy_stress(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    acc_t* __restrict__ kinetic, acc_t* __restrict__ stress_kinetic, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t my_ke = acc_t(0);
    acc_t my_s[6] = {};
    if (i < n) {
        acc_t m = static_cast<acc_t>(mass[i]);
        acc_t vxi = static_cast<acc_t>(vx[i]);
        acc_t vyi = static_cast<acc_t>(vy[i]);
        acc_t vzi = static_cast<acc_t>(vz[i]);
        acc_t v2 = vxi * vxi + vyi * vyi + vzi * vzi;
        my_ke = acc_t(0.5) * ATV::units::md_mvv2e * m * v2;
        if (stress_kinetic != nullptr) {
            my_s[0] = m * vxi * vxi; // xx
            my_s[1] = m * vyi * vyi; // yy
            my_s[2] = m * vzi * vzi; // zz
            my_s[3] = m * vxi * vyi; // xy
            my_s[4] = m * vxi * vzi; // xz
            my_s[5] = m * vyi * vzi; // yz
        }
    }

    ::cuda::std::array<acc_t, 7> vals = {my_ke,   my_s[0], my_s[1], my_s[2],
                                         my_s[3], my_s[4], my_s[5]};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(vals, SumComponents<acc_t, 7>{});
    if (threadIdx.x == 0) {
        atomicAdd(kinetic, static_cast<acc_t>(out[0]));
        if (stress_kinetic != nullptr) {
            const acc_t mvv2e = static_cast<acc_t>(ATV::units::md_mvv2e);
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress_kinetic[c],
                          static_cast<acc_t>(out[c + 1] * mvv2e));
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
    constexpr int shmem = 0;
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
