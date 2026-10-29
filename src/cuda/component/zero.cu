#include "atomiverse/cuda/component/zero.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include <cstddef>
#include <cuda_runtime.h>
namespace ATV::cuda {
// ── Zero initializers ────────────────────────────────────────────────

void ZeroForces::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    if (n > 0) {
        // fx|fy|fz is one block of stride capacity: clear through fz[n).
        const std::size_t rows =
            2 * static_cast<std::size_t>(frame.capacity()) + n;
        ATV_CUDA_CHECK(
            cudaMemsetAsync(frame.atoms.fx, 0, rows * sizeof(fixed_t), stream));
    }
}

void ZeroEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // The total and every per-term energy (pair .. improper).
    cudaMemsetAsync(&frame.scalars->energy, 0,
                    offsetof(SystemScalars, kinetic)
                        - offsetof(SystemScalars, energy),
                    stream);
}

void ZeroStress::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Without this clear, force kernels (LJ + bond/angle/dihedral)
    // accumulate virial across steps and `NoseHooverNPT` reads runaway
    // pressure.
    cudaMemsetAsync(&frame.scalars->stress[0], 0, 6 * sizeof(double), stream);
}

void ZeroScalars::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    cudaMemsetAsync(ctx.frame.scalars, 0, sizeof(SystemScalars), stream);
}

} // namespace ATV::cuda
