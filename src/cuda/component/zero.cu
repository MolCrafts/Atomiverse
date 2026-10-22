#include "atomiverse/cuda/component/zero.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include <cuda_runtime.h>
namespace ATV::cuda {
// ── Zero initializers ────────────────────────────────────────────────

void ZeroForces::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    if (n > 0) {
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(fixed_t);
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fx, 0, bytes, stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fy, 0, bytes, stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fz, 0, bytes, stream));
    }
}

void ZeroEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Zero `energy` + 4 per-term fields. The pair/bonded virial lives on
    // `ZeroStress` (same split as `cpu::ZeroEnergy` / `cpu::ZeroStress`).
    cudaMemsetAsync(&frame.scalars->energy, 0, 5 * sizeof(double), stream);
}

void ZeroStress::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Without this clear, force kernels (LJ + bond/angle/dihedral)
    // accumulate virial across steps and `NoseHooverNPT` reads runaway
    // pressure.
    cudaMemsetAsync(&frame.scalars->stress[0], 0, 6 * sizeof(double), stream);
}

void ZeroKinetic::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    cudaMemsetAsync(&frame.scalars->kinetic, 0, sizeof(acc_t), stream);
}

} // namespace ATV::cuda
