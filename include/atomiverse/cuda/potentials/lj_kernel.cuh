/// @file cuda/potentials/lj_kernel.cuh
/// @brief Shared Lennard-Jones constants and cutoff-shift helper.
#pragma once

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace kernels {

/// @brief Minimum pair distance to avoid LJ singularity [Angstrom].
inline constexpr cuda::work_t kMinPairDist = cuda::work_t(1e-10);

/// @brief Compute LJ potential shift at cutoff.
/// @param epsilon Well depth [eV].
/// @param sigma Collision diameter [Angstrom].
/// @param cutoff Interaction cutoff [Angstrom].
/// @return Shift value V(cutoff).
inline double lj_shift(double epsilon, double sigma, double cutoff) {
    double sr = sigma / cutoff;
    double sr2 = sr * sr;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 4.0 * epsilon * (sr12 - sr6);
}

} // namespace kernels
} // namespace cuda
} // namespace ATV
