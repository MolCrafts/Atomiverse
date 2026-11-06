/// @file cuda/potentials/lj_kernel.cuh
/// @brief LJ shift shared by cuda::PairLJCut and its tests.
#pragma once

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace kernels {

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
