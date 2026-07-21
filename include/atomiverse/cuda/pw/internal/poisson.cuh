/// @file cuda/pw/internal/poisson.cuh
/// @brief Reciprocal-space Hartree (Poisson) solver — device kernel, the CUDA
///        mirror of cpu::pw::PW_Poisson::launch.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Solve the Poisson equation in reciprocal space: V_H(G) = 4π ρ(G)/|G|².
///
/// Elementwise over the G-vector grid, exactly matching cpu::pw::PW_Poisson:
/// the G=0 component (where |G|² < 1e-16) is set to zero (the neutralizing
/// background), every other component is `4π ρ(G) / |G|²`. Pure, trivially
/// parallel, device-resident — no host transfer.
///
/// @param d_rho_G  ρ(G), n_g complex (device, Hartree-units density in G-space).
/// @param d_g2     |G|² for each G-vector, n_g reals (device, Å⁻² / a.u.⁻²).
/// @param d_V_H_G  Output V_H(G), n_g complex (device). May alias d_rho_G.
/// @param n_g      Number of G-vectors.
/// @param stream   CUDA stream to launch on (default stream if null).
void poisson_hartree(const cufftDoubleComplex* d_rho_G, const double* d_g2,
                     cufftDoubleComplex* d_V_H_G, int n_g,
                     cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
