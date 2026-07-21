/// @file cuda/pw/internal/veff.cuh
/// @brief Reciprocal-space effective-potential assembly — device kernel
///        mirroring the G-space sum in cpu::pw::internal::VeffBuilder::step.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Assemble V_eff(G) = V_loc(G) + V_H(G) + V_xc(G), elementwise.
///
/// Matches cpu::pw::internal::VeffBuilder::step. V_loc is stored as a real array (the
/// local pseudopotential's real G-space part), so it adds only to the real
/// component; V_H and V_xc are complex.
///
/// @param d_Vloc_real V_loc(G) real part, n_g reals (device) [Hartree].
/// @param d_VH_G      Hartree potential V_H(G), n_g complex (device).
/// @param d_Vxc_G     XC potential V_xc(G), n_g complex (device).
/// @param d_Veff_G    Output V_eff(G), n_g complex (device). May alias an input.
/// @param n_g         Number of G-vectors.
/// @param stream      CUDA stream (default stream if null).
void veff_assemble_g(const double* d_Vloc_real,
                     const cufftDoubleComplex* d_VH_G,
                     const cufftDoubleComplex* d_Vxc_G,
                     cufftDoubleComplex* d_Veff_G, int n_g,
                     cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
