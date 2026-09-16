/// @file cuda/planewave/internal/veff.cuh
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
/// Matches cpu::pw::internal::VeffBuilder::step. **All three inputs are
/// complex**: `ws.V_loc_G` carries the structure factor Σ_a e^{-iG·R_a}
/// (assembled in cpu/planewave/internal/pw_setup.cpp), whose imaginary part is
/// nonzero for any cell without inversion symmetry about the origin — the
/// 2-atom fcc Si basis every PW test uses is exactly such a cell. Treating
/// V_loc as a real array (the historical signature here) silently dropped
/// Im[V_loc] from V_eff.
///
/// @param d_Vloc_G V_loc(G) incl. structure factor, n_g complex (device)
///                 [Hartree].
/// @param d_VH_G   Hartree potential V_H(G), n_g complex (device) [Hartree].
/// @param d_Vxc_G  XC potential V_xc(G), n_g complex (device) [Hartree].
/// @param d_Veff_G Output V_eff(G), n_g complex (device). Must NOT alias any
///                 input: the kernel marks all four pointers `__restrict__`,
///                 so an overlapping range is undefined behaviour.
/// @param n_g      Number of G-vectors.
/// @param stream   CUDA stream (default stream if null).
void veff_assemble_g(const cufftDoubleComplex* d_Vloc_G,
                     const cufftDoubleComplex* d_VH_G,
                     const cufftDoubleComplex* d_Vxc_G,
                     cufftDoubleComplex* d_Veff_G, int n_g,
                     cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
