/// @file cuda/pw/internal/grid_map.cuh
/// @brief Sparse G-vector ↔ dense Hermitian half-complex FFT-grid mapping —
///        the device bridge shared by the CUDA pw LDA / Veff / density kernels,
///        mirroring the half-complex pad/gather loops in cpu::pw::internal::XCBuilder and
///        cpu::pw::internal::VeffBuilder.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Scatter a sparse G-space array onto the dense r2c half-complex grid.
///
/// Each G-vector `ig` carries Miller indices (h,k,l); they fold onto the
/// half-complex grid index `kx + (nx/2+1)*(ky + ny*kz)` exactly as the CPU pad
/// loops do, including the Hermitian fold for `kx > nx/2` (mirror the index and
/// conjugate the stored imaginary part). The destination grid (`nc =
/// (nx/2+1)*ny*nz` complex) is zeroed first, so unreferenced cells are 0 —
/// matching the CPU `for i in nc: vg[i] = 0` prelude.
///
/// Race-free: a Hermitian pair (G, −G) folds to the same cell with identical
/// stored values, so concurrent writes agree.
///
/// @param d_gspace      n_g complex G-space values (device).
/// @param d_h,d_k,d_l   n_g Miller indices each (device).
/// @param n_g           Number of G-vectors.
/// @param d_halfcomplex Output dense half-complex grid, nc complex (device).
/// @param nx,ny,nz      FFT grid dimensions (x fastest).
/// @param stream        CUDA stream (default stream if null).
void scatter_g_to_halfcomplex(const cufftDoubleComplex* d_gspace,
                              const int* d_h, const int* d_k, const int* d_l,
                              int n_g, cufftDoubleComplex* d_halfcomplex,
                              int nx, int ny, int nz,
                              cudaStream_t stream = nullptr);

/// @brief Gather from the dense half-complex grid back to the sparse G-space
///        array (inverse of scatter), applying the same Hermitian fold and an
///        optional uniform scale.
///
/// Mirrors the r2c-output gather in cpu::pw::internal::XCBuilder (which scales V_xc(G) by
/// 1/nr). Out-of-range folds write 0, matching the CPU guard.
///
/// @param d_halfcomplex Dense half-complex grid, nc complex (device).
/// @param d_h,d_k,d_l   n_g Miller indices each (device).
/// @param n_g           Number of G-vectors.
/// @param d_gspace      Output n_g complex G-space values (device).
/// @param nx,ny,nz      FFT grid dimensions (x fastest).
/// @param scale         Uniform factor applied to every gathered value.
/// @param stream        CUDA stream (default stream if null).
void gather_halfcomplex_to_g(const cufftDoubleComplex* d_halfcomplex,
                             const int* d_h, const int* d_k, const int* d_l,
                             int n_g, cufftDoubleComplex* d_gspace, int nx,
                             int ny, int nz, double scale = 1.0,
                             cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
