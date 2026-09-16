/// @file cuda/gpw/internal/spectral_fft.cuh
/// @brief Device spectral gradient / divergence on the GPW uniform grid.
///
/// Mirrors `cpu::gpw::gradient` / `cpu::gpw::divergence` (FFTW r2c last-axis
/// Hermitian, G_d = 2π m_d / L_d, Nyquist of odd derivatives zeroed, c2r / N).
/// Used by the closed-shell PBE path of `KsChainDevice`. Not a full GPW product.
#pragma once

#include "atomiverse/cuda/gpw/internal/poisson_fft.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Device-resident spectral derivative engine for one orthorhombic grid.
///
/// Owns cuFFT r2c/c2r plans and half-complex scratch. Caller owns the real
/// field buffers. All scientific work stays on device.
class SpectralFftDevice {
    int n0_ = 0, n1_ = 0, n2_ = 0, n2c_ = 0;
    long ntot_ = 0, ncplx_ = 0;
    double inv_n_ = 0.0;
    double gx_unit_ = 0.0, gy_unit_ = 0.0, gz_unit_ = 0.0;

    cufftHandle r2c_ = 0;
    cufftHandle c2r_ = 0;
    cudaStream_t stream_ = nullptr;
    cufftDoubleComplex* d_c_in_ = nullptr;  ///< r2c output / preserved fhat
    cufftDoubleComplex* d_c_work_ = nullptr; ///< iG-multiplied / c2r input
    cufftDoubleComplex* d_c_acc_ = nullptr;  ///< divergence G-space accum

  public:
    /// @brief Build plans + scratch for grid shape / cell.
    /// @param grid Orthorhombic grid (n, L in Bohr).
    /// @param stream CUDA stream (default stream if null).
    explicit SpectralFftDevice(const GpwGridDesc& grid,
                               cudaStream_t stream = nullptr);
    ~SpectralFftDevice();

    SpectralFftDevice(const SpectralFftDevice&) = delete;
    SpectralFftDevice& operator=(const SpectralFftDevice&) = delete;

    /// @brief Spectral ∇f → (∂x, ∂y, ∂z) on device.
    /// @param d_f  Real periodic field, length npts.
    /// @param d_gx Output ∂f/∂x, length npts.
    /// @param d_gy Output ∂f/∂y, length npts.
    /// @param d_gz Output ∂f/∂z, length npts.
    void gradient(const double* d_f, double* d_gx, double* d_gy,
                  double* d_gz);

    /// @brief Spectral ∇·v for real periodic vector field components.
    /// @param d_vx, d_vy, d_vz Components, each length npts.
    /// @param d_div Output divergence, length npts.
    void divergence(const double* d_vx, const double* d_vy, const double* d_vz,
                    double* d_div);

    long ntot() const { return ntot_; }
    cudaStream_t stream() const { return stream_; }
};

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
