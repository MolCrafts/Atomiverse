/// @file cuda/planewave/internal/fft.cuh
/// @brief RAII wrapper around cuFFT 3D transforms — device-resident plane-wave
///        FFT, the CUDA mirror of cpu::pw::internal::FFTPlan3D.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief 3D FFT plans (cuFFT) for the plane-wave SCF grid.
///
/// Layout and normalization match cpu::pw::internal::FFTPlan3D exactly so the
/// two backends are numerically interchangeable: the fastest-varying axis is x
/// (logical index `ix + nx*(iy + ny*iz)`), the real grid holds `nr = nx*ny*nz`
/// points, and the Hermitian half-complex grid holds `nc = (nx/2+1)*ny*nz`.
/// Transforms are **unnormalized** (a forward+backward round trip scales by
/// `nr`), identical to FFTW and cuFFT defaults; callers divide by `nr`.
///
/// Unlike the CPU class, every entry point takes **device** pointers — data
/// stays resident on the GPU between transforms per the no-implicit-transfer
/// rule. Double precision (D2Z / Z2D / Z2Z) matches the CPU `fftw_complex`
/// scratch and the energy tolerances of the SCF.
class FFTPlan3D {
    cufftHandle r2c_ = 0; ///< real → Hermitian complex (D2Z)
    cufftHandle c2r_ = 0; ///< Hermitian complex → real (Z2D)
    cufftHandle c2c_ = 0; ///< full complex → complex (Z2Z), fwd and bwd
    int nx_, ny_, nz_;
    int nr_, nc_;                          ///< real size, half-complex size
    cudaStream_t stream_ = nullptr;        ///< stream all plans execute on
    cufftDoubleComplex* d_cbuf_ = nullptr; ///< nc scratch (preserves c2r in)

  public:
    /// @brief Build the three plans for an nx×ny×nz grid.
    /// @param nx Fast (contiguous) grid dimension.
    /// @param ny Middle grid dimension.
    /// @param nz Slow grid dimension.
    /// @param stream CUDA stream the plans execute on (default stream if null).
    FFTPlan3D(int nx, int ny, int nz, cudaStream_t stream = nullptr);
    ~FFTPlan3D();
    FFTPlan3D(const FFTPlan3D&) = delete;
    FFTPlan3D& operator=(const FFTPlan3D&) = delete;

    /// @brief Real → Hermitian complex (out-of-place). @param d_in nr reals
    ///        (device). @param d_out nc complex (device).
    void r2c(const double* d_in, cufftDoubleComplex* d_out);
    /// @brief Hermitian complex → real (out-of-place, UNNORMALIZED). Input is
    ///        preserved (copied into internal scratch first, mirroring the CPU
    ///        class). @param d_in nc complex (device). @param d_out nr reals.
    void c2r(const cufftDoubleComplex* d_in, double* d_out);
    /// @brief Full complex → complex, forward, in place. @param d_inout nr
    ///        complex (device).
    void c2c_fwd(cufftDoubleComplex* d_inout);
    /// @brief Full complex → complex, backward, in place. @param d_inout nr
    ///        complex (device).
    void c2c_bwd(cufftDoubleComplex* d_inout);

    /// @brief Rebind every plan to a new stream.
    void set_stream(cudaStream_t stream);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int nr() const { return nr_; }
    int nc() const { return nc_; }
};

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
