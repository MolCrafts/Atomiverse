/// @file cpu/planewave/internal/fft.cpp
/// @brief FFTW3 wrapper implementation.

#include "atomiverse/cpu/planewave/internal/fft.h"
#include <cstring>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

FFTPlan3D::FFTPlan3D(int nx, int ny, int nz) : nx_(nx), ny_(ny), nz_(nz) {
    nr_ = nx * ny * nz;
    nc_ = (nx / 2 + 1) * ny * nz;
    rbuf_ = fftw_alloc_real(static_cast<size_t>(nr_));
    cbuf_ = fftw_alloc_complex(static_cast<size_t>(nc_));
    cbuf2_ = fftw_alloc_complex(static_cast<size_t>(nr_)); // full complex grid
    if (!rbuf_ || !cbuf_ || !cbuf2_) {
        if (rbuf_)
            fftw_free(rbuf_);
        if (cbuf_)
            fftw_free(cbuf_);
        if (cbuf2_)
            fftw_free(cbuf2_);
        throw std::runtime_error("FFTPlan3D: allocation failed");
    }
    fwd_ = fftw_plan_dft_r2c_3d(nz, ny, nx, rbuf_, cbuf_, FFTW_ESTIMATE);
    bwd_ = fftw_plan_dft_c2r_3d(nz, ny, nx, cbuf_, rbuf_, FFTW_ESTIMATE);
    fwd_cc_ = fftw_plan_dft_3d(nz, ny, nx, cbuf2_, cbuf2_, FFTW_FORWARD,
                               FFTW_ESTIMATE);
    bwd_cc_ = fftw_plan_dft_3d(nz, ny, nx, cbuf2_, cbuf2_, FFTW_BACKWARD,
                               FFTW_ESTIMATE);
}

FFTPlan3D::~FFTPlan3D() {
    if (fwd_)
        fftw_destroy_plan(fwd_);
    if (bwd_)
        fftw_destroy_plan(bwd_);
    if (fwd_cc_)
        fftw_destroy_plan(fwd_cc_);
    if (bwd_cc_)
        fftw_destroy_plan(bwd_cc_);
    if (rbuf_)
        fftw_free(rbuf_);
    if (cbuf_)
        fftw_free(cbuf_);
    if (cbuf2_)
        fftw_free(cbuf2_);
}

void FFTPlan3D::r2c(const AccReal* in, fftw_complex* out) {
    std::memcpy(rbuf_, in, static_cast<size_t>(nr_) * sizeof(AccReal));
    fftw_execute_dft_r2c(fwd_, rbuf_, out);
}

void FFTPlan3D::c2r(const fftw_complex* in, AccReal* out) {
    std::memcpy(cbuf_, in, static_cast<size_t>(nc_) * sizeof(fftw_complex));
    fftw_execute_dft_c2r(bwd_, cbuf_, rbuf_);
    std::memcpy(out, rbuf_, static_cast<size_t>(nr_) * sizeof(AccReal));
}

void FFTPlan3D::c2c_fwd(fftw_complex* inout) {
    fftw_execute_dft(fwd_cc_, inout, inout);
}

void FFTPlan3D::c2c_bwd(fftw_complex* inout) {
    fftw_execute_dft(bwd_cc_, inout, inout);
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
