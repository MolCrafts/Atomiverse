/// @file cuda/planewave/internal/fft.cu
/// @brief cuFFT wrapper implementation — CUDA mirror of the FFTW3 plane-wave
///        FFT. Built only with ATV_BUILD_PW && ATV_BUILD_CUDA.

#include "atomiverse/cuda/planewave/internal/fft.cuh"
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {
void check_cufft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuda::pw FFTPlan3D: ") + what
                                 + " failed (cufftResult "
                                 + std::to_string(static_cast<int>(r)) + ")");
}
void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::pw FFTPlan3D: ") + what
                                 + " failed (" + cudaGetErrorString(e) + ")");
}
} // namespace

FFTPlan3D::FFTPlan3D(int nx, int ny, int nz, cudaStream_t stream)
    : nx_(nx), ny_(ny), nz_(nz), stream_(stream) {
    nr_ = nx * ny * nz;
    nc_ = (nx / 2 + 1) * ny * nz;

    // cuFFT takes dims slowest-first; pass (nz, ny, nx) so the contiguous axis
    // is x — matching FFTW's fftw_plan_dft_*_3d(nz, ny, nx, ...) in the CPU
    // class and the `ix + nx*(iy + ny*iz)` indexing used throughout pw_solver.
    try {
        check_cufft(cufftPlan3d(&r2c_, nz, ny, nx, CUFFT_D2Z),
                    "cufftPlan3d D2Z");
        check_cufft(cufftPlan3d(&c2r_, nz, ny, nx, CUFFT_Z2D),
                    "cufftPlan3d Z2D");
        check_cufft(cufftPlan3d(&c2c_, nz, ny, nx, CUFFT_Z2Z),
                    "cufftPlan3d Z2Z");
        check_cuda(cudaMalloc(&d_cbuf_, static_cast<size_t>(nc_)
                                            * sizeof(cufftDoubleComplex)),
                   "cudaMalloc scratch");
    } catch (...) {
        if (r2c_)
            cufftDestroy(r2c_);
        if (c2r_)
            cufftDestroy(c2r_);
        if (c2c_)
            cufftDestroy(c2c_);
        if (d_cbuf_)
            cudaFree(d_cbuf_);
        throw;
    }
    set_stream(stream_);
}

FFTPlan3D::~FFTPlan3D() {
    if (r2c_)
        cufftDestroy(r2c_);
    if (c2r_)
        cufftDestroy(c2r_);
    if (c2c_)
        cufftDestroy(c2c_);
    if (d_cbuf_)
        cudaFree(d_cbuf_);
}

void FFTPlan3D::set_stream(cudaStream_t stream) {
    stream_ = stream;
    check_cufft(cufftSetStream(r2c_, stream_), "cufftSetStream r2c");
    check_cufft(cufftSetStream(c2r_, stream_), "cufftSetStream c2r");
    check_cufft(cufftSetStream(c2c_, stream_), "cufftSetStream c2c");
}

void FFTPlan3D::r2c(const double* d_in, cufftDoubleComplex* d_out) {
    // cuFFT exec is not const-correct but does not modify the D2Z input.
    check_cufft(cufftExecD2Z(r2c_, const_cast<double*>(d_in), d_out),
                "cufftExecD2Z");
}

void FFTPlan3D::c2r(const cufftDoubleComplex* d_in, double* d_out) {
    // cuFFT Z2D destroys its input; copy into scratch first so the caller's
    // buffer is preserved (matches the CPU class's memcpy-into-cbuf_ behavior).
    check_cuda(
        cudaMemcpyAsync(d_cbuf_, d_in,
                        static_cast<size_t>(nc_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_),
        "cudaMemcpyAsync c2r in");
    check_cufft(cufftExecZ2D(c2r_, d_cbuf_, d_out), "cufftExecZ2D");
}

void FFTPlan3D::c2c_fwd(cufftDoubleComplex* d_inout) {
    check_cufft(cufftExecZ2Z(c2c_, d_inout, d_inout, CUFFT_FORWARD),
                "cufftExecZ2Z forward");
}

void FFTPlan3D::c2c_bwd(cufftDoubleComplex* d_inout) {
    check_cufft(cufftExecZ2Z(c2c_, d_inout, d_inout, CUFFT_INVERSE),
                "cufftExecZ2Z inverse");
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
