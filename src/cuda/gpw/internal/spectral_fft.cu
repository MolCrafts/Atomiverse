/// @file cuda/gpw/internal/spectral_fft.cu
/// @brief Device spectral gradient / divergence (see header).
///
/// Convention matches `cpu::gpw` grid_fft: unnormalized r2c, multiply by i G_d
/// with Nyquist modes of odd derivatives zeroed, unnormalized c2r scaled by
/// 1/N. Layout idx = (i0 * n1 + i1) * n2c + i2.

#include "atomiverse/cuda/gpw/internal/spectral_fft.cuh"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int kBlock = 256;

void check_cufft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuda::gpw SpectralFftDevice: ")
                                 + what + " failed (cufftResult "
                                 + std::to_string(static_cast<int>(r)) + ")");
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::gpw SpectralFftDevice: ")
                                 + what + " failed ("
                                 + cudaGetErrorString(e) + ")");
}

int grid_blocks(long nwork) {
    if (nwork <= 0)
        return 0;
    const long need = (nwork + kBlock - 1) / kBlock;
    constexpr int kMaxBlocks = 2048;
    return static_cast<int>(need < kMaxBlocks ? need : kMaxBlocks);
}

/// Signed reciprocal lattice component with Nyquist zeroed (odd-derivative).
__device__ __forceinline__ double g_component(int i, int n, double g_unit) {
    int m = (i <= n / 2) ? i : i - n;
    // 2|m| == n ⇒ Nyquist mode of odd derivative is pure-imaginary on the
    // Hermitian grid; zero it so the inverse transform stays real (CPU).
    if (2 * (m < 0 ? -m : m) == n)
        m = 0;
    return static_cast<double>(m) * g_unit;
}

/// @brief out = i * G_axis * in  (axis 0/1/2 → gx/gy/gz units).
__global__ void
mul_ig_kernel(const cufftDoubleComplex* __restrict__ in,
              cufftDoubleComplex* __restrict__ out, int n0, int n1, int n2c,
              int n2, int axis, double gx_unit, double gy_unit,
              double gz_unit) {
    const long ncplx = static_cast<long>(n0) * n1 * n2c;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < ncplx; idx += static_cast<long>(blockDim.x) * gridDim.x) {
        const int i2 = static_cast<int>(idx % n2c);
        long t = idx / n2c;
        const int i1 = static_cast<int>(t % n1);
        const int i0 = static_cast<int>(t / n1);

        double g = 0.0;
        if (axis == 0)
            g = g_component(i0, n0, gx_unit);
        else if (axis == 1)
            g = g_component(i1, n1, gy_unit);
        else {
            // r2c last axis: only non-negative freqs; Nyquist when 2*i2 == n2.
            int m2 = i2;
            if (2 * m2 == n2)
                m2 = 0;
            g = static_cast<double>(m2) * gz_unit;
        }

        // i*g*(re + i im) = -g*im + i g*re
        const double re = in[idx].x;
        const double im = in[idx].y;
        out[idx].x = -g * im;
        out[idx].y = g * re;
    }
}

/// @brief acc += i * G_axis * in  (accumulate for divergence).
__global__ void
axpy_ig_kernel(const cufftDoubleComplex* __restrict__ in,
               cufftDoubleComplex* __restrict__ acc, int n0, int n1, int n2c,
               int n2, int axis, double gx_unit, double gy_unit,
               double gz_unit) {
    const long ncplx = static_cast<long>(n0) * n1 * n2c;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < ncplx; idx += static_cast<long>(blockDim.x) * gridDim.x) {
        const int i2 = static_cast<int>(idx % n2c);
        long t = idx / n2c;
        const int i1 = static_cast<int>(t % n1);
        const int i0 = static_cast<int>(t / n1);

        double g = 0.0;
        if (axis == 0)
            g = g_component(i0, n0, gx_unit);
        else if (axis == 1)
            g = g_component(i1, n1, gy_unit);
        else {
            int m2 = i2;
            if (2 * m2 == n2)
                m2 = 0;
            g = static_cast<double>(m2) * gz_unit;
        }

        const double re = in[idx].x;
        const double im = in[idx].y;
        acc[idx].x += -g * im;
        acc[idx].y += g * re;
    }
}

__global__ void scale_field_kernel(double* __restrict__ f, long ntot,
                                   double inv_n) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < ntot; i += static_cast<long>(blockDim.x) * gridDim.x)
        f[i] *= inv_n;
}

} // namespace

SpectralFftDevice::SpectralFftDevice(const GpwGridDesc& grid,
                                     cudaStream_t stream)
    : n0_(grid.n[0]), n1_(grid.n[1]), n2_(grid.n[2]), n2c_(grid.n2c()),
      ntot_(grid.npts()), ncplx_(grid.ncplx()), stream_(stream) {
    if (n0_ <= 0 || n1_ <= 0 || n2_ <= 0)
        throw std::invalid_argument("SpectralFftDevice: non-positive grid");
    if (grid.L[0] <= 0.0 || grid.L[1] <= 0.0 || grid.L[2] <= 0.0)
        throw std::invalid_argument("SpectralFftDevice: non-positive L");

    inv_n_ = 1.0 / static_cast<double>(ntot_);
    gx_unit_ = kTwoPi / grid.L[0];
    gy_unit_ = kTwoPi / grid.L[1];
    gz_unit_ = kTwoPi / grid.L[2];

    try {
        check_cufft(cufftPlan3d(&r2c_, n0_, n1_, n2_, CUFFT_D2Z),
                    "cufftPlan3d D2Z");
        check_cufft(cufftPlan3d(&c2r_, n0_, n1_, n2_, CUFFT_Z2D),
                    "cufftPlan3d Z2D");
        check_cufft(cufftSetStream(r2c_, stream_), "cufftSetStream r2c");
        check_cufft(cufftSetStream(c2r_, stream_), "cufftSetStream c2r");
        const size_t cbytes =
            static_cast<size_t>(ncplx_) * sizeof(cufftDoubleComplex);
        check_cuda(cudaMalloc(&d_c_in_, cbytes), "malloc c_in");
        check_cuda(cudaMalloc(&d_c_work_, cbytes), "malloc c_work");
        check_cuda(cudaMalloc(&d_c_acc_, cbytes), "malloc c_acc");
    } catch (...) {
        if (r2c_)
            cufftDestroy(r2c_);
        if (c2r_)
            cufftDestroy(c2r_);
        if (d_c_in_)
            cudaFree(d_c_in_);
        if (d_c_work_)
            cudaFree(d_c_work_);
        if (d_c_acc_)
            cudaFree(d_c_acc_);
        r2c_ = c2r_ = 0;
        d_c_in_ = d_c_work_ = d_c_acc_ = nullptr;
        throw;
    }
}

SpectralFftDevice::~SpectralFftDevice() {
    if (r2c_)
        cufftDestroy(r2c_);
    if (c2r_)
        cufftDestroy(c2r_);
    if (d_c_in_)
        cudaFree(d_c_in_);
    if (d_c_work_)
        cudaFree(d_c_work_);
    if (d_c_acc_)
        cudaFree(d_c_acc_);
}

void SpectralFftDevice::gradient(const double* d_f, double* d_gx, double* d_gy,
                                 double* d_gz) {
    if (!d_f || !d_gx || !d_gy || !d_gz)
        throw std::invalid_argument("SpectralFftDevice::gradient: null ptr");

    check_cufft(cufftExecD2Z(r2c_, const_cast<double*>(d_f), d_c_in_),
                "cufftExecD2Z gradient");

    double* outs[3] = {d_gx, d_gy, d_gz};
    const int gblocks = grid_blocks(ncplx_);
    const int rblocks = grid_blocks(ntot_);
    for (int axis = 0; axis < 3; ++axis) {
        mul_ig_kernel<<<gblocks, kBlock, 0, stream_>>>(
            d_c_in_, d_c_work_, n0_, n1_, n2c_, n2_, axis, gx_unit_, gy_unit_,
            gz_unit_);
        check_cuda(cudaGetLastError(), "mul_ig_kernel");
        // cuFFT Z2D destroys input; work is recomputed each axis so no preserve.
        check_cufft(cufftExecZ2D(c2r_, d_c_work_, outs[axis]),
                    "cufftExecZ2D gradient");
        scale_field_kernel<<<rblocks, kBlock, 0, stream_>>>(outs[axis], ntot_,
                                                            inv_n_);
        check_cuda(cudaGetLastError(), "scale gradient");
    }
}

void SpectralFftDevice::divergence(const double* d_vx, const double* d_vy,
                                   const double* d_vz, double* d_div) {
    if (!d_vx || !d_vy || !d_vz || !d_div)
        throw std::invalid_argument("SpectralFftDevice::divergence: null ptr");

    check_cuda(cudaMemsetAsync(d_c_acc_, 0,
                               static_cast<size_t>(ncplx_)
                                   * sizeof(cufftDoubleComplex),
                               stream_),
               "zero div acc");

    const double* comps[3] = {d_vx, d_vy, d_vz};
    const int gblocks = grid_blocks(ncplx_);
    const int rblocks = grid_blocks(ntot_);
    for (int axis = 0; axis < 3; ++axis) {
        check_cufft(cufftExecD2Z(r2c_, const_cast<double*>(comps[axis]),
                                 d_c_in_),
                    "cufftExecD2Z divergence");
        axpy_ig_kernel<<<gblocks, kBlock, 0, stream_>>>(
            d_c_in_, d_c_acc_, n0_, n1_, n2c_, n2_, axis, gx_unit_, gy_unit_,
            gz_unit_);
        check_cuda(cudaGetLastError(), "axpy_ig_kernel");
    }

    // Preserve acc: Z2D destroys input.
    check_cuda(cudaMemcpyAsync(d_c_work_, d_c_acc_,
                               static_cast<size_t>(ncplx_)
                                   * sizeof(cufftDoubleComplex),
                               cudaMemcpyDeviceToDevice, stream_),
               "preserve div G");
    check_cufft(cufftExecZ2D(c2r_, d_c_work_, d_div), "cufftExecZ2D div");
    scale_field_kernel<<<rblocks, kBlock, 0, stream_>>>(d_div, ntot_, inv_n_);
    check_cuda(cudaGetLastError(), "scale div");
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
