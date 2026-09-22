/// @file cuda/gpw/internal/multigrid_transfer.cu
/// @brief Device spectral prolong / restrict (see header).
///
/// Convention matches `cpu::gpw::prolong_fft` / `restrict_fft`: unnormalized
/// c2c forward/backward (cuFFT Z2Z ≡ FFTW c2c), mode embedding by signed
/// frequency with Nyquist half/half split, prolong spectral scale Nf/Nc,
/// both paths inverse-normalize by 1/Nf.

#include "atomiverse/cuda/gpw/internal/multigrid_transfer.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr int kBlock = 256;

void check_cufft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(
            std::string("cuda::gpw MultigridTransferDevice: ") + what
            + " failed (cufftResult " + std::to_string(static_cast<int>(r))
            + ")");
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(
            std::string("cuda::gpw MultigridTransferDevice: ") + what
            + " failed (" + cudaGetErrorString(e) + ")");
}

int grid_blocks(long nwork) {
    if (nwork <= 0)
        return 0;
    const long need = (nwork + kBlock - 1) / kBlock;
    constexpr int kMaxBlocks = 2048;
    return static_cast<int>(need < kMaxBlocks ? need : kMaxBlocks);
}

/// Pack real → complex (imag = 0).
__global__ void pack_real_kernel(const double* __restrict__ in,
                                 cufftDoubleComplex* __restrict__ out, long n) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<long>(blockDim.x) * gridDim.x) {
        out[i].x = in[i];
        out[i].y = 0.0;
    }
}

/// Unpack complex → real * scale (take real part).
__global__ void unpack_scale_kernel(const cufftDoubleComplex* __restrict__ in,
                                    double* __restrict__ out, long n,
                                    double scale) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<long>(blockDim.x) * gridDim.x)
        out[i] = in[i].x * scale;
}

__global__ void zero_cplx_kernel(cufftDoubleComplex* __restrict__ buf, long n) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<long>(blockDim.x) * gridDim.x) {
        buf[i].x = 0.0;
        buf[i].y = 0.0;
    }
}

/// Per-axis coarse→fine map (device). Returns count 1 or 2.
__device__ __forceinline__ int axis_coarse_to_fine(int ic, int nc, int nf,
                                                   int dst[2], double wt[2]) {
    const int m = (ic <= nc / 2) ? ic : ic - nc;
    const int am = (m < 0) ? -m : m;
    if (am == nc / 2) {
        dst[0] = nc / 2;
        dst[1] = nf - nc / 2;
        wt[0] = 0.5;
        wt[1] = 0.5;
        return 2;
    }
    dst[0] = (m >= 0) ? m : nf + m;
    wt[0] = 1.0;
    return 1;
}

/// Embed coarse spectrum into fine (scaled by scale_prolong). One thread /
/// coarse mode; atomics for Nyquist multi-dest.
__global__ void embed_kernel(const cufftDoubleComplex* __restrict__ Fc,
                             cufftDoubleComplex* __restrict__ Ff, int nc0,
                             int nc1, int nc2, int nf0, int nf1, int nf2,
                             double scale) {
    const long ntot_c = static_cast<long>(nc0) * static_cast<long>(nc1)
                        * static_cast<long>(nc2);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < ntot_c; idx += static_cast<long>(blockDim.x) * gridDim.x) {
        const int i2 = static_cast<int>(idx % nc2);
        long t = idx / nc2;
        const int i1 = static_cast<int>(t % nc1);
        const int i0 = static_cast<int>(t / nc1);

        int d0[2], d1[2], d2[2];
        double w0[2], w1[2], w2[2];
        const int c0 = axis_coarse_to_fine(i0, nc0, nf0, d0, w0);
        const int c1 = axis_coarse_to_fine(i1, nc1, nf1, d1, w1);
        const int c2 = axis_coarse_to_fine(i2, nc2, nf2, d2, w2);

        const double vre = scale * Fc[idx].x;
        const double vim = scale * Fc[idx].y;
        for (int a = 0; a < c0; ++a)
            for (int b = 0; b < c1; ++b)
                for (int c = 0; c < c2; ++c) {
                    const double w = w0[a] * w1[b] * w2[c];
                    const long fidx =
                        (static_cast<long>(d0[a]) * nf1 + d1[b]) * nf2 + d2[c];
                    atomicAdd(&Ff[fidx].x, w * vre);
                    atomicAdd(&Ff[fidx].y, w * vim);
                }
    }
}

/// Gather fine spectrum into coarse (adjoint of embed without prolong scale).
__global__ void gather_kernel(const cufftDoubleComplex* __restrict__ Ff,
                              cufftDoubleComplex* __restrict__ Fc, int nc0,
                              int nc1, int nc2, int nf0, int nf1, int nf2) {
    const long ntot_c = static_cast<long>(nc0) * static_cast<long>(nc1)
                        * static_cast<long>(nc2);
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < ntot_c; idx += static_cast<long>(blockDim.x) * gridDim.x) {
        const int i2 = static_cast<int>(idx % nc2);
        long t = idx / nc2;
        const int i1 = static_cast<int>(t % nc1);
        const int i0 = static_cast<int>(t / nc1);

        int d0[2], d1[2], d2[2];
        double w0[2], w1[2], w2[2];
        const int c0 = axis_coarse_to_fine(i0, nc0, nf0, d0, w0);
        const int c1 = axis_coarse_to_fine(i1, nc1, nf1, d1, w1);
        const int c2 = axis_coarse_to_fine(i2, nc2, nf2, d2, w2);

        double re = 0.0, im = 0.0;
        for (int a = 0; a < c0; ++a)
            for (int b = 0; b < c1; ++b)
                for (int c = 0; c < c2; ++c) {
                    const double w = w0[a] * w1[b] * w2[c];
                    const long fidx =
                        (static_cast<long>(d0[a]) * nf1 + d1[b]) * nf2 + d2[c];
                    re += w * Ff[fidx].x;
                    im += w * Ff[fidx].y;
                }
        Fc[idx].x = re;
        Fc[idx].y = im;
    }
}

__global__ void axpy_add_kernel(const double* __restrict__ x,
                                double* __restrict__ y, long n) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<long>(blockDim.x) * gridDim.x)
        y[i] += x[i];
}

__global__ void copy_field_kernel(const double* __restrict__ x,
                                  double* __restrict__ y, long n) {
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<long>(blockDim.x) * gridDim.x)
        y[i] = x[i];
}

} // namespace

MultigridTransferDevice::MultigridTransferDevice(
    const CollocateGridDesc& coarse, const CollocateGridDesc& fine,
    cudaStream_t stream)
    : nc0_(coarse.n[0]), nc1_(coarse.n[1]), nc2_(coarse.n[2]), nf0_(fine.n[0]),
      nf1_(fine.n[1]), nf2_(fine.n[2]), ntot_c_(coarse.npts()),
      ntot_f_(fine.npts()), stream_(stream) {
    if (ntot_c_ <= 0 || ntot_f_ <= 0)
        throw std::invalid_argument(
            "MultigridTransferDevice: empty coarse or fine grid");
    for (int d = 0; d < 3; ++d) {
        if (coarse.n[d] <= 0 || fine.n[d] <= 0)
            throw std::invalid_argument(
                "MultigridTransferDevice: non-positive n");
        if (coarse.n[d] % 2 != 0 || fine.n[d] % 2 != 0)
            throw std::invalid_argument(
                "MultigridTransferDevice: n must be even (FFT Nyquist)");
        if (fine.n[d] < coarse.n[d])
            throw std::invalid_argument(
                "MultigridTransferDevice: fine n < coarse n on an axis");
        if (coarse.L[d] <= 0.0 || fine.L[d] <= 0.0)
            throw std::invalid_argument(
                "MultigridTransferDevice: non-positive L");
        // Same physical box required for mode-index embedding.
        if (std::abs(coarse.L[d] - fine.L[d]) > 1e-12 * fine.L[d])
            throw std::invalid_argument(
                "MultigridTransferDevice: coarse/fine L mismatch");
    }

    scale_prolong_ =
        static_cast<double>(ntot_f_) / static_cast<double>(ntot_c_);
    inv_nf_ = 1.0 / static_cast<double>(ntot_f_);

    try {
        check_cufft(cufftPlan3d(&fwd_c_, nc0_, nc1_, nc2_, CUFFT_Z2Z),
                    "cufftPlan3d coarse Z2Z fwd");
        check_cufft(cufftPlan3d(&bwd_c_, nc0_, nc1_, nc2_, CUFFT_Z2Z),
                    "cufftPlan3d coarse Z2Z bwd");
        check_cufft(cufftPlan3d(&fwd_f_, nf0_, nf1_, nf2_, CUFFT_Z2Z),
                    "cufftPlan3d fine Z2Z fwd");
        check_cufft(cufftPlan3d(&bwd_f_, nf0_, nf1_, nf2_, CUFFT_Z2Z),
                    "cufftPlan3d fine Z2Z bwd");
        if (stream_) {
            check_cufft(cufftSetStream(fwd_c_, stream_),
                        "cufftSetStream fwd_c");
            check_cufft(cufftSetStream(bwd_c_, stream_),
                        "cufftSetStream bwd_c");
            check_cufft(cufftSetStream(fwd_f_, stream_),
                        "cufftSetStream fwd_f");
            check_cufft(cufftSetStream(bwd_f_, stream_),
                        "cufftSetStream bwd_f");
        }
        check_cuda(cudaMalloc(&d_c_c_, static_cast<size_t>(ntot_c_)
                                           * sizeof(cufftDoubleComplex)),
                   "malloc coarse cplx");
        check_cuda(cudaMalloc(&d_c_f_, static_cast<size_t>(ntot_f_)
                                           * sizeof(cufftDoubleComplex)),
                   "malloc fine cplx");
    } catch (...) {
        if (fwd_c_)
            cufftDestroy(fwd_c_);
        if (bwd_c_)
            cufftDestroy(bwd_c_);
        if (fwd_f_)
            cufftDestroy(fwd_f_);
        if (bwd_f_)
            cufftDestroy(bwd_f_);
        cudaFree(d_c_c_);
        cudaFree(d_c_f_);
        throw;
    }
}

MultigridTransferDevice::~MultigridTransferDevice() {
    if (fwd_c_)
        cufftDestroy(fwd_c_);
    if (bwd_c_)
        cufftDestroy(bwd_c_);
    if (fwd_f_)
        cufftDestroy(fwd_f_);
    if (bwd_f_)
        cufftDestroy(bwd_f_);
    cudaFree(d_c_c_);
    cudaFree(d_c_f_);
}

void MultigridTransferDevice::prolong(const double* d_coarse, double* d_fine) {
    if (d_coarse == nullptr || d_fine == nullptr)
        throw std::invalid_argument("MultigridTransferDevice::prolong: null");
    const int bc = grid_blocks(ntot_c_);
    const int bf = grid_blocks(ntot_f_);

    pack_real_kernel<<<bc, kBlock, 0, stream_>>>(d_coarse, d_c_c_, ntot_c_);
    check_cufft(cufftExecZ2Z(fwd_c_, d_c_c_, d_c_c_, CUFFT_FORWARD),
                "fwd coarse prolong");

    zero_cplx_kernel<<<bf, kBlock, 0, stream_>>>(d_c_f_, ntot_f_);
    embed_kernel<<<bc, kBlock, 0, stream_>>>(d_c_c_, d_c_f_, nc0_, nc1_, nc2_,
                                             nf0_, nf1_, nf2_, scale_prolong_);
    check_cuda(cudaGetLastError(), "embed_kernel");

    check_cufft(cufftExecZ2Z(bwd_f_, d_c_f_, d_c_f_, CUFFT_INVERSE),
                "bwd fine prolong");
    unpack_scale_kernel<<<bf, kBlock, 0, stream_>>>(d_c_f_, d_fine, ntot_f_,
                                                    inv_nf_);
}

void MultigridTransferDevice::restrict(const double* d_fine, double* d_coarse) {
    if (d_coarse == nullptr || d_fine == nullptr)
        throw std::invalid_argument("MultigridTransferDevice::restrict: null");
    const int bc = grid_blocks(ntot_c_);
    const int bf = grid_blocks(ntot_f_);

    pack_real_kernel<<<bf, kBlock, 0, stream_>>>(d_fine, d_c_f_, ntot_f_);
    check_cufft(cufftExecZ2Z(fwd_f_, d_c_f_, d_c_f_, CUFFT_FORWARD),
                "fwd fine restrict");

    gather_kernel<<<bc, kBlock, 0, stream_>>>(d_c_f_, d_c_c_, nc0_, nc1_, nc2_,
                                              nf0_, nf1_, nf2_);
    check_cuda(cudaGetLastError(), "gather_kernel");

    check_cufft(cufftExecZ2Z(bwd_c_, d_c_c_, d_c_c_, CUFFT_INVERSE),
                "bwd coarse restrict");
    // Adjoint normalization is 1/Nf (CPU restrict_fft), not 1/Nc.
    unpack_scale_kernel<<<bc, kBlock, 0, stream_>>>(d_c_c_, d_coarse, ntot_c_,
                                                    inv_nf_);
}

void device_axpy_add(const double* d_x, double* d_y, long n,
                     cudaStream_t stream) {
    if (n <= 0 || d_x == nullptr || d_y == nullptr)
        return;
    const int b = grid_blocks(n);
    axpy_add_kernel<<<b, kBlock, 0, stream>>>(d_x, d_y, n);
}

void device_copy_field(const double* d_x, double* d_y, long n,
                       cudaStream_t stream) {
    if (n <= 0 || d_x == nullptr || d_y == nullptr)
        return;
    const int b = grid_blocks(n);
    copy_field_kernel<<<b, kBlock, 0, stream>>>(d_x, d_y, n);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
