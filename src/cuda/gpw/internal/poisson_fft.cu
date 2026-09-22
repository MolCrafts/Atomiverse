/// @file cuda/gpw/internal/poisson_fft.cu
/// @brief Device Γ-point periodic Poisson (cuFFT r2c / Coulomb / c2r).
///
/// Mirrors `cpu::gpw::solve_poisson` (FFTW r2c last-axis Hermitian). All
/// scientific loops are device kernels — no host loops over grid points.

#include "atomiverse/cuda/gpw/internal/poisson_fft.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kFourPi = 4.0 * kPi;
constexpr int kBlock = 256;

void check_cufft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuda::gpw PoissonFftDevice: ")
                                 + what + " failed (cufftResult "
                                 + std::to_string(static_cast<int>(r)) + ")");
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::gpw PoissonFftDevice: ")
                                 + what + " failed (" + cudaGetErrorString(e)
                                 + ")");
}

/// @brief One thread per half-complex G-mode: V(G) = 4π ρ(G)/|G|², V(0)=0.
/// Layout: idx = (i0 * n1 + i1) * n2c + i2  (i2 = non-negative last-axis freqs).
__global__ void apply_coulomb_kernel(cufftDoubleComplex* __restrict__ cbuf,
                                     int n0, int n1, int n2c, double gx_unit,
                                     double gy_unit, double gz_unit) {
    const long ncplx = static_cast<long>(n0) * n1 * n2c;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < ncplx; idx += static_cast<long>(blockDim.x) * gridDim.x) {
        const int i2 = static_cast<int>(idx % n2c);
        long t = idx / n2c;
        const int i1 = static_cast<int>(t % n1);
        const int i0 = static_cast<int>(t / n1);

        const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
        const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
        const int m2 = i2; // r2c keeps only non-negative last-dim freqs
        const double gx = m0 * gx_unit;
        const double gy = m1 * gy_unit;
        const double gz = m2 * gz_unit;
        const double g2 = gx * gx + gy * gy + gz * gz;
        if (g2 == 0.0) {
            cbuf[idx].x = 0.0;
            cbuf[idx].y = 0.0;
        } else {
            const double f = kFourPi / g2;
            cbuf[idx].x *= f;
            cbuf[idx].y *= f;
        }
    }
}

/// @brief Normalize c2r output by 1/N and accumulate e_H = 0.5 w Σ ρ V.
/// Uses a per-block shared reduction then one atomicAdd into d_e.
__global__ void normalize_and_energy_kernel(const double* __restrict__ rho,
                                            double* __restrict__ v, long ntot,
                                            double inv_n, double half_w,
                                            double* __restrict__ d_e) {
    __shared__ double smem[kBlock];
    const int tid = threadIdx.x;
    double local = 0.0;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + tid; i < ntot;
         i += static_cast<long>(blockDim.x) * gridDim.x) {
        const double vi = v[i] * inv_n;
        v[i] = vi;
        local += rho[i] * vi;
    }
    smem[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_e, half_w * smem[0]);
}

/// @brief Cap grid-stride blocks so tiny grids still get a full launch.
int grid_blocks(long nwork) {
    if (nwork <= 0)
        return 0;
    const long need = (nwork + kBlock - 1) / kBlock;
    // Cap keeps occupancy high without launching millions of blocks on large grids.
    constexpr int kMaxBlocks = 2048;
    return static_cast<int>(need < kMaxBlocks ? need : kMaxBlocks);
}

} // namespace

PoissonFftDevice::PoissonFftDevice(const GpwGridDesc& grid, cudaStream_t stream)
    : n0_(grid.n[0]), n1_(grid.n[1]), n2_(grid.n[2]), n2c_(grid.n2c()),
      ntot_(grid.npts()), ncplx_(grid.ncplx()), stream_(stream) {
    if (n0_ <= 0 || n1_ <= 0 || n2_ <= 0)
        throw std::invalid_argument(
            "PoissonFftDevice: non-positive grid shape");
    if (grid.L[0] <= 0.0 || grid.L[1] <= 0.0 || grid.L[2] <= 0.0)
        throw std::invalid_argument(
            "PoissonFftDevice: non-positive cell lengths");

    L_[0] = grid.L[0];
    L_[1] = grid.L[1];
    L_[2] = grid.L[2];
    weight_ = grid.weight();
    inv_n_ = 1.0 / static_cast<double>(ntot_);
    gx_unit_ = kTwoPi / L_[0];
    gy_unit_ = kTwoPi / L_[1];
    gz_unit_ = kTwoPi / L_[2];

    // cufftPlan3d(nx,ny,nz): nz is contiguous / Hermitian-halved for D2Z —
    // matches FFTW fftw_plan_dft_r2c_3d(n0,n1,n2) used by cpu::gpw::R2cFft.
    try {
        check_cufft(cufftPlan3d(&r2c_, n0_, n1_, n2_, CUFFT_D2Z),
                    "cufftPlan3d D2Z");
        check_cufft(cufftPlan3d(&c2r_, n0_, n1_, n2_, CUFFT_Z2D),
                    "cufftPlan3d Z2D");
        check_cufft(cufftSetStream(r2c_, stream_), "cufftSetStream r2c");
        check_cufft(cufftSetStream(c2r_, stream_), "cufftSetStream c2r");
        check_cuda(cudaMalloc(&d_cbuf_, static_cast<size_t>(ncplx_)
                                            * sizeof(cufftDoubleComplex)),
                   "cudaMalloc cbuf");
        check_cuda(cudaMalloc(&d_c_scratch_, static_cast<size_t>(ncplx_)
                                                 * sizeof(cufftDoubleComplex)),
                   "cudaMalloc c_scratch");
        check_cuda(cudaMalloc(&d_e_scratch_, sizeof(double)),
                   "cudaMalloc e_scratch");
    } catch (...) {
        if (r2c_)
            cufftDestroy(r2c_);
        if (c2r_)
            cufftDestroy(c2r_);
        if (d_cbuf_)
            cudaFree(d_cbuf_);
        if (d_c_scratch_)
            cudaFree(d_c_scratch_);
        if (d_e_scratch_)
            cudaFree(d_e_scratch_);
        r2c_ = c2r_ = 0;
        d_cbuf_ = d_c_scratch_ = nullptr;
        d_e_scratch_ = nullptr;
        throw;
    }
}

PoissonFftDevice::~PoissonFftDevice() {
    if (r2c_)
        cufftDestroy(r2c_);
    if (c2r_)
        cufftDestroy(c2r_);
    if (d_cbuf_)
        cudaFree(d_cbuf_);
    if (d_c_scratch_)
        cudaFree(d_c_scratch_);
    if (d_e_scratch_)
        cudaFree(d_e_scratch_);
}

void PoissonFftDevice::solve(const double* d_rho, double* d_v,
                             double* d_e_hartree) {
    if (!d_rho || !d_v)
        throw std::invalid_argument("PoissonFftDevice::solve: null rho/v");
    if (d_rho == d_v)
        throw std::invalid_argument(
            "PoissonFftDevice::solve: d_rho must not alias d_v");

    // r2c: ρ(r) → ρ(G) into owned half-complex buffer.
    check_cufft(cufftExecD2Z(r2c_, const_cast<double*>(d_rho), d_cbuf_),
                "cufftExecD2Z");

    const int gblocks = grid_blocks(ncplx_);
    apply_coulomb_kernel<<<gblocks, kBlock, 0, stream_>>>(
        d_cbuf_, n0_, n1_, n2c_, gx_unit_, gy_unit_, gz_unit_);
    check_cuda(cudaGetLastError(), "apply_coulomb_kernel launch");

    // cuFFT Z2D destroys its input; copy so re-solves keep a clean path.
    check_cuda(cudaMemcpyAsync(d_c_scratch_, d_cbuf_,
                               static_cast<size_t>(ncplx_)
                                   * sizeof(cufftDoubleComplex),
                               cudaMemcpyDeviceToDevice, stream_),
               "cudaMemcpyAsync c2r preserve");
    check_cufft(cufftExecZ2D(c2r_, d_c_scratch_, d_v), "cufftExecZ2D");

    // Zero energy accumulator then normalize + reduce.
    check_cuda(cudaMemsetAsync(d_e_scratch_, 0, sizeof(double), stream_),
               "cudaMemsetAsync e");
    const int rblocks = grid_blocks(ntot_);
    const double half_w = 0.5 * weight_;
    normalize_and_energy_kernel<<<rblocks, kBlock, 0, stream_>>>(
        d_rho, d_v, ntot_, inv_n_, half_w, d_e_scratch_);
    check_cuda(cudaGetLastError(), "normalize_and_energy_kernel launch");

    if (d_e_hartree && d_e_hartree != d_e_scratch_) {
        check_cuda(cudaMemcpyAsync(d_e_hartree, d_e_scratch_, sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpyAsync e_out");
    }
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
