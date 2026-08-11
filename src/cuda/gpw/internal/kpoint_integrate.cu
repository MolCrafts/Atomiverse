/// @file cuda/gpw/internal/kpoint_integrate.cu
/// @brief Device multi-k potential integrate: V_μν^k = Σ_g w V_g conj(χ_μ) χ_ν.
///
/// Matches host `integrate_potential_k` scientifically. One CUDA block per
/// matrix element (u,t); threads grid-stride over grid points and block-reduce
/// (deterministic within the block tree — not bit-identical to host chunked
/// fold; dual-gate energy tol covers residual).
#include "atomiverse/cuda/gpw/internal/kpoint_integrate.cuh"

#include <cuComplex.h>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("KpointIntegrateDevice: ") + what
                                 + ": " + cudaGetErrorString(e));
}

/// @brief One block per (u,t); threads accumulate over g with shared reduce.
__global__ void integrate_potential_k_kernel(
    const double* __restrict__ V, const cuDoubleComplex* __restrict__ chi,
    int nb, long npts, double w, cuDoubleComplex* __restrict__ Vmat) {
    const int ut = static_cast<int>(blockIdx.x);
    const int nn = nb * nb;
    if (ut >= nn)
        return;
    const int u = ut / nb;
    const int t = ut % nb;

    double re = 0.0;
    double im = 0.0;
    for (long g = static_cast<long>(threadIdx.x); g < npts;
         g += static_cast<long>(blockDim.x)) {
        const double vg = w * V[g];
        if (vg == 0.0)
            continue;
        const cuDoubleComplex cu = chi[g * nb + u];
        const cuDoubleComplex ct = chi[g * nb + t];
        // conj(cu) * ct: (a-bi)(c+di) = (ac+bd) + i(ad-bc)
        const double cre = cu.x * ct.x + cu.y * ct.y;
        const double cim = cu.x * ct.y - cu.y * ct.x;
        re += vg * cre;
        im += vg * cim;
    }

    __shared__ double sre[256];
    __shared__ double sim[256];
    sre[threadIdx.x] = re;
    sim[threadIdx.x] = im;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
         stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            sre[threadIdx.x] += sre[threadIdx.x + stride];
            sim[threadIdx.x] += sim[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        Vmat[ut] = make_cuDoubleComplex(sre[0], sim[0]);
}

constexpr int kBlock = 256;

} // namespace

void KpointIntegrateDevice::build(cudaStream_t stream, int nb, long npts,
                                  int nk) {
    destroy();
    if (nb < 1)
        throw std::invalid_argument(
            "KpointIntegrateDevice::build: nb must be >= 1");
    if (npts < 1)
        throw std::invalid_argument(
            "KpointIntegrateDevice::build: npts must be >= 1");
    if (nk < 1)
        throw std::invalid_argument(
            "KpointIntegrateDevice::build: nk must be >= 1");
    stream_ = stream;
    nb_ = nb;
    npts_ = npts;
    nk_ = nk;
    d_chi_.assign(static_cast<std::size_t>(nk), nullptr);
    const size_t chi_bytes =
        static_cast<size_t>(npts) * static_cast<size_t>(nb)
        * sizeof(cuDoubleComplex);
    for (int ik = 0; ik < nk; ++ik) {
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, chi_bytes), "malloc d_chi");
        d_chi_[static_cast<std::size_t>(ik)] = p;
    }
    check_cuda(cudaMalloc(&d_Vmat_,
                          static_cast<size_t>(nb) * nb * sizeof(cuDoubleComplex)),
               "malloc d_Vmat");
    built_ = true;
}

void KpointIntegrateDevice::destroy() {
    for (void* p : d_chi_) {
        if (p)
            cudaFree(p);
    }
    d_chi_.clear();
    if (d_Vmat_) {
        cudaFree(d_Vmat_);
        d_Vmat_ = nullptr;
    }
    stream_ = nullptr;
    nb_ = 0;
    npts_ = 0;
    nk_ = 0;
    built_ = false;
}

void KpointIntegrateDevice::ensure_built_() const {
    if (!built_)
        throw std::logic_error(
            "KpointIntegrateDevice: used before build()");
}

void KpointIntegrateDevice::upload_chi(int ik,
                                       const std::complex<double>* chi) {
    ensure_built_();
    if (ik < 0 || ik >= nk_)
        throw std::invalid_argument(
            "KpointIntegrateDevice::upload_chi: ik out of range");
    if (chi == nullptr)
        throw std::invalid_argument(
            "KpointIntegrateDevice::upload_chi: null chi");
    static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                  "std::complex<double> layout must match cuDoubleComplex");
    const size_t chi_bytes =
        static_cast<size_t>(npts_) * static_cast<size_t>(nb_)
        * sizeof(cuDoubleComplex);
    check_cuda(cudaMemcpyAsync(d_chi_[static_cast<std::size_t>(ik)], chi,
                               chi_bytes, cudaMemcpyHostToDevice, stream_),
               "H2D chi");
}

void KpointIntegrateDevice::integrate(const double* d_V, double w, int ik) {
    ensure_built_();
    if (ik < 0 || ik >= nk_)
        throw std::invalid_argument(
            "KpointIntegrateDevice::integrate: ik out of range");
    if (d_V == nullptr)
        throw std::invalid_argument(
            "KpointIntegrateDevice::integrate: null d_V");
    const int nn = nb_ * nb_;
    auto* d_chi = static_cast<const cuDoubleComplex*>(
        d_chi_[static_cast<std::size_t>(ik)]);
    auto* d_Vmat = static_cast<cuDoubleComplex*>(d_Vmat_);
    integrate_potential_k_kernel<<<nn, kBlock, 0, stream_>>>(
        d_V, d_chi, nb_, npts_, w, d_Vmat);
    check_cuda(cudaGetLastError(), "integrate_potential_k_kernel");
}

void KpointIntegrateDevice::download_Vmat(
    std::complex<double>* out) const {
    ensure_built_();
    if (out == nullptr)
        throw std::invalid_argument(
            "KpointIntegrateDevice::download_Vmat: null out");
    const size_t bytes =
        static_cast<size_t>(nb_) * nb_ * sizeof(cuDoubleComplex);
    check_cuda(cudaMemcpyAsync(out, d_Vmat_, bytes, cudaMemcpyDeviceToHost,
                               stream_),
               "D2H Vmat");
    check_cuda(cudaStreamSynchronize(stream_), "sync Vmat D2H");
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
