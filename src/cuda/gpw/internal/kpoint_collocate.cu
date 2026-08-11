/// @file cuda/gpw/internal/kpoint_collocate.cu
/// @brief Device multi-k density collocate kernels (χ^k · P_k · χ^{k†}).
///
/// Twin of `cpu::gpw::internal::collocate_density_k`. All scientific loops
/// over grid points run on device — no host loop over g in the hot path.
#include "atomiverse/cuda/gpw/internal/kpoint_collocate.cuh"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {
namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("KpointCollocateDevice: ") + what
                                 + ": " + cudaGetErrorString(e));
}

/// @brief ρ(g) += wk · Re[ Σ_u v_u · (P · conj(v))_u ] with v = χ^k(g).
///
/// One thread per grid point. P is row-major nb×nb (matches host
/// `CplxMatrix` / xtensor). chi_k points at the ik slice of the packed χ.
__global__ void collocate_density_k_kernel(const double2* __restrict__ chi_k,
                                           const double2* __restrict__ P,
                                           int nb, long npts, double wk,
                                           double* __restrict__ rho) {
    const long g = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
    if (g >= npts)
        return;

    const double2* v = chi_k + g * static_cast<long>(nb);
    double acc = 0.0;
    for (int u = 0; u < nb; ++u) {
        const double2 vu = v[u];
        // s = (P · conj(v))_u
        double sr = 0.0;
        double si = 0.0;
        const double2* Pu = P + static_cast<long>(u) * nb;
        for (int w = 0; w < nb; ++w) {
            const double2 Puw = Pu[w];
            const double2 vw = v[w];
            // Puw * conj(vw) = (pr+i pi)(wr - i wi) = pr wr + pi wi + i(pi wr - pr wi)
            const double wr = vw.x;
            const double wi = vw.y;
            sr += Puw.x * wr + Puw.y * wi;
            si += Puw.y * wr - Puw.x * wi;
        }
        // Re(vu * s) = vu.x * sr - vu.y * si
        acc += vu.x * sr - vu.y * si;
    }
    rho[g] += wk * acc;
}

} // namespace

void KpointCollocateDevice::build(cudaStream_t stream, int nk, int nb,
                                  long npts) {
    destroy();
    if (nk < 1 || nb < 1 || npts < 1)
        throw std::invalid_argument(
            "KpointCollocateDevice::build: nk, nb, npts must be >= 1");
    stream_ = stream;
    nk_ = nk;
    nb_ = nb;
    npts_ = npts;
    const size_t nchi = static_cast<size_t>(nk) * static_cast<size_t>(npts)
                        * static_cast<size_t>(nb);
    const size_t np = static_cast<size_t>(nb) * static_cast<size_t>(nb);
    check_cuda(cudaMalloc(&d_chi_, nchi * sizeof(double2)), "malloc d_chi");
    check_cuda(cudaMalloc(&d_P_, np * sizeof(double2)), "malloc d_P");
    built_ = true;
}

void KpointCollocateDevice::upload_chi(int ik,
                                       const std::complex<double>* chi_host) {
    if (!built_)
        throw std::runtime_error(
            "KpointCollocateDevice::upload_chi: not built");
    if (ik < 0 || ik >= nk_)
        throw std::invalid_argument(
            "KpointCollocateDevice::upload_chi: ik out of range");
    if (!chi_host)
        throw std::invalid_argument(
            "KpointCollocateDevice::upload_chi: null chi_host");
    const size_t n = static_cast<size_t>(npts_) * static_cast<size_t>(nb_);
    const size_t off = static_cast<size_t>(ik) * n;
    // std::complex<double> is layout-compatible with double2 {re, im}.
    static_assert(sizeof(std::complex<double>) == sizeof(double2),
                  "std::complex<double> must match double2");
    check_cuda(cudaMemcpyAsync(d_chi_ + off, chi_host, n * sizeof(double2),
                               cudaMemcpyHostToDevice, stream_),
               "H2D chi");
}

void KpointCollocateDevice::zero_rho(double* d_rho) const {
    if (!built_)
        throw std::runtime_error("KpointCollocateDevice::zero_rho: not built");
    if (!d_rho)
        throw std::invalid_argument(
            "KpointCollocateDevice::zero_rho: null d_rho");
    check_cuda(cudaMemsetAsync(d_rho, 0,
                               static_cast<size_t>(npts_) * sizeof(double),
                               stream_),
               "memset d_rho");
}

void KpointCollocateDevice::collocate_k(int ik,
                                        const std::complex<double>* P_host,
                                        double wk, double* d_rho) {
    if (!built_)
        throw std::runtime_error(
            "KpointCollocateDevice::collocate_k: not built");
    if (ik < 0 || ik >= nk_)
        throw std::invalid_argument(
            "KpointCollocateDevice::collocate_k: ik out of range");
    if (!P_host || !d_rho)
        throw std::invalid_argument(
            "KpointCollocateDevice::collocate_k: null P_host or d_rho");
    const size_t np = static_cast<size_t>(nb_) * static_cast<size_t>(nb_);
    static_assert(sizeof(std::complex<double>) == sizeof(double2),
                  "std::complex<double> must match double2");
    check_cuda(cudaMemcpyAsync(d_P_, P_host, np * sizeof(double2),
                               cudaMemcpyHostToDevice, stream_),
               "H2D P_k");
    const size_t nchi = static_cast<size_t>(npts_) * static_cast<size_t>(nb_);
    const double2* chi_k = d_chi_ + static_cast<size_t>(ik) * nchi;
    constexpr int kBlock = 256;
    const int grid =
        static_cast<int>((npts_ + kBlock - 1) / static_cast<long>(kBlock));
    collocate_density_k_kernel<<<grid, kBlock, 0, stream_>>>(
        chi_k, d_P_, nb_, npts_, wk, d_rho);
    check_cuda(cudaGetLastError(), "collocate_density_k_kernel launch");
}

void KpointCollocateDevice::destroy() {
    if (d_chi_) {
        cudaFree(d_chi_);
        d_chi_ = nullptr;
    }
    if (d_P_) {
        cudaFree(d_P_);
        d_P_ = nullptr;
    }
    stream_ = nullptr;
    nk_ = 0;
    nb_ = 0;
    npts_ = 0;
    built_ = false;
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
