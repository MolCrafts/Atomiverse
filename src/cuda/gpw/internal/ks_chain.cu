/// @file cuda/gpw/internal/ks_chain.cu
/// @brief Composed device KS stage: collocate → Poisson → Padé XC → V_eff → integrate.
///
/// All scientific loops are device kernels. Host only owns the plan/workspace
/// object and D2H of energy scalars at the end of `run`.

#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"

#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

namespace {

constexpr int kBlock = 256;

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::gpw KsChainDevice: ")
                                 + what + " failed ("
                                 + cudaGetErrorString(e) + ")");
}

/// @brief n[i] = rho[i] − n_ion[i]  (n_ion null ⇒ n = rho).
__global__ void n_tot_kernel(const double* __restrict__ rho,
                             const double* __restrict__ n_ion, double* __restrict__ n,
                             long nr) {
    const long i =
        static_cast<long>(blockIdx.x) * blockDim.x + static_cast<long>(threadIdx.x);
    if (i >= nr)
        return;
    const double ion = (n_ion != nullptr) ? n_ion[i] : 0.0;
    n[i] = rho[i] - ion;
}

/// @brief veff = v_es + vxc [+ v_loc]; optional local e_loc = Σ ρ V_loc (unscaled).
__global__ void
assemble_veff_kernel(const double* __restrict__ v_es,
                     const double* __restrict__ vxc,
                     const double* __restrict__ v_loc,
                     const double* __restrict__ rho, double* __restrict__ veff,
                     double* __restrict__ e_loc_acc, long nr) {
    __shared__ double s[kBlock];
    const long i =
        static_cast<long>(blockIdx.x) * blockDim.x + static_cast<long>(threadIdx.x);
    double loc = 0.0;
    if (i < nr) {
        const double vloc = (v_loc != nullptr) ? v_loc[i] : 0.0;
        veff[i] = v_es[i] + vxc[i] + vloc;
        if (v_loc != nullptr)
            loc = rho[i] * vloc;
    }
    if (e_loc_acc != nullptr) {
        s[threadIdx.x] = loc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (static_cast<int>(threadIdx.x) < stride)
                s[threadIdx.x] += s[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            atomicAdd(e_loc_acc, s[0]);
    }
}

__global__ void scale_scalar_kernel(double* x, double factor) { *x *= factor; }

GpwGridDesc to_poisson_grid(const CollocateGridDesc& c) {
    GpwGridDesc g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = c.n[d];
        g.L[d] = c.L[d];
    }
    return g;
}

} // namespace

KsChainDevice::KsChainDevice(const CollocateGridDesc& coll, cudaStream_t stream)
    : coll_grid_(coll)
    , poisson_grid_(to_poisson_grid(coll))
    , ntot_(coll.npts())
    , weight_(coll.weight())
    , stream_(stream)
    , poisson_(poisson_grid_, stream) {
    if (ntot_ <= 0)
        throw std::runtime_error("cuda::gpw KsChainDevice: empty grid");
    const size_t bytes = static_cast<size_t>(ntot_) * sizeof(double);
    check_cuda(cudaMalloc(&d_rho_, bytes), "malloc rho");
    check_cuda(cudaMalloc(&d_n_tot_, bytes), "malloc n_tot");
    check_cuda(cudaMalloc(&d_v_es_, bytes), "malloc v_es");
    check_cuda(cudaMalloc(&d_vxc_, bytes), "malloc vxc");
    check_cuda(cudaMalloc(&d_exc_, bytes), "malloc exc");
    check_cuda(cudaMalloc(&d_veff_, bytes), "malloc veff");
    check_cuda(cudaMalloc(&d_e_xc_, sizeof(double)), "malloc e_xc");
    check_cuda(cudaMalloc(&d_e_loc_, sizeof(double)), "malloc e_loc");
}

KsChainDevice::~KsChainDevice() {
    cudaFree(d_rho_);
    cudaFree(d_n_tot_);
    cudaFree(d_v_es_);
    cudaFree(d_vxc_);
    cudaFree(d_exc_);
    cudaFree(d_veff_);
    cudaFree(d_e_xc_);
    cudaFree(d_e_loc_);
}

KsChainEnergies KsChainDevice::run(const DeviceProduct* d_prods, int nprod,
                                   const double* d_n_ion, const double* d_v_loc,
                                   double* d_acc) {
    if (d_prods == nullptr || nprod <= 0)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run: need ≥1 product");

    const size_t bytes = static_cast<size_t>(ntot_) * sizeof(double);
    check_cuda(cudaMemsetAsync(d_rho_, 0, bytes, stream_), "zero rho");

    // 1. Collocate products → ρ
    collocate_products(coll_grid_, d_prods, nprod, d_rho_, stream_);

    // 2. n = ρ − n_ion
    const int grid_dim =
        static_cast<int>((ntot_ + kBlock - 1) / static_cast<long>(kBlock));
    n_tot_kernel<<<grid_dim, kBlock, 0, stream_>>>(d_rho_, d_n_ion, d_n_tot_,
                                                   ntot_);

    // 3. Poisson(n) → V_es, e_hartree (writes into Poisson internal scratch)
    poisson_.solve(d_n_tot_, d_v_es_);

    // 4. Padé LDA on total ρ → V_xc, ε_xc; E_xc
    const int nr = static_cast<int>(ntot_);
    xc_lda_grid(d_rho_, d_vxc_, d_exc_, nr, stream_);
    xc_energy(d_rho_, d_exc_, nr, weight_, d_e_xc_, stream_);

    // 5. Assemble V_eff; optional ∫ρ V_loc
    if (d_v_loc != nullptr)
        check_cuda(cudaMemsetAsync(d_e_loc_, 0, sizeof(double), stream_),
                   "zero e_loc");
    assemble_veff_kernel<<<grid_dim, kBlock, 0, stream_>>>(
        d_v_es_, d_vxc_, d_v_loc, d_rho_, d_veff_,
        (d_v_loc != nullptr) ? d_e_loc_ : nullptr, ntot_);
    if (d_v_loc != nullptr)
        scale_scalar_kernel<<<1, 1, 0, stream_>>>(d_e_loc_, weight_);

    // 6. Optional integrate V_eff against product shapes
    if (d_acc != nullptr)
        integrate_products(coll_grid_, d_prods, nprod, d_veff_, d_acc, stream_);

    // Host energy scalars (one sync for the three doubles).
    KsChainEnergies out;
    check_cuda(cudaMemcpyAsync(&out.e_hartree, poisson_.last_e_hartree_device(),
                               sizeof(double), cudaMemcpyDeviceToHost, stream_),
               "D2H e_hartree");
    check_cuda(cudaMemcpyAsync(&out.e_xc, d_e_xc_, sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H e_xc");
    if (d_v_loc != nullptr) {
        check_cuda(cudaMemcpyAsync(&out.e_locgrid, d_e_loc_, sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H e_loc");
    } else {
        out.e_locgrid = 0.0;
    }
    check_cuda(cudaStreamSynchronize(stream_), "sync energies");
    return out;
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
