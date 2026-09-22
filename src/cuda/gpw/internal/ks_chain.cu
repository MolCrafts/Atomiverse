/// @file cuda/gpw/internal/ks_chain.cu
/// @brief Composed device KS stage: collocate → Poisson → XC → V_eff → integrate.
///
/// All scientific loops are device kernels. Host only owns the plan/workspace
/// object and D2H of energy scalars at the end of `run`. Closed-shell PBE-GGA
/// reuses `cuda::pw::internal::xc_pbe_grid` and `SpectralFftDevice` (CPU
/// `gradient` / `divergence` conventions). Multigrid path reuses
/// `MultigridTransferDevice` (CPU `prolong_fft` / `restrict_fft`).

#include "atomiverse/cuda/gpw/internal/ks_chain.cuh"

#if defined(ATV_HAS_PW)
#include "atomiverse/cuda/planewave/internal/xc_pbe.cuh"
#endif

#include <algorithm>
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
        throw std::runtime_error(std::string("cuda::gpw KsChainDevice: ") + what
                                 + " failed (" + cudaGetErrorString(e) + ")");
}

/// @brief n[i] = rho[i] − n_ion[i]  (n_ion null ⇒ n = rho).
__global__ void n_tot_kernel(const double* __restrict__ rho,
                             const double* __restrict__ n_ion,
                             double* __restrict__ n, long nr) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
    if (i >= nr)
        return;
    const double ion = (n_ion != nullptr) ? n_ion[i] : 0.0;
    n[i] = rho[i] - ion;
}

/// @brief veff = v_es + vxc [+ v_loc]; optional local e_loc = Σ ρ V_loc (unscaled).
__global__ void assemble_veff_kernel(const double* __restrict__ v_es,
                                     const double* __restrict__ vxc,
                                     const double* __restrict__ v_loc,
                                     const double* __restrict__ rho,
                                     double* __restrict__ veff,
                                     double* __restrict__ e_loc_acc, long nr) {
    __shared__ double s[kBlock];
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
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

/// @brief Dual V_eff^σ = V_es + V_xc^σ [+ V_loc]; e_loc from total ρ = ρ_α+ρ_β.
__global__ void assemble_veff_uks_kernel(
    const double* __restrict__ v_es, const double* __restrict__ vxc_a,
    const double* __restrict__ vxc_b, const double* __restrict__ v_loc,
    const double* __restrict__ rho_a, const double* __restrict__ rho_b,
    double* __restrict__ veff_a, double* __restrict__ veff_b,
    double* __restrict__ e_loc_acc, long nr) {
    __shared__ double s[kBlock];
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
    double loc = 0.0;
    if (i < nr) {
        const double vloc = (v_loc != nullptr) ? v_loc[i] : 0.0;
        const double ves = v_es[i] + vloc;
        veff_a[i] = ves + vxc_a[i];
        veff_b[i] = ves + vxc_b[i];
        if (v_loc != nullptr)
            loc = (rho_a[i] + rho_b[i]) * vloc;
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

/// @brief out[i] = a[i] + b[i]
__global__ void add_fields_kernel(const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double* __restrict__ out, long nr) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
    if (i < nr)
        out[i] = a[i] + b[i];
}

/// @brief vxc[i] -= div[i]  (GGA: local v_ρ minus spectral ∇·flux).
__global__ void vxc_sub_div_kernel(double* __restrict__ vxc,
                                   const double* __restrict__ div, long nr) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x
                   + static_cast<long>(threadIdx.x);
    if (i < nr)
        vxc[i] -= div[i];
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
    : coll_grid_(coll), poisson_grid_(to_poisson_grid(coll)),
      ntot_(coll.npts()), weight_(coll.weight()), stream_(stream),
      poisson_(poisson_grid_, stream), spectral_(poisson_grid_, stream) {
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
    check_cuda(cudaMalloc(&d_gx_, bytes), "malloc gx");
    check_cuda(cudaMalloc(&d_gy_, bytes), "malloc gy");
    check_cuda(cudaMalloc(&d_gz_, bytes), "malloc gz");
    check_cuda(cudaMalloc(&d_div_, bytes), "malloc div");
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
    cudaFree(d_gx_);
    cudaFree(d_gy_);
    cudaFree(d_gz_);
    cudaFree(d_div_);
    cudaFree(d_rho_a_);
    cudaFree(d_rho_b_);
    cudaFree(d_vxc_b_);
    cudaFree(d_veff_b_);
}

void KsChainDevice::ensure_uks_buffers_() {
    if (d_rho_a_ != nullptr)
        return;
    const size_t bytes = static_cast<size_t>(ntot_) * sizeof(double);
    check_cuda(cudaMalloc(&d_rho_a_, bytes), "malloc rho_a");
    check_cuda(cudaMalloc(&d_rho_b_, bytes), "malloc rho_b");
    check_cuda(cudaMalloc(&d_vxc_b_, bytes), "malloc vxc_b");
    check_cuda(cudaMalloc(&d_veff_b_, bytes), "malloc veff_b");
}

KsChainEnergies
KsChainDevice::run_from_rho(const double* d_rho_in, const double* d_n_ion,
                            const double* d_v_loc, const DeviceProduct* d_prods,
                            int nprod, double* d_acc, KsXcKind xc) {
    if (d_rho_in == nullptr)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run_from_rho: null density");

    const size_t bytes = static_cast<size_t>(ntot_) * sizeof(double);
    device_copy_field(d_rho_in, d_rho_, ntot_, stream_);

    // 2. n = ρ − n_ion
    const int grid_dim =
        static_cast<int>((ntot_ + kBlock - 1) / static_cast<long>(kBlock));
    n_tot_kernel<<<grid_dim, kBlock, 0, stream_>>>(d_rho_, d_n_ion, d_n_tot_,
                                                   ntot_);

    // 3. Poisson(n) → V_es, e_hartree
    poisson_.solve(d_n_tot_, d_v_es_);

    // 4. XC → V_xc, ε_xc; E_xc
    const int nr = static_cast<int>(ntot_);
    if (xc == KsXcKind::PADE) {
        xc_lda_grid(d_rho_, d_vxc_, d_exc_, nr, stream_);
        xc_energy(d_rho_, d_exc_, nr, weight_, d_e_xc_, stream_);
    } else if (xc == KsXcKind::PBE) {
#if defined(ATV_HAS_PW)
        spectral_.gradient(d_rho_, d_gx_, d_gy_, d_gz_);
        ::ATV::cuda::pw::internal::xc_pbe_grid(d_rho_, d_gx_, d_gy_, d_gz_,
                                               d_vxc_, d_exc_, nr, stream_);
        xc_energy(d_rho_, d_exc_, nr, weight_, d_e_xc_, stream_);
        spectral_.divergence(d_gx_, d_gy_, d_gz_, d_div_);
        vxc_sub_div_kernel<<<grid_dim, kBlock, 0, stream_>>>(d_vxc_, d_div_,
                                                             ntot_);
        check_cuda(cudaGetLastError(), "vxc_sub_div_kernel");
#else
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run_from_rho: KsXcKind::PBE requires "
            "ATV_BUILD_PW (shared cuda::pw::internal::xc_pbe_grid)");
#endif
    } else {
        throw std::invalid_argument(
            "cuda::gpw KsChainDevice::run_from_rho: unsupported KsXcKind");
    }

    // 5. Assemble V_eff; optional ∫ρ V_loc
    if (d_v_loc != nullptr)
        check_cuda(cudaMemsetAsync(d_e_loc_, 0, sizeof(double), stream_),
                   "zero e_loc");
    assemble_veff_kernel<<<grid_dim, kBlock, 0, stream_>>>(
        d_v_es_, d_vxc_, d_v_loc, d_rho_, d_veff_,
        (d_v_loc != nullptr) ? d_e_loc_ : nullptr, ntot_);
    if (d_v_loc != nullptr)
        scale_scalar_kernel<<<1, 1, 0, stream_>>>(d_e_loc_, weight_);

    // 6. Optional integrate on this grid
    if (d_acc != nullptr) {
        if (d_prods == nullptr || nprod <= 0)
            throw std::runtime_error(
                "cuda::gpw KsChainDevice::run_from_rho: d_acc needs products");
        integrate_products(coll_grid_, d_prods, nprod, d_veff_, d_acc, stream_);
    }

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

KsChainEnergies KsChainDevice::run(const DeviceProduct* d_prods, int nprod,
                                   const double* d_n_ion, const double* d_v_loc,
                                   double* d_acc, KsXcKind xc) {
    if (d_prods == nullptr || nprod <= 0)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run: need ≥1 product");

    const size_t bytes = static_cast<size_t>(ntot_) * sizeof(double);
    check_cuda(cudaMemsetAsync(d_rho_, 0, bytes, stream_), "zero rho");

    // 1. Collocate products → ρ
    collocate_products(coll_grid_, d_prods, nprod, d_rho_, stream_);

    // Reuse run_from_rho body from the density we just built (copy is identity
    // source = d_rho_; use internal path via temporary self-copy avoided —
    // inline steps by calling run_from_rho with d_rho_ as input is OK: copy
    // field to itself is safe as element-wise y=x).
    return run_from_rho(d_rho_, d_n_ion, d_v_loc, d_prods, nprod, d_acc, xc);
}

KsChainEnergies KsChainDevice::run_uks(const DeviceProduct* d_prods_a,
                                       int nprod_a,
                                       const DeviceProduct* d_prods_b,
                                       int nprod_b, const double* d_n_ion,
                                       const double* d_v_loc, double* d_acc_a,
                                       double* d_acc_b, KsXcKind xc) {
    if (nprod_a < 0 || nprod_b < 0)
        throw std::invalid_argument(
            "cuda::gpw KsChainDevice::run_uks: nprod_a/b must be >= 0");
    if (nprod_a > 0 && d_prods_a == nullptr)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run_uks: null α products with nprod_a>0");
    if (nprod_b > 0 && d_prods_b == nullptr)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run_uks: null β products with nprod_b>0");
    if (nprod_a == 0 && nprod_b == 0)
        throw std::runtime_error(
            "cuda::gpw KsChainDevice::run_uks: need ≥1 product in α or β");
    if (xc != KsXcKind::PADE)
        throw std::invalid_argument(
            "cuda::gpw KsChainDevice::run_uks: first UKS cut is Padé LSD only "
            "(KsXcKind::PADE). PBE-UKS is a deeper product slug.");

    ensure_uks_buffers_();
    const size_t bytes_rho = static_cast<size_t>(ntot_) * sizeof(double);
    const int grid_dim =
        static_cast<int>((ntot_ + kBlock - 1) / static_cast<long>(kBlock));
    const int nr = static_cast<int>(ntot_);

    // 1. Collocate dual spin densities (or zero).
    check_cuda(cudaMemsetAsync(d_rho_a_, 0, bytes_rho, stream_), "zero rho_a");
    check_cuda(cudaMemsetAsync(d_rho_b_, 0, bytes_rho, stream_), "zero rho_b");
    if (nprod_a > 0)
        collocate_products(coll_grid_, d_prods_a, nprod_a, d_rho_a_, stream_);
    if (nprod_b > 0)
        collocate_products(coll_grid_, d_prods_b, nprod_b, d_rho_b_, stream_);

    // 2. ρ = ρ_α + ρ_β; n = ρ − n_ion
    add_fields_kernel<<<grid_dim, kBlock, 0, stream_>>>(d_rho_a_, d_rho_b_,
                                                        d_rho_, ntot_);
    n_tot_kernel<<<grid_dim, kBlock, 0, stream_>>>(d_rho_, d_n_ion, d_n_tot_,
                                                   ntot_);

    // 3. Poisson(n) → V_es
    poisson_.solve(d_n_tot_, d_v_es_);

    // 4. Padé LSD → V_xc^α/β, E_xc
    xc_lda_grid_lsd(d_rho_a_, d_rho_b_, d_vxc_, d_vxc_b_, d_exc_, nr, stream_);
    xc_energy_lsd(d_rho_a_, d_rho_b_, d_exc_, nr, weight_, d_e_xc_, stream_);

    // 5. Dual V_eff; optional ∫ρ V_loc on total ρ
    if (d_v_loc != nullptr)
        check_cuda(cudaMemsetAsync(d_e_loc_, 0, sizeof(double), stream_),
                   "zero e_loc");
    assemble_veff_uks_kernel<<<grid_dim, kBlock, 0, stream_>>>(
        d_v_es_, d_vxc_, d_vxc_b_, d_v_loc, d_rho_a_, d_rho_b_, d_veff_,
        d_veff_b_, (d_v_loc != nullptr) ? d_e_loc_ : nullptr, ntot_);
    if (d_v_loc != nullptr)
        scale_scalar_kernel<<<1, 1, 0, stream_>>>(d_e_loc_, weight_);
    check_cuda(cudaGetLastError(), "assemble_veff_uks");

    // 6. Optional dual integrate (shape-only; products may differ only by
    // weight — product list geometry is shared on the product path).
    if (d_acc_a != nullptr) {
        if (d_prods_a == nullptr || nprod_a <= 0)
            throw std::runtime_error(
                "cuda::gpw KsChainDevice::run_uks: d_acc_a needs α products");
        integrate_products(coll_grid_, d_prods_a, nprod_a, d_veff_, d_acc_a,
                           stream_);
    }
    if (d_acc_b != nullptr) {
        if (d_prods_b == nullptr || nprod_b <= 0)
            throw std::runtime_error(
                "cuda::gpw KsChainDevice::run_uks: d_acc_b needs β products");
        integrate_products(coll_grid_, d_prods_b, nprod_b, d_veff_b_, d_acc_b,
                           stream_);
    }

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
    check_cuda(cudaStreamSynchronize(stream_), "sync uks energies");
    return out;
}

// ── Multi-grid chain ──────────────────────────────────────────────────────

KsMultiGridChainDevice::KsMultiGridChainDevice(const MultigridDesc& mg,
                                               cudaStream_t stream)
    : mg_(mg), stream_(stream), nlev_(mg.nlevels()) {
    if (nlev_ < 1)
        throw std::runtime_error(
            "cuda::gpw KsMultiGridChainDevice: need ≥1 multigrid level");
    fine_ = std::make_unique<KsChainDevice>(mg.fine(), stream);
    d_rho_level_.assign(static_cast<std::size_t>(nlev_), nullptr);
    d_v_level_.assign(static_cast<std::size_t>(nlev_), nullptr);

    try {
        for (int l = 0; l < nlev_; ++l) {
            const long nl = mg.level(l).npts();
            if (nl <= 0)
                throw std::runtime_error(
                    "cuda::gpw KsMultiGridChainDevice: empty level grid");
            const size_t bytes = static_cast<size_t>(nl) * sizeof(double);
            check_cuda(
                cudaMalloc(&d_rho_level_[static_cast<std::size_t>(l)], bytes),
                "malloc rho_level");
            if (l > 0)
                check_cuda(
                    cudaMalloc(&d_v_level_[static_cast<std::size_t>(l)], bytes),
                    "malloc v_level");
        }
        const long nfine = mg.fine().npts();
        const size_t fbytes = static_cast<size_t>(nfine) * sizeof(double);
        check_cuda(cudaMalloc(&d_rho_fine_, fbytes), "malloc rho_fine");
        check_cuda(cudaMalloc(&d_prolong_scratch_, fbytes),
                   "malloc prolong scratch");

        xfer_.resize(static_cast<std::size_t>(std::max(0, nlev_ - 1)));
        for (int l = 1; l < nlev_; ++l) {
            xfer_[static_cast<std::size_t>(l - 1)] =
                std::make_unique<MultigridTransferDevice>(mg.level(l),
                                                          mg.fine(), stream);
        }
    } catch (...) {
        for (double* p : d_rho_level_)
            cudaFree(p);
        for (double* p : d_v_level_)
            cudaFree(p);
        cudaFree(d_rho_fine_);
        cudaFree(d_prolong_scratch_);
        throw;
    }
}

KsMultiGridChainDevice::~KsMultiGridChainDevice() {
    for (double* p : d_rho_level_)
        cudaFree(p);
    for (double* p : d_v_level_)
        cudaFree(p);
    cudaFree(d_rho_fine_);
    cudaFree(d_prolong_scratch_);
}

KsChainEnergies
KsMultiGridChainDevice::run(const DeviceProduct* const* d_prods_by_level,
                            const int* nprod_by_level, const double* d_n_ion,
                            const double* d_v_loc,
                            double* const* d_acc_by_level, KsXcKind xc) {
    if (d_prods_by_level == nullptr || nprod_by_level == nullptr)
        throw std::runtime_error(
            "cuda::gpw KsMultiGridChainDevice::run: null product tables");

    int nprod_tot = 0;
    for (int l = 0; l < nlev_; ++l) {
        if (nprod_by_level[l] < 0)
            throw std::runtime_error(
                "cuda::gpw KsMultiGridChainDevice::run: negative nprod");
        if (nprod_by_level[l] > 0 && d_prods_by_level[l] == nullptr)
            throw std::runtime_error(
                "cuda::gpw KsMultiGridChainDevice::run: null prods on level "
                "with nprod>0");
        nprod_tot += nprod_by_level[l];
    }
    if (nprod_tot <= 0)
        throw std::runtime_error(
            "cuda::gpw KsMultiGridChainDevice::run: need ≥1 product total");

    const long nfine = mg_.fine().npts();
    const size_t fbytes = static_cast<size_t>(nfine) * sizeof(double);
    check_cuda(cudaMemsetAsync(d_rho_fine_, 0, fbytes, stream_),
               "zero rho_fine");

    // 1–2. Per-level collocate + prolong into fine density.
    for (int l = 0; l < nlev_; ++l) {
        const CollocateGridDesc& g = mg_.level(l);
        const long nl = g.npts();
        const size_t bytes = static_cast<size_t>(nl) * sizeof(double);
        double* d_rl = d_rho_level_[static_cast<std::size_t>(l)];
        check_cuda(cudaMemsetAsync(d_rl, 0, bytes, stream_), "zero rho_level");
        if (nprod_by_level[l] > 0)
            collocate_products(g, d_prods_by_level[l], nprod_by_level[l], d_rl,
                               stream_);
        if (l == 0) {
            device_axpy_add(d_rl, d_rho_fine_, nfine, stream_);
        } else {
            MultigridTransferDevice& tr =
                *xfer_[static_cast<std::size_t>(l - 1)];
            tr.prolong(d_rl, d_prolong_scratch_);
            device_axpy_add(d_prolong_scratch_, d_rho_fine_, nfine, stream_);
        }
    }

    // 3–5. Fine-grid KS (no fine integrate — done per-level below).
    KsChainEnergies out = fine_->run_from_rho(d_rho_fine_, d_n_ion, d_v_loc,
                                              nullptr, 0, nullptr, xc);

    // 6. Restrict V_eff + integrate per level.
    if (d_acc_by_level != nullptr) {
        for (int l = 0; l < nlev_; ++l) {
            if (d_acc_by_level[l] == nullptr || nprod_by_level[l] <= 0)
                continue;
            const double* d_V = nullptr;
            if (l == 0) {
                d_V = fine_->d_veff();
            } else {
                double* d_vl = d_v_level_[static_cast<std::size_t>(l)];
                xfer_[static_cast<std::size_t>(l - 1)]->restrict(
                    fine_->d_veff(), d_vl);
                d_V = d_vl;
            }
            integrate_products(mg_.level(l), d_prods_by_level[l],
                               nprod_by_level[l], d_V, d_acc_by_level[l],
                               stream_);
        }
        check_cuda(cudaStreamSynchronize(stream_), "sync integrate");
    }
    return out;
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
