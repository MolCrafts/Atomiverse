/// @file cuda/planewave/internal/device_density.cu
/// @brief Device-resident density side of the plane-wave SCF.
///
/// Stage-for-stage mirror of the CPU helpers driven by `cpu::pw::PW::run`:
///
/// | method                    | CPU twin                                  |
/// |---------------------------|-------------------------------------------|
/// | `hartree()`               | `PoissonSolver::step`                     |
/// | `exchange_correlation()`  | `XCBuilder::step(ws, xc)` (LSDA or PBE)   |
/// | `assemble_veff()`         | `VeffBuilder::step`                       |
/// | `zero_accumulator()`      | `DensityBuilder::begin`                   |
/// | `collect()`               | first half of `DensityBuilder::finish`    |
/// | `mix(iter)`               | `pulay_mix`                               |
/// | `reduce(vol)`             | `ConvergenceCheck::step` + `iteration_energy`|
///
/// See the NUMERICAL PARITY CONTRACT block in device_density.cuh: the algebra
/// matches the CPU term for term, but cuFFT vs FFTW butterfly order, tree vs
/// serial reduction, and libdevice vs libm transcendentals put the two
/// backends ~1e-15 relative apart. Every device reduction here uses a FIXED
/// launch geometry, so it is bitwise reproducible run-to-run.
///
/// Synchronization: everything runs on one non-default stream owned by this
/// object (`stream_`). Kernels, cuFFT and cuBLAS are stream-ordered; no
/// `cudaDeviceSynchronize` is issued. The only host waits are the two
/// stream-scoped D2H fences documented at their call sites (`mix`'s Gram
/// readback and `reduce`'s 5-scalar readback); both are unavoidable because
/// the Pulay coefficient solve and the SCF convergence/energy report are
/// host decisions.

#include "atomiverse/cuda/planewave/internal/device_density.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/planewave/internal/grid_map.cuh"
#include "atomiverse/cuda/planewave/internal/poisson.cuh"
#include "atomiverse/cuda/planewave/internal/veff.cuh"
#include "atomiverse/cuda/planewave/internal/xc_lda.cuh"
#include "atomiverse/cuda/planewave/internal/xc_pbe.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

static_assert(std::is_same<AccReal, double>::value,
              "cuda::pw::internal::DeviceDensity assumes ATV_REAL_WORK=double "
              "— every host↔device copy reinterprets std::complex<AccReal> as "
              "cufftDoubleComplex");
static_assert(sizeof(std::complex<double>) == sizeof(cufftDoubleComplex),
              "std::complex<double> must be layout-compatible with "
              "cufftDoubleComplex for the raw G-sphere copies");

namespace {

constexpr int kBlock = ::ATV::cuda::kDefaultBlock; // 256
/// Fixed stage-1 grid for every reduction in this TU. Keeping it independent
/// of n_g / nr is what makes the summation tree — and hence the result —
/// reproducible run to run.
constexpr int kReduceBlocks = 256;
static_assert(kReduceBlocks == kBlock,
              "stage 2 reduces kReduceBlocks partials with kBlock threads");
/// Reduction lanes fused into one pass over the G-sphere.
constexpr int kLanes = 4;

inline int n_blocks(int n) { return ::ATV::cuda::grid_size(n, kBlock); }

/// Per-lane finalize instructions, passed by value to the stage-2 kernel.
struct FinalizeSpec {
    double scale[kLanes];
    int take_sqrt[kLanes];
};

// ── Kernels ──────────────────────────────────────────────────────────────

/// Zero the imaginary part of the G = 0 component. Mirrors the CPU's
/// `ws.V_xc_G[0] = {V_xc_G[0].real(), 0}` closing line of XCBuilder::step.
__global__ void pw_zero_imag0_kernel(cufftDoubleComplex* __restrict__ v) {
    v[0].y = 0.0;
}

/// out = i·G_d·z on the G-sphere — the spectral-derivative factor of
/// XCBuilder's gradient pack lambda: complex(0, G_d)·ρ = (−G_d·Im, G_d·Re).
__global__ void pw_mul_ig_kernel(const cufftDoubleComplex* __restrict__ z,
                                 const double* __restrict__ gd, int n_g,
                                 cufftDoubleComplex* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    const cufftDoubleComplex v = z[i];
    const double g = gd[i];
    out[i].x = -g * v.y;
    out[i].y = g * v.x;
}

/// acc += i·G_d·z — the divergence accumulation of XCBuilder's GGA pass
/// (`V_xc_G[ig] += (−G_d·im, G_d·re)`), launched once per axis in x,y,z
/// order so the association matches the CPU loop.
__global__ void pw_axpy_ig_kernel(const cufftDoubleComplex* __restrict__ z,
                                  const double* __restrict__ gd, int n_g,
                                  cufftDoubleComplex* __restrict__ acc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    const cufftDoubleComplex v = z[i];
    const double g = gd[i];
    acc[i].x += -g * v.y;
    acc[i].y += g * v.x;
}

/// v -= d, elementwise — XCBuilder's closing `V_xc_r[i] -= div_r[i]`.
__global__ void pw_sub_kernel(double* __restrict__ v,
                              const double* __restrict__ d, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    v[i] -= d[i];
}

/// F = ρ − ρ_old into the residual ring slot, ρ_old into the density ring
/// slot — one pass, matching `pulay_res.push_back(F)` /
/// `pulay_rho.push_back(rho_G_old)`.
__global__ void
pw_push_history_kernel(const cufftDoubleComplex* __restrict__ rho,
                       const cufftDoubleComplex* __restrict__ rho_old, int n_g,
                       cufftDoubleComplex* __restrict__ res_slot,
                       cufftDoubleComplex* __restrict__ rho_slot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    const cufftDoubleComplex r = rho[i];
    const cufftDoubleComplex o = rho_old[i];
    res_slot[i].x = r.x - o.x;
    res_slot[i].y = r.y - o.y;
    rho_slot[i] = o;
}

/// ρ = ρ_old + β F (the CPU's `depth < 2` branch).
__global__ void
pw_linear_mix_kernel(cufftDoubleComplex* __restrict__ rho,
                     const cufftDoubleComplex* __restrict__ rho_old,
                     const cufftDoubleComplex* __restrict__ res, double beta,
                     int n_g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    rho[i].x = rho_old[i].x + beta * res[i].x;
    rho[i].y = rho_old[i].y + beta * res[i].y;
}

/// ρ = Σ_{i<m} c_i (ρ_in^i + β F^i), accumulating over i in the CPU's order
/// (oldest history entry first) so the association matches.
__global__ void
pw_pulay_combine_kernel(cufftDoubleComplex* __restrict__ rho,
                        const cufftDoubleComplex* __restrict__ rho_hist,
                        const cufftDoubleComplex* __restrict__ res_hist,
                        const double* __restrict__ c, double beta, int n_g,
                        int m, int hist_start, int max_depth) {
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= n_g)
        return;
    double acc_r = 0.0, acc_i = 0.0;
    for (int i = 0; i < m; ++i) {
        const int slot = (hist_start + i) % max_depth;
        const size_t o = static_cast<size_t>(slot) * n_g + ig;
        const double ci = c[i];
        acc_r += ci * (rho_hist[o].x + beta * res_hist[o].x);
        acc_i += ci * (rho_hist[o].y + beta * res_hist[o].y);
    }
    rho[ig].x = acc_r;
    rho[ig].y = acc_i;
}

/// Fused stage-1 reduction over the G-sphere. One pass over ρ(G) feeds four
/// lanes (Hartree, XC double counting, local PP, density change), which is
/// 5 array reads and 1 launch instead of 8 reads and 4 launches. At the
/// ng ~ 1e4 scale of a PW SCF these kernels are launch-bound, which is the
/// honest justification for fusing them.
///
/// Grid is FIXED at kReduceBlocks blocks with a grid-stride loop, so the
/// partition of the sum — and therefore the reduction tree — is identical on
/// every run. Threads past n_g contribute the additive identity 0.
__global__ void
pw_reduce_terms_kernel(const cufftDoubleComplex* __restrict__ rho,
                       const cufftDoubleComplex* __restrict__ rho_old,
                       const cufftDoubleComplex* __restrict__ vh,
                       const cufftDoubleComplex* __restrict__ vxc,
                       const cufftDoubleComplex* __restrict__ vloc, int n_g,
                       double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc[kLanes] = {0.0, 0.0, 0.0, 0.0};

    // Compile-time stride, not gridDim.x*blockDim.x: the partition of the sum
    // is part of this kernel's contract (it is what makes the result
    // reproducible), and it must match the fixed <<<kReduceBlocks, kBlock>>>
    // launch below the same way the output stride does.
    for (int i = blockIdx.x * kBlock + threadIdx.x; i < n_g;
         i += kReduceBlocks * kBlock) {
        const cufftDoubleComplex r = rho[i];
        const cufftDoubleComplex o = rho_old[i];
        const cufftDoubleComplex h = vh[i];
        const cufftDoubleComplex x = vxc[i];
        const cufftDoubleComplex p = vloc[i];
        // Re[conj(ρ)·V] = ρ_re V_re + ρ_im V_im
        acc[0] += r.x * h.x + r.y * h.y;
        acc[1] += r.x * x.x + r.y * x.y;
        acc[2] += r.x * p.x + r.y * p.y;
        const double dr = r.x - o.x, di = r.y - o.y;
        acc[3] += dr * dr + di * di;
    }

    for (int lane = 0; lane < kLanes; ++lane) {
        s[threadIdx.x] = acc[lane];
        __syncthreads();
        for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                s[threadIdx.x] += s[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            // kReduceBlocks, not gridDim.x: the stride must match what
            // pw_reduce_finalize_kernel reads, by contract not by launch.
            partials[lane * kReduceBlocks + blockIdx.x] = s[0];
        // Write-after-read fence: thread 0's read of s[0] must complete on
        // every lane before the next iteration overwrites s[threadIdx.x].
        __syncthreads();
    }
}

/// Stage 2: one block per lane, each reducing exactly kReduceBlocks partials
/// and applying that lane's scale (and √ for the residual norm).
__global__ void pw_reduce_finalize_kernel(const double* __restrict__ partials,
                                          int n_parts, FinalizeSpec spec,
                                          double* __restrict__ out) {
    __shared__ double s[kReduceBlocks];
    const int lane = blockIdx.x;
    const int t = threadIdx.x;
    s[t] = (t < n_parts) ? partials[lane * n_parts + t] : 0.0;
    __syncthreads();
    for (int stride = kReduceBlocks / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        const double v = s[0] * spec.scale[lane];
        out[lane] = spec.take_sqrt[lane] ? sqrt(v) : v;
    }
}

// ── Allocation helpers (same checked idiom as device_solver.cu) ──────────

template <typename T>
void device_free(T*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

template <typename T>
T* device_alloc(size_t n) {
    T* p = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(
            std::string("cuda::pw::internal::DeviceDensity cuBLAS error in ")
            + what + ": " + std::to_string(static_cast<int>(s)));
}

} // namespace

// ── DeviceDensity ────────────────────────────────────────────────────────

DeviceDensity::DeviceDensity(const cpu::pw::internal::PWProblem& prob,
                             double mix_beta, int pulay_depth,
                             cpu::scf::XCType xc, bool spin_polarized,
                             int n_alpha, int n_beta, int n_elec)
    : ng0_(prob.gv0.count()), nr_(prob.ws.nr), nc_(prob.ws.nc),
      depth_(pulay_depth), mix_beta_(mix_beta),
      omega_(static_cast<double>(prob.ws.omega)), xc_(xc),
      gga_(cpu::scf::is_gga(xc)), spin_(spin_polarized),
      // Temporary default-stream FFT; rebound after stream_ is created below.
      // FFTPlan3D cannot be default-constructed, so we build it first then
      // rebind via set_stream once stream_ exists.
      fft_(prob.ws.fft_nx, prob.ws.fft_ny, prob.ws.fft_nz) {
    if (pulay_depth < 1)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity: pulay_depth must be >= 1");
    // Only the two functionals with device kernels; no silent fallback. The
    // PBE-family variants (PBEsol/revPBE/RPBE) differ by constants the
    // xc_pbe.cu kernel hard-wires, so they are NOT accepted here.
    if (xc_ != cpu::scf::XCType::LSDA && xc_ != cpu::scf::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity: only XCType::LSDA and "
            "XCType::PBE have device XC kernels");
    // Collinear: LSDA and true spin-GGA PBE are landed; other GGAs are not.
    if (spin_ && xc_ != cpu::scf::XCType::LSDA && xc_ != cpu::scf::XCType::PBE)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity: spin_polarized supports "
            "XCType::LSDA and XCType::PBE only on the device path");
    if (ng0_ <= 0 || nr_ <= 0 || nc_ <= 0)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity: the assembled problem has an "
            "empty density G-sphere or FFT grid");
    // d_half_ is sized from ws.nc, the transforms from ws.fft_n*. If those
    // ever disagree, `scatter_g_to_halfcomplex` writes past the buffer and
    // the only symptom is garbage or an illegal-access at run time.
    if (nr_ != fft_.nr() || nc_ != fft_.nc())
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity: ws.nr/ws.nc disagree with the "
            "ws.fft_nx/ny/nz grid the cuFFT plans were built from");

    try {
        // One non-default stream for the whole density side. The band solver
        // borrows it so SCF stages stay ordered without events.
        ATV_CUDA_CHECK(cudaStreamCreate(&stream_));
        fft_.set_stream(stream_);

        check_cublas(cublasCreate(&blas_), "cublasCreate");
        check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream");

        const size_t ng = static_cast<size_t>(ng0_);
        const size_t nr = static_cast<size_t>(nr_);
        const size_t nc = static_cast<size_t>(nc_);
        const size_t ring = static_cast<size_t>(depth_) * ng;

        d_h_ = device_alloc<int>(ng);
        d_k_ = device_alloc<int>(ng);
        d_l_ = device_alloc<int>(ng);
        d_g2_ = device_alloc<double>(ng);
        d_rho_G_ = device_alloc<cufftDoubleComplex>(ng);
        d_rho_G_old_ = device_alloc<cufftDoubleComplex>(ng);
        d_V_H_G_ = device_alloc<cufftDoubleComplex>(ng);
        d_V_xc_G_ = device_alloc<cufftDoubleComplex>(ng);
        d_V_loc_G_ = device_alloc<cufftDoubleComplex>(ng);
        d_V_eff_G_ = device_alloc<cufftDoubleComplex>(ng);
        d_res_hist_ = device_alloc<cufftDoubleComplex>(ring);
        d_rho_hist_ = device_alloc<cufftDoubleComplex>(ring);
        d_gram_ = device_alloc<cufftDoubleComplex>(static_cast<size_t>(depth_)
                                                   * depth_);
        d_coeff_ = device_alloc<double>(static_cast<size_t>(depth_));
        d_half_ = device_alloc<cufftDoubleComplex>(nc);
        d_rho_r_in_ = device_alloc<double>(nr);
        d_rho_r_acc_ = device_alloc<double>(nr);
        d_V_xc_r_ = device_alloc<double>(nr);
        d_exc_r_ = device_alloc<double>(nr);
        d_V_eff_r_ = device_alloc<double>(nr);
        d_partials_ =
            device_alloc<double>(static_cast<size_t>(kLanes) * kReduceBlocks);
        d_xc_partials_ =
            device_alloc<double>(static_cast<size_t>(kXcEnergyPartials));
        d_scalars_ = device_alloc<double>(5);

        if (gga_) {
            d_gx_ = device_alloc<double>(ng);
            d_gy_ = device_alloc<double>(ng);
            d_gz_ = device_alloc<double>(ng);
            d_sph_ = device_alloc<cufftDoubleComplex>(ng);
            d_grad_x_r_ = device_alloc<double>(nr);
            d_grad_y_r_ = device_alloc<double>(nr);
            d_grad_z_r_ = device_alloc<double>(nr);
            d_div_r_ = device_alloc<double>(nr);
            if (spin_) {
                d_grad_dn_x_r_ = device_alloc<double>(nr);
                d_grad_dn_y_r_ = device_alloc<double>(nr);
                d_grad_dn_z_r_ = device_alloc<double>(nr);
                d_div_dn_r_ = device_alloc<double>(nr);
            }
        }

        if (spin_) {
            for (int s = 0; s < 2; ++s) {
                d_rho_spin_G_[s] = device_alloc<cufftDoubleComplex>(ng);
                d_rho_spin_G_old_[s] = device_alloc<cufftDoubleComplex>(ng);
                d_rho_spin_out_G_[s] = device_alloc<cufftDoubleComplex>(ng);
                d_V_xc_spin_G_[s] = device_alloc<cufftDoubleComplex>(ng);
                d_rho_spin_r_[s] = device_alloc<double>(nr);
                d_V_xc_spin_r_[s] = device_alloc<double>(nr);
            }
        }

        // Zero the Pulay ring: unfilled slots then contribute finite zeros to
        // the Gram matrix instead of whatever the allocator handed back. The
        // corresponding rows/columns are discarded on the host, but a NaN
        // there would be a debugging trap for no reason. Stream-ordered so
        // the first SCF stage cannot race an incomplete memset.
        ATV_CUDA_CHECK(cudaMemsetAsync(
            d_res_hist_, 0, ring * sizeof(cufftDoubleComplex), stream_));
        ATV_CUDA_CHECK(cudaMemsetAsync(
            d_rho_hist_, 0, ring * sizeof(cufftDoubleComplex), stream_));
        ATV_CUDA_CHECK(cudaMemsetAsync(
            d_rho_G_old_, 0, ng * sizeof(cufftDoubleComplex), stream_));
        ATV_CUDA_CHECK(
            cudaMemsetAsync(d_scalars_, 0, 5 * sizeof(double), stream_));

        // ── One-time uploads (never repeated per SCF iteration) ──
        const auto& gv = prob.gv0;
        std::vector<double> g2(ng);
        for (int i = 0; i < ng0_; ++i)
            g2[static_cast<size_t>(i)] = static_cast<double>(
                gv.g[static_cast<size_t>(i)].norm_squared());

        ATV_CUDA_CHECK(cudaMemcpyAsync(d_h_, gv.h.data(), ng * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_k_, gv.k.data(), ng * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_l_, gv.l.data(), ng * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_g2_, g2.data(), ng * sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_V_loc_G_, prob.ws.V_loc_G.data(),
                                       ng * sizeof(cufftDoubleComplex),
                                       cudaMemcpyHostToDevice, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_G_, prob.ws.rho_G.data(),
                                       ng * sizeof(cufftDoubleComplex),
                                       cudaMemcpyHostToDevice, stream_));

        if (spin_) {
            // Seed ρ_↑ / ρ_↓ by splitting the atomic total density (CPU twin).
            const double fa = (n_elec > 0) ? static_cast<double>(n_alpha)
                                                 / static_cast<double>(n_elec)
                                           : 0.5;
            const double fb = (n_elec > 0) ? static_cast<double>(n_beta)
                                                 / static_cast<double>(n_elec)
                                           : 0.5;
            std::vector<std::complex<double>> rho_tot(ng);
            ATV_CUDA_CHECK(cudaMemcpyAsync(rho_tot.data(), d_rho_G_,
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyDeviceToHost, stream_));
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::vector<std::complex<double>> rho_a(ng), rho_b(ng);
            for (size_t ig = 0; ig < ng; ++ig) {
                rho_a[ig] = fa * rho_tot[ig];
                rho_b[ig] = fb * rho_tot[ig];
            }
            ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_spin_G_[0], rho_a.data(),
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyHostToDevice, stream_));
            ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_spin_G_[1], rho_b.data(),
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyHostToDevice, stream_));
            ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_spin_G_old_[0], rho_a.data(),
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyHostToDevice, stream_));
            ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_spin_G_old_[1], rho_b.data(),
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyHostToDevice, stream_));
            // Total ρ must match the spin sum after the seed.
            for (size_t ig = 0; ig < ng; ++ig)
                rho_tot[ig] = rho_a[ig] + rho_b[ig];
            ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_G_, rho_tot.data(),
                                           ng * sizeof(cufftDoubleComplex),
                                           cudaMemcpyHostToDevice, stream_));
        }

        if (gga_) {
            // Cartesian G components per axis — the CPU's gv.g[ig][d],
            // geometry only, uploaded once for the spectral ∇ρ / divergence.
            std::vector<double> gcart(ng);
            double* dst[3] = {d_gx_, d_gy_, d_gz_};
            for (int d = 0; d < 3; ++d) {
                for (int i = 0; i < ng0_; ++i)
                    gcart[static_cast<size_t>(i)] =
                        static_cast<double>(gv.g[static_cast<size_t>(i)][d]);
                ATV_CUDA_CHECK(
                    cudaMemcpyAsync(dst[d], gcart.data(), ng * sizeof(double),
                                    cudaMemcpyHostToDevice, stream_));
            }
        }
        // Fence before the host returns: the atomic ρ seed and V_loc must be
        // resident before the first hartree()/band solve can touch them, and
        // the host g2 vector goes out of scope here.
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    } catch (...) {
        free_all();
        throw;
    }
}

DeviceDensity::~DeviceDensity() { free_all(); }

void DeviceDensity::free_all() {
    // Drain the density stream, rebind cuFFT off it, then free. Order matters:
    // cufftDestroy in ~FFTPlan3D must not see a destroyed stream handle.
    if (stream_) {
        cudaStreamSynchronize(stream_);
        try {
            fft_.set_stream(nullptr);
        } catch (...) {
            // Best-effort rebind on the teardown path.
        }
    }
    device_free(d_h_);
    device_free(d_k_);
    device_free(d_l_);
    device_free(d_g2_);
    device_free(d_rho_G_);
    device_free(d_rho_G_old_);
    device_free(d_V_H_G_);
    device_free(d_V_xc_G_);
    device_free(d_V_loc_G_);
    device_free(d_V_eff_G_);
    device_free(d_res_hist_);
    device_free(d_rho_hist_);
    device_free(d_gram_);
    device_free(d_coeff_);
    device_free(d_half_);
    device_free(d_rho_r_in_);
    device_free(d_rho_r_acc_);
    device_free(d_V_xc_r_);
    device_free(d_exc_r_);
    device_free(d_V_eff_r_);
    device_free(d_partials_);
    device_free(d_xc_partials_);
    device_free(d_scalars_);
    device_free(d_gx_);
    device_free(d_gy_);
    device_free(d_gz_);
    device_free(d_sph_);
    device_free(d_grad_x_r_);
    device_free(d_grad_y_r_);
    device_free(d_grad_z_r_);
    device_free(d_div_r_);
    device_free(d_grad_dn_x_r_);
    device_free(d_grad_dn_y_r_);
    device_free(d_grad_dn_z_r_);
    device_free(d_div_dn_r_);
    for (int s = 0; s < 2; ++s) {
        device_free(d_rho_spin_G_[s]);
        device_free(d_rho_spin_G_old_[s]);
        device_free(d_rho_spin_out_G_[s]);
        device_free(d_V_xc_spin_G_[s]);
        device_free(d_rho_spin_r_[s]);
        device_free(d_V_xc_spin_r_[s]);
    }
    if (blas_) {
        cublasDestroy(blas_);
        blas_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void DeviceDensity::hartree() {
    poisson_hartree(d_rho_G_, d_g2_, d_V_H_G_, ng0_, stream_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::exchange_correlation() {
    const int nx = fft_.nx(), ny = fft_.ny(), nz = fft_.nz();

    // ρ(G) → ρ(r). c2r is UNNORMALIZED on both backends:
    // ρ(r) = Σ_G ρ(G) e^{iGr}.
    scatter_g_to_halfcomplex(d_rho_G_, d_h_, d_k_, d_l_, ng0_, d_half_, nx, ny,
                             nz, stream_);
    fft_.c2r(d_half_, d_rho_r_in_);

    if (gga_) {
        // ── Spectral ∇ρ: ∂_dρ(r) = c2r[ i G_d ρ(G) ], per axis ──
        double* grad_r[3] = {d_grad_x_r_, d_grad_y_r_, d_grad_z_r_};
        const double* gcart[3] = {d_gx_, d_gy_, d_gz_};
        for (int d = 0; d < 3; ++d) {
            pw_mul_ig_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
                d_rho_G_, gcart[d], ng0_, d_sph_);
            ATV_CUDA_CHECK(cudaGetLastError());
            scatter_g_to_halfcomplex(d_sph_, d_h_, d_k_, d_l_, ng0_, d_half_,
                                     nx, ny, nz, stream_);
            fft_.c2r(d_half_, grad_r[d]);
        }

        // Pointwise closed-shell PBE: local v_ρ, ε_xc, and the flux written
        // over the gradient arrays. E_xc quadrature exactly as the LSDA path
        // (dV = Ω/nr); the CPU accumulates it in the same grid loop, BEFORE
        // the divergence correction, and ε_xc is untouched by it.
        xc_pbe_grid(d_rho_r_in_, d_grad_x_r_, d_grad_y_r_, d_grad_z_r_,
                    d_V_xc_r_, d_exc_r_, nr_, stream_);
        xc_energy(d_rho_r_in_, d_exc_r_, nr_, omega_ / static_cast<double>(nr_),
                  d_scalars_ + 0, d_xc_partials_, stream_);

        // ── v_xc −= ∇·flux, via div(G) = Σ_d i G_d flux_d(G) ──
        // d_V_xc_G_ is the accumulation scratch here, exactly as the CPU
        // reuses ws.V_xc_G — it is overwritten by the final r2c gather below.
        ATV_CUDA_CHECK(cudaMemsetAsync(
            d_V_xc_G_, 0,
            static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex), stream_));
        for (int d = 0; d < 3; ++d) {
            fft_.r2c(grad_r[d], d_half_); // grad_r now holds flux_d
            gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_, d_sph_, nx,
                                    ny, nz, 1.0 / static_cast<double>(nr_),
                                    stream_);
            pw_axpy_ig_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
                d_sph_, gcart[d], ng0_, d_V_xc_G_);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        scatter_g_to_halfcomplex(d_V_xc_G_, d_h_, d_k_, d_l_, ng0_, d_half_, nx,
                                 ny, nz, stream_);
        fft_.c2r(d_half_, d_div_r_);
        pw_sub_kernel<<<n_blocks(nr_), kBlock, 0, stream_>>>(d_V_xc_r_,
                                                             d_div_r_, nr_);
        ATV_CUDA_CHECK(cudaGetLastError());
    } else {
        // Pointwise LSDA + the E_xc quadrature (dV = Ω/nr, as on the CPU).
        xc_lsda_grid(d_rho_r_in_, d_V_xc_r_, d_exc_r_, nr_, stream_);
        xc_energy(d_rho_r_in_, d_exc_r_, nr_, omega_ / static_cast<double>(nr_),
                  d_scalars_ + 0, d_xc_partials_, stream_);
    }

    // V_xc(r) → V_xc(G), scaled by 1/nr (r2c is unnormalized).
    // NOTE: d_half_ is reused by every scatter/gather+FFT pair of an
    // iteration. They are strictly sequential on stream_, and
    // FFTPlan3D::c2r copies its input into private scratch first, so no pair
    // can clobber another. An "optimization" that drops a re-scatter would
    // break that silently.
    fft_.r2c(d_V_xc_r_, d_half_);
    gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_, d_V_xc_G_, nx, ny,
                            nz, 1.0 / static_cast<double>(nr_), stream_);
    pw_zero_imag0_kernel<<<1, 1, 0, stream_>>>(d_V_xc_G_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::assemble_veff() {
    veff_assemble_g(d_V_loc_G_, d_V_H_G_, d_V_xc_G_, d_V_eff_G_, ng0_, stream_);
    scatter_g_to_halfcomplex(d_V_eff_G_, d_h_, d_k_, d_l_, ng0_, d_half_,
                             fft_.nx(), fft_.ny(), fft_.nz(), stream_);
    // Unnormalized c2r: V_eff(r) = Σ_G V_eff(G) e^{iGr}, matching VeffBuilder.
    fft_.c2r(d_half_, d_V_eff_r_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::zero_accumulator() {
    ATV_CUDA_CHECK(cudaMemsetAsync(
        d_rho_r_acc_, 0, static_cast<size_t>(nr_) * sizeof(double), stream_));
}

void DeviceDensity::collect() {
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(d_rho_G_old_, d_rho_G_,
                        static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_));
    fft_.r2c(d_rho_r_acc_, d_half_);
    gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_, d_rho_G_,
                            fft_.nx(), fft_.ny(), fft_.nz(),
                            1.0 / static_cast<double>(nr_), stream_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::mix(int iter) {
    const size_t ng = static_cast<size_t>(ng0_);
    // Ring bookkeeping reproducing the CPU's front-trimmed std::vector:
    // history entry i of m corresponds to SCF iteration (iter - m + 1 + i).
    const int slot_push = iter % depth_;
    const int m = std::min(iter + 1, depth_);
    const int hist_start = (iter + 1 > depth_) ? (iter + 1) % depth_ : 0;

    pw_push_history_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
        d_rho_G_, d_rho_G_old_, ng0_,
        d_res_hist_ + static_cast<size_t>(slot_push) * ng,
        d_rho_hist_ + static_cast<size_t>(slot_push) * ng);
    ATV_CUDA_CHECK(cudaGetLastError());

    if (m < 2) {
        // CPU's `depth < 2` branch: plain linear mixing.
        pw_linear_mix_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
            d_rho_G_, d_rho_G_old_,
            d_res_hist_ + static_cast<size_t>(slot_push) * ng, mix_beta_, ng0_);
        ATV_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Gram matrix of the whole ring in one call: G = FᴴF, column-major
    // depth×depth, so G[j*depth + i] = ⟨F_i|F_j⟩. Columns belonging to
    // not-yet-filled slots are computed and discarded — depth ≤ 8 against
    // ng ~ 1e4, so the waste is noise, and one GEMM beats m(m+1)/2 dots each
    // of which would be its own launch.
    const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
    check_cublas(
        cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, depth_, depth_, ng0_, &one,
                    reinterpret_cast<const cuDoubleComplex*>(d_res_hist_), ng0_,
                    reinterpret_cast<const cuDoubleComplex*>(d_res_hist_), ng0_,
                    &zero, reinterpret_cast<cuDoubleComplex*>(d_gram_), depth_),
        "Zgemm pulay gram");

    // ── SYNC POINT 1 (per iteration) ──
    // Stream-scoped D2H of depth² complex doubles (≤ 8²·16 B = 1 KB).
    // Unavoidable: the augmented Lagrangian solve is a host algorithm shared
    // bit-for-bit with cpu::pw::pulay_mix, and its coefficients feed the very
    // next kernel. Async copy + stream sync fences only this chain — not the
    // default stream / whole device.
    std::vector<std::complex<double>> gram(static_cast<size_t>(depth_)
                                           * depth_);
    ATV_CUDA_CHECK(cudaMemcpyAsync(gram.data(), d_gram_,
                                   gram.size() * sizeof(cufftDoubleComplex),
                                   cudaMemcpyDeviceToHost, stream_));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Row-major m×m real Gram, B[i][j] = Re⟨F_i|F_j⟩ — the CPU's
    // Σ(Fi.re·Fj.re + Fi.im·Fj.im). Only the upper triangle is read and it is
    // mirrored, exactly as the CPU does: cuBLAS computes G[i][j] and G[j][i]
    // independently, so their real parts agree mathematically but need not
    // agree bit-for-bit, and an asymmetric B would send the pivoting down a
    // different path than the CPU's.
    std::vector<AccReal> B(static_cast<size_t>(m) * m, AccReal(0));
    for (int i = 0; i < m; ++i) {
        const int si = (hist_start + i) % depth_;
        for (int j = i; j < m; ++j) {
            const int sj = (hist_start + j) % depth_;
            const AccReal dot = static_cast<AccReal>(
                gram[static_cast<size_t>(sj) * depth_ + si].real());
            B[static_cast<size_t>(i) * m + j] = dot;
            B[static_cast<size_t>(j) * m + i] = dot;
        }
    }

    std::vector<AccReal> c(static_cast<size_t>(m), AccReal(0));
    cpu::pw::internal::solve_pulay_coefficients(B.data(), m, c.data());

    std::vector<double> c_d(c.begin(), c.end());
    ATV_CUDA_CHECK(cudaMemcpyAsync(d_coeff_, c_d.data(),
                                   static_cast<size_t>(m) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));

    pw_pulay_combine_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
        d_rho_G_, d_rho_hist_, d_res_hist_, d_coeff_, mix_beta_, ng0_, m,
        hist_start, depth_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

DeviceDensity::IterationScalars DeviceDensity::reduce(double vol_bohr) {
    pw_reduce_terms_kernel<<<kReduceBlocks, kBlock, 0, stream_>>>(
        d_rho_G_, d_rho_G_old_, d_V_H_G_, d_V_xc_G_, d_V_loc_G_, ng0_,
        d_partials_);
    ATV_CUDA_CHECK(cudaGetLastError());

    FinalizeSpec spec{};
    // CPU: e_h accumulates 0.5·Re[·] then multiplies by Ω. Factoring the 0.5
    // out is exact (a power of two), so this is the same number.
    spec.scale[0] = 0.5 * vol_bohr;
    spec.scale[1] = vol_bohr;
    spec.scale[2] = vol_bohr;
    spec.scale[3] = 1.0;
    spec.take_sqrt[0] = 0;
    spec.take_sqrt[1] = 0;
    spec.take_sqrt[2] = 0;
    spec.take_sqrt[3] = 1; // ‖Δρ‖₂
    pw_reduce_finalize_kernel<<<kLanes, kReduceBlocks, 0, stream_>>>(
        d_partials_, kReduceBlocks, spec, d_scalars_ + 1);
    ATV_CUDA_CHECK(cudaGetLastError());

    // ── SYNC POINT 2 (per iteration) ──
    // Stream-scoped D2H of 5 doubles (40 B): E_xc (written by
    // exchange_correlation) plus the four lanes above. Unavoidable: the SCF
    // stop decision and the per-iteration energy report are host-side, and
    // `e_one = e_band − 2 e_h − e_xc_int` must be evaluated on the host in the
    // CPU's exact order (catastrophic cancellation — never reassociate it).
    double host[5] = {0, 0, 0, 0, 0};
    ATV_CUDA_CHECK(cudaMemcpyAsync(host, d_scalars_, 5 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));

    IterationScalars out;
    out.e_xc = host[0];
    out.e_h = host[1];
    out.e_xc_int = host[2];
    out.e_loc = host[3];
    out.resid = host[4];
    return out;
}

void DeviceDensity::fetch_g(GField which, std::complex<double>* out) const {
    const cufftDoubleComplex* src = nullptr;
    switch (which) {
    case GField::rho:
        src = d_rho_G_;
        break;
    case GField::v_hartree:
        src = d_V_H_G_;
        break;
    case GField::v_xc:
        src = d_V_xc_G_;
        break;
    case GField::v_loc:
        src = d_V_loc_G_;
        break;
    case GField::v_eff:
        src = d_V_eff_G_;
        break;
    }
    // Diagnostic path: fence the density stream so the readback sees the
    // last stage's writes, then blocking D2H (not on the SCF hot path).
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    ATV_CUDA_CHECK(cudaMemcpy(
        out, src, static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
        cudaMemcpyDeviceToHost));
}

void DeviceDensity::fetch_veff_r(double* out) const {
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    ATV_CUDA_CHECK(cudaMemcpy(out, d_V_eff_r_,
                              static_cast<size_t>(nr_) * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void DeviceDensity::fetch_rho_r(double* out) {
    // Unnormalized c2r: ρ(r) = Σ_G ρ(G) e^{iGr} — same convention as
    // exchange_correlation()'s first step and the CPU rho_g_to_real path.
    const int nx = fft_.nx(), ny = fft_.ny(), nz = fft_.nz();
    scatter_g_to_halfcomplex(d_rho_G_, d_h_, d_k_, d_l_, ng0_, d_half_, nx, ny,
                             nz, stream_);
    fft_.c2r(d_half_, d_rho_r_in_);
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    ATV_CUDA_CHECK(cudaMemcpy(out, d_rho_r_in_,
                              static_cast<size_t>(nr_) * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void DeviceDensity::load_rho_r_scaled(const double* rho_r0, int nr0,
                                      double scale) {
    // Host-side scale + optional mean fallback when FFT dims differ under
    // strain (rare at fixed ecut for small h). Then r2c → ρ(G) with 1/nr.
    std::vector<double> rho_r(static_cast<size_t>(nr_), 0.0);
    if (nr0 == nr_ && rho_r0 != nullptr) {
        for (int i = 0; i < nr_; ++i)
            rho_r[static_cast<size_t>(i)] =
                rho_r0[static_cast<size_t>(i)] * scale;
    } else {
        double mean = 0.0;
        if (rho_r0 != nullptr && nr0 > 0) {
            for (int i = 0; i < nr0; ++i)
                mean += rho_r0[static_cast<size_t>(i)];
            mean /= static_cast<double>(nr0);
        }
        const double v = mean * scale;
        for (int i = 0; i < nr_; ++i)
            rho_r[static_cast<size_t>(i)] = v;
    }
    ATV_CUDA_CHECK(cudaMemcpyAsync(d_rho_r_acc_, rho_r.data(),
                                   static_cast<size_t>(nr_) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));
    fft_.r2c(d_rho_r_acc_, d_half_);
    gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_, d_rho_G_,
                            fft_.nx(), fft_.ny(), fft_.nz(),
                            1.0 / static_cast<double>(nr_), stream_);
    ATV_CUDA_CHECK(cudaGetLastError());
    // Keep rho_old in sync so a stray reduce() residual is not huge noise
    // (NSCF stress never mixes; residual is unused).
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(d_rho_G_old_, d_rho_G_,
                        static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_));
}

void DeviceDensity::fetch_rho_spin_r(int spin, double* out) {
    if (!spin_)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::fetch_rho_spin_r: not a spin "
            "density object");
    if (spin < 0 || spin > 1)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::fetch_rho_spin_r: spin must "
            "be 0 or 1");
    if (out == nullptr)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::fetch_rho_spin_r: out is null");
    const int nx = fft_.nx(), ny = fft_.ny(), nz = fft_.nz();
    scatter_g_to_halfcomplex(d_rho_spin_G_[spin], d_h_, d_k_, d_l_, ng0_,
                             d_half_, nx, ny, nz, stream_);
    fft_.c2r(d_half_, d_rho_r_in_);
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    ATV_CUDA_CHECK(cudaMemcpy(out, d_rho_r_in_,
                              static_cast<size_t>(nr_) * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

void DeviceDensity::load_rho_spin_r_scaled(const double* rho_a0,
                                           const double* rho_b0, int nr0,
                                           double scale) {
    if (!spin_)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::load_rho_spin_r_scaled: not a "
            "spin density object");
    auto load_one = [&](const double* rho0, cufftDoubleComplex* d_rho_G) {
        std::vector<double> rho_r(static_cast<size_t>(nr_), 0.0);
        if (nr0 == nr_ && rho0 != nullptr) {
            for (int i = 0; i < nr_; ++i)
                rho_r[static_cast<size_t>(i)] =
                    rho0[static_cast<size_t>(i)] * scale;
        } else {
            double mean = 0.0;
            if (rho0 != nullptr && nr0 > 0) {
                for (int i = 0; i < nr0; ++i)
                    mean += rho0[static_cast<size_t>(i)];
                mean /= static_cast<double>(nr0);
            }
            const double v = mean * scale;
            for (int i = 0; i < nr_; ++i)
                rho_r[static_cast<size_t>(i)] = v;
        }
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(d_rho_r_acc_, rho_r.data(),
                            static_cast<size_t>(nr_) * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
        fft_.r2c(d_rho_r_acc_, d_half_);
        gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_, d_rho_G,
                                fft_.nx(), fft_.ny(), fft_.nz(),
                                1.0 / static_cast<double>(nr_), stream_);
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            // keep spin old in sync for residual hygiene
            (d_rho_G == d_rho_spin_G_[0]) ? d_rho_spin_G_old_[0]
                                          : d_rho_spin_G_old_[1],
            d_rho_G, static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToDevice, stream_));
    };
    load_one(rho_a0, d_rho_spin_G_[0]);
    load_one(rho_b0, d_rho_spin_G_[1]);
    sync_total_rho();
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(d_rho_G_old_, d_rho_G_,
                        static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_));
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Collinear spin path ──────────────────────────────────────────────────

namespace {

__global__ void pw_sum_spin_rho_kernel(const cufftDoubleComplex* __restrict__ a,
                                       const cufftDoubleComplex* __restrict__ b,
                                       cufftDoubleComplex* __restrict__ tot,
                                       int n_g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    tot[i].x = a[i].x + b[i].x;
    tot[i].y = a[i].y + b[i].y;
}

__global__ void
pw_linear_mix_spin_kernel(cufftDoubleComplex* __restrict__ rho,
                          cufftDoubleComplex* __restrict__ rho_old,
                          const cufftDoubleComplex* __restrict__ rho_out,
                          double beta, int n_g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g)
        return;
    const cufftDoubleComplex o = rho[i];
    rho_old[i] = o;
    const cufftDoubleComplex n = rho_out[i];
    rho[i].x = o.x + beta * (n.x - o.x);
    rho[i].y = o.y + beta * (n.y - o.y);
}

/// ‖Δρ‖₂ residual partials over total density change (G-sphere).
__global__ void
pw_resid_total_stage1(const cufftDoubleComplex* __restrict__ rho,
                      const cufftDoubleComplex* __restrict__ rho_old, int n_g,
                      double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = blockIdx.x * kBlock + threadIdx.x; i < n_g;
         i += kReduceBlocks * kBlock) {
        const double dr = rho[i].x - rho_old[i].x;
        const double di = rho[i].y - rho_old[i].y;
        acc += dr * dr + di * di;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[blockIdx.x] = s[0];
}

__global__ void pw_resid_total_stage2(const double* __restrict__ partials,
                                      double* __restrict__ out) {
    __shared__ double s[kReduceBlocks];
    const int t = threadIdx.x;
    s[t] = partials[t];
    __syncthreads();
    for (int stride = kReduceBlocks / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0)
        *out = sqrt(s[0]);
}

/// Hartree + local-PP energy partials only (spin uses real-space e_xc_int).
__global__ void pw_reduce_hl_kernel(const cufftDoubleComplex* __restrict__ rho,
                                    const cufftDoubleComplex* __restrict__ vh,
                                    const cufftDoubleComplex* __restrict__ vloc,
                                    int n_g, double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc0 = 0.0, acc1 = 0.0;
    for (int i = blockIdx.x * kBlock + threadIdx.x; i < n_g;
         i += kReduceBlocks * kBlock) {
        const cufftDoubleComplex r = rho[i];
        const cufftDoubleComplex h = vh[i];
        const cufftDoubleComplex p = vloc[i];
        acc0 += r.x * h.x + r.y * h.y;
        acc1 += r.x * p.x + r.y * p.y;
    }
    s[threadIdx.x] = acc0;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[0 * kReduceBlocks + blockIdx.x] = s[0];
    __syncthreads();
    s[threadIdx.x] = acc1;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[1 * kReduceBlocks + blockIdx.x] = s[0];
}

__global__ void pw_reduce_hl_finalize(const double* __restrict__ partials,
                                      double scale0, double scale1,
                                      double* __restrict__ out) {
    __shared__ double s[kReduceBlocks];
    const int lane = blockIdx.x; // 0 → e_h, 1 → e_loc
    const int t = threadIdx.x;
    s[t] = partials[lane * kReduceBlocks + t];
    __syncthreads();
    for (int stride = kReduceBlocks / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0)
        out[lane] = s[0] * (lane == 0 ? scale0 : scale1);
}

} // namespace

void DeviceDensity::sync_total_rho() {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::sync_total_rho: not a spin "
            "density object");
    pw_sum_spin_rho_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
        d_rho_spin_G_[0], d_rho_spin_G_[1], d_rho_G_, ng0_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::exchange_correlation_spin() {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::exchange_correlation_spin: "
            "not a spin density object");
    if (xc_ != cpu::scf::XCType::LSDA && xc_ != cpu::scf::XCType::PBE)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::exchange_correlation_spin: "
            "only XCType::LSDA and XCType::PBE are implemented");

    const int nx = fft_.nx(), ny = fft_.ny(), nz = fft_.nz();
    const double dV = omega_ / static_cast<double>(nr_);

    // ρ_σ(G) → ρ_σ(r)
    for (int s = 0; s < 2; ++s) {
        scatter_g_to_halfcomplex(d_rho_spin_G_[s], d_h_, d_k_, d_l_, ng0_,
                                 d_half_, nx, ny, nz, stream_);
        fft_.c2r(d_half_, d_rho_spin_r_[s]);
    }

    if (gga_) {
        // ── Spectral ∇ρ_σ: ∂_d ρ_σ(r) = c2r[i G_d ρ_σ(G)] ──
        // α → d_grad_*; β → d_grad_dn_*.
        double* grad_up[3] = {d_grad_x_r_, d_grad_y_r_, d_grad_z_r_};
        double* grad_dn[3] = {d_grad_dn_x_r_, d_grad_dn_y_r_, d_grad_dn_z_r_};
        const double* gcart[3] = {d_gx_, d_gy_, d_gz_};
        for (int s = 0; s < 2; ++s) {
            double** gcomp = (s == 0) ? grad_up : grad_dn;
            for (int d = 0; d < 3; ++d) {
                pw_mul_ig_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
                    d_rho_spin_G_[s], gcart[d], ng0_, d_sph_);
                ATV_CUDA_CHECK(cudaGetLastError());
                scatter_g_to_halfcomplex(d_sph_, d_h_, d_k_, d_l_, ng0_,
                                         d_half_, nx, ny, nz, stream_);
                fft_.c2r(d_half_, gcomp[d]);
            }
        }

        // Pointwise collinear PBE + flux assembly (grads → fluxes).
        // E_xc uses local ε_xc before the divergence correction (CPU order).
        xc_pbe_spin_grid(d_rho_spin_r_[0], d_rho_spin_r_[1], d_grad_x_r_,
                         d_grad_y_r_, d_grad_z_r_, d_grad_dn_x_r_,
                         d_grad_dn_y_r_, d_grad_dn_z_r_, d_V_xc_spin_r_[0],
                         d_V_xc_spin_r_[1], d_exc_r_, nr_, stream_);
        xc_energy_spin(d_rho_spin_r_[0], d_rho_spin_r_[1], d_exc_r_, nr_, dV,
                       d_scalars_ + 0, d_xc_partials_, stream_);

        // ── v_σ −= ∇·flux_σ ──
        double* flux_up[3] = {d_grad_x_r_, d_grad_y_r_, d_grad_z_r_};
        double* flux_dn[3] = {d_grad_dn_x_r_, d_grad_dn_y_r_, d_grad_dn_z_r_};
        double* div_buf[2] = {d_div_r_, d_div_dn_r_};
        for (int s = 0; s < 2; ++s) {
            double** flux = (s == 0) ? flux_up : flux_dn;
            ATV_CUDA_CHECK(cudaMemsetAsync(d_V_xc_spin_G_[s], 0,
                                           static_cast<size_t>(ng0_)
                                               * sizeof(cufftDoubleComplex),
                                           stream_));
            for (int d = 0; d < 3; ++d) {
                fft_.r2c(flux[d], d_half_);
                gather_halfcomplex_to_g(
                    d_half_, d_h_, d_k_, d_l_, ng0_, d_sph_, nx, ny, nz,
                    1.0 / static_cast<double>(nr_), stream_);
                pw_axpy_ig_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
                    d_sph_, gcart[d], ng0_, d_V_xc_spin_G_[s]);
                ATV_CUDA_CHECK(cudaGetLastError());
            }
            scatter_g_to_halfcomplex(d_V_xc_spin_G_[s], d_h_, d_k_, d_l_, ng0_,
                                     d_half_, nx, ny, nz, stream_);
            fft_.c2r(d_half_, div_buf[s]);
            pw_sub_kernel<<<n_blocks(nr_), kBlock, 0, stream_>>>(
                d_V_xc_spin_r_[s], div_buf[s], nr_);
            ATV_CUDA_CHECK(cudaGetLastError());
        }

        // Double-counting ∫(ρ↑V↑+ρ↓V↓) uses the FULL functional derivative
        // (local + div), matching CPU step_spin after the GGA correction.
        xc_int_spin(d_rho_spin_r_[0], d_rho_spin_r_[1], d_V_xc_spin_r_[0],
                    d_V_xc_spin_r_[1], nr_, dV, d_scalars_ + 2, d_xc_partials_,
                    stream_);
    } else {
        // Pointwise collinear LSDA + E_xc + e_xc_int_spin (real-space).
        xc_lsda_spin_grid(d_rho_spin_r_[0], d_rho_spin_r_[1], d_V_xc_spin_r_[0],
                          d_V_xc_spin_r_[1], d_exc_r_, nr_, stream_);
        xc_energy_spin(d_rho_spin_r_[0], d_rho_spin_r_[1], d_exc_r_, nr_, dV,
                       d_scalars_ + 0, d_xc_partials_, stream_);
        // e_xc_int into scalars[2] temporarily (reduce_spin remaps).
        xc_int_spin(d_rho_spin_r_[0], d_rho_spin_r_[1], d_V_xc_spin_r_[0],
                    d_V_xc_spin_r_[1], nr_, dV, d_scalars_ + 2, d_xc_partials_,
                    stream_);
    }

    // V_xc_σ(r) → V_xc_σ(G)
    for (int s = 0; s < 2; ++s) {
        fft_.r2c(d_V_xc_spin_r_[s], d_half_);
        gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_,
                                d_V_xc_spin_G_[s], nx, ny, nz,
                                1.0 / static_cast<double>(nr_), stream_);
        pw_zero_imag0_kernel<<<1, 1, 0, stream_>>>(d_V_xc_spin_G_[s]);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::assemble_veff_spin(int spin) {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::assemble_veff_spin: not a "
            "spin density object");
    if (spin < 0 || spin > 1)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::assemble_veff_spin: spin must "
            "be 0 (α) or 1 (β)");
    // Copy V_xc_σ into the restricted V_xc slot, then the usual assembly.
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(d_V_xc_G_, d_V_xc_spin_G_[spin],
                        static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_));
    assemble_veff();
}

void DeviceDensity::collect_spin_out(int spin) {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::collect_spin_out: not a spin "
            "density object");
    if (spin < 0 || spin > 1)
        throw std::invalid_argument(
            "cuda::pw::internal::DeviceDensity::collect_spin_out: spin must "
            "be 0 (α) or 1 (β)");
    fft_.r2c(d_rho_r_acc_, d_half_);
    gather_halfcomplex_to_g(d_half_, d_h_, d_k_, d_l_, ng0_,
                            d_rho_spin_out_G_[spin], fft_.nx(), fft_.ny(),
                            fft_.nz(), 1.0 / static_cast<double>(nr_), stream_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void DeviceDensity::mix_spin_linear() {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::mix_spin_linear: not a spin "
            "density object");
    // Snapshot total ρ before the mix for residual (CPU uses pre-mix total as
    // rho_G_old after rewriting rho_G from mixed spins — equivalent: residual
    // on total density change).
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(d_rho_G_old_, d_rho_G_,
                        static_cast<size_t>(ng0_) * sizeof(cufftDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream_));

    for (int s = 0; s < 2; ++s) {
        pw_linear_mix_spin_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
            d_rho_spin_G_[s], d_rho_spin_G_old_[s], d_rho_spin_out_G_[s],
            mix_beta_, ng0_);
    }
    // Mixed total for next Poisson / residual.
    pw_sum_spin_rho_kernel<<<n_blocks(ng0_), kBlock, 0, stream_>>>(
        d_rho_spin_G_[0], d_rho_spin_G_[1], d_rho_G_, ng0_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

DeviceDensity::IterationScalars DeviceDensity::reduce_spin(double vol_bohr) {
    if (!spin_)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceDensity::reduce_spin: not a spin "
            "density object");

    // e_h, e_loc from total ρ (G-space Parseval); e_xc already in scalars[0]
    // and e_xc_int_spin in scalars[2] from exchange_correlation_spin.
    pw_reduce_hl_kernel<<<kReduceBlocks, kBlock, 0, stream_>>>(
        d_rho_G_, d_V_H_G_, d_V_loc_G_, ng0_, d_partials_);
    ATV_CUDA_CHECK(cudaGetLastError());
    // Write e_h → scalars[1], e_loc → scalars[3] (keep [0]=E_xc, [2]=e_xc_int).
    // Use a tiny host staging via existing d_scalars_ slots carefully:
    // finalize writes out[0], out[1] into a temp then we pack.
    double* d_hl = d_partials_ + static_cast<size_t>(2) * kReduceBlocks;
    pw_reduce_hl_finalize<<<2, kReduceBlocks, 0, stream_>>>(
        d_partials_, 0.5 * vol_bohr, vol_bohr, d_hl);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Residual ‖ρ − ρ_old‖₂ on total density.
    pw_resid_total_stage1<<<kReduceBlocks, kBlock, 0, stream_>>>(
        d_rho_G_, d_rho_G_old_, ng0_, d_partials_);
    pw_resid_total_stage2<<<1, kReduceBlocks, 0, stream_>>>(d_partials_,
                                                            d_scalars_ + 4);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Pack e_h / e_loc into the host-facing slots.
    ATV_CUDA_CHECK(cudaMemcpyAsync(d_scalars_ + 1, d_hl, sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream_));
    ATV_CUDA_CHECK(cudaMemcpyAsync(d_scalars_ + 3, d_hl + 1, sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream_));

    double host[5] = {0, 0, 0, 0, 0};
    ATV_CUDA_CHECK(cudaMemcpyAsync(host, d_scalars_, 5 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));

    IterationScalars out;
    out.e_xc = host[0];
    out.e_h = host[1];
    out.e_xc_int = host[2]; // real-space ∫(ρ↑V↑+ρ↓V↓)
    out.e_loc = host[3];
    out.resid = host[4];
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
