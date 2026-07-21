/// @file cuda/pw/internal/device_solver.cu
/// @brief Device-resident plane-wave band solver (block Davidson).
///
/// Mirrors cpu/pw/pw_davidson.cpp step for step: same subspace
/// initialization (shared host helper), same Jacobi subspace eigensolver
/// (shared host helper, eigenpairs sorted ascending), same
/// preconditioner and residual policy. The per-G hot loops run as CUDA
/// kernels; subspace algebra runs as cuBLAS ZGEMM/ZGEMV on column-major
/// matrices whose column j is subspace vector j.

#include "atomiverse/cpu/pw/internal/pw_subspace.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/pw/internal/device_solver.cuh"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

static_assert(std::is_same<Real, double>::value,
              "cuda::pw requires ATV_REAL=double (cuFFT Z2Z path, matching "
              "the CPU PW solver which always runs FFTW in double)");

namespace {

constexpr int kBlock = 256;

inline void check_cufft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuda::pw cuFFT error in ") + what
                                 + ": " + std::to_string(r));
}

inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuda::pw cuBLAS error in ") + what
                                 + ": " + std::to_string(s));
}

inline int n_blocks(int n) { return (n + kBlock - 1) / kBlock; }

// ── Kernels ──────────────────────────────────────────────────────────

__global__ void k_scatter(const cufftDoubleComplex* __restrict__ psi,
                          const int* __restrict__ idx, int ng,
                          cufftDoubleComplex* __restrict__ grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ng)
        grid[idx[i]] = psi[i];
}

__global__ void k_mul_veff(cufftDoubleComplex* __restrict__ grid,
                           const double* __restrict__ veff, int nr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nr) {
        double v = veff[i];
        grid[i].x *= v;
        grid[i].y *= v;
    }
}

/// hpsi[i] = grid[idx[i]]·inv_nr + T(G_i)·psi[i]
__global__ void k_gather_combine(const cufftDoubleComplex* __restrict__ grid,
                                 const int* __restrict__ idx,
                                 const double* __restrict__ tg,
                                 const cufftDoubleComplex* __restrict__ psi,
                                 double inv_nr, int ng,
                                 cufftDoubleComplex* __restrict__ hpsi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ng) {
        cufftDoubleComplex g = grid[idx[i]];
        hpsi[i].x = g.x * inv_nr + tg[i] * psi[i].x;
        hpsi[i].y = g.y * inv_nr + tg[i] * psi[i].y;
    }
}

__global__ void k_accum_density(const cufftDoubleComplex* __restrict__ grid,
                                double scale, int nr,
                                double* __restrict__ rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nr)
        rho[i] += scale * (grid[i].x * grid[i].x + grid[i].y * grid[i].y);
}

__global__ void k_real_to_cplx(const double* __restrict__ rho, int nr,
                               cufftDoubleComplex* __restrict__ grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nr) {
        grid[i].x = rho[i];
        grid[i].y = 0.0;
    }
}

__global__ void k_gather_scaled(const cufftDoubleComplex* __restrict__ grid,
                                const int* __restrict__ idx, double scale,
                                int ng, cufftDoubleComplex* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ng) {
        cufftDoubleComplex g = grid[idx[i]];
        out[i].x = g.x * scale;
        out[i].y = g.y * scale;
    }
}

__global__ void k_scale_cols(cufftDoubleComplex* __restrict__ c,
                             const double* __restrict__ d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i].x *= d[i];
        c[i].y *= d[i];
    }
}

/// R = Hψ − ε·ψ
__global__ void k_residual(const cufftDoubleComplex* __restrict__ hpsi,
                           const cufftDoubleComplex* __restrict__ psi,
                           double eps, int ng,
                           cufftDoubleComplex* __restrict__ r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ng) {
        r[i].x = hpsi[i].x - eps * psi[i].x;
        r[i].y = hpsi[i].y - eps * psi[i].y;
    }
}

/// δ = −R / (T(G) − ε), damped near degeneracy (same policy as CPU).
__global__ void k_precondition(cufftDoubleComplex* __restrict__ r,
                               const double* __restrict__ tg, double eps,
                               int ng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ng) {
        double denom = tg[i] - eps;
        // Clamp keeps the T(G) ~ eps crossing from amplifying noise.
        if (fabs(denom) < 1e-2)
            denom = (denom >= 0 ? 1e-2 : -1e-2);
        double inv = -1.0 / denom;
        r[i].x *= inv;
        r[i].y *= inv;
    }
}

// Real spherical harmonics for one (l, m), matching
// cpu::pw::internal::compute_ylm exactly.
__device__ double ylm_single(int l, int m, double x, double y, double z,
                             double r) {
    constexpr double four_pi = 4.0 * M_PI;
    if (l == 0)
        return 1.0 / sqrt(four_pi);
    if (l == 1) {
        double n = sqrt(3.0 / four_pi);
        double ir = (r > 1e-16) ? 1.0 / r : 0.0;
        if (m == 0)
            return n * y * ir; // p_y
        if (m == 1)
            return n * z * ir; // p_z
        return n * x * ir;     // p_x
    }
    if (l == 2) {
        double ir2 = (r > 1e-16) ? 1.0 / (r * r) : 0.0;
        switch (m) {
        case 0:
            return sqrt(15.0 / four_pi) * x * y * ir2; // d_xy
        case 1:
            return sqrt(15.0 / four_pi) * y * z * ir2; // d_yz
        case 2:
            return sqrt(5.0 / (4.0 * four_pi)) * (3.0 * z * z * ir2 - 1.0);
        case 3:
            return sqrt(15.0 / four_pi) * x * z * ir2; // d_xz
        default:
            return sqrt(15.0 / (4.0 * four_pi)) * (x * x - y * y) * ir2;
        }
    }
    return 0.0;
}

/// β_{j}(G_i) = (4π/√Ω)·(−i)^l·Y_lm(Ĝ)·f_l(G)·exp(−iG·R_a), column-major
/// (ng × ncol). Zero where the radial table is negligible or absent —
/// identical to the CPU `continue` on |f_l| < 1e-30.
__global__ void
k_build_beta(const double* __restrict__ gx, const double* __restrict__ gy,
             const double* __restrict__ gz, const int* __restrict__ col_l,
             const int* __restrict__ col_m, const double* __restrict__ col_pos,
             const int* __restrict__ col_fgoff,
             const int* __restrict__ col_fglen, const double* __restrict__ fg,
             double pref, int ng, int ncol,
             cufftDoubleComplex* __restrict__ beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= ng || j >= ncol)
        return;

    cufftDoubleComplex out{0.0, 0.0};
    double fl = (i < col_fglen[j]) ? fg[col_fgoff[j] + i] : 0.0;
    if (fabs(fl) >= 1e-30) {
        double Gx = gx[i], Gy = gy[i], Gz = gz[i];
        double Gr = sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
        int l = col_l[j];
        double y = ylm_single(l, col_m[j], Gx, Gy, Gz, Gr);
        double phase = Gx * col_pos[3 * j] + Gy * col_pos[3 * j + 1]
                       + Gz * col_pos[3 * j + 2];
        double c = cos(phase), s = sin(phase);
        // (−i)^l for l ≤ 2: 1, −i, −1
        double re_il = (l == 0) ? 1.0 : (l == 1 ? 0.0 : -1.0);
        double im_il = (l == 1) ? -1.0 : 0.0;
        // im_l · exp(−i·phase) = (re_il + i·im_il)(c − i·s)
        double pr = re_il * c + im_il * s;
        double pi = im_il * c - re_il * s;
        double amp = pref * y * fl;
        out.x = amp * pr;
        out.y = amp * pi;
    }
    beta[static_cast<size_t>(j) * ng + i] = out;
}

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

} // namespace

// ── DeviceBandSolver ─────────────────────────────────────────────────

DeviceBandSolver::DeviceBandSolver(int fft_nx, int fft_ny, int fft_nz,
                                   int n_bands)
    : nx_(fft_nx), ny_(fft_ny), nz_(fft_nz), nr_(fft_nx * fft_ny * fft_nz),
      n_bands_(n_bands) {
    // FFTW plans in fft.cpp are (nz, ny, nx) with x fastest; match exactly.
    check_cufft(cufftPlan3d(&fft_plan_, nz_, ny_, nx_, CUFFT_Z2Z),
                "cufftPlan3d");
    check_cublas(cublasCreate(&blas_), "cublasCreate");
    d_grid_ = device_alloc<cufftDoubleComplex>(nr_);
    d_veff_ = device_alloc<double>(nr_);
    d_rho_ = device_alloc<double>(nr_);
}

DeviceBandSolver::~DeviceBandSolver() {
    device_free(d_grid_);
    device_free(d_veff_);
    device_free(d_rho_);
    device_free(d_idx_);
    device_free(d_tg_);
    device_free(d_beta_);
    device_free(d_betad_);
    device_free(d_proj_c_);
    device_free(d_sub_);
    device_free(d_hsub_);
    device_free(d_ritz_);
    device_free(d_hritz_);
    device_free(d_hs_);
    device_free(d_x_);
    device_free(d_work_);
    device_free(d_ovlp_);
    if (fft_plan_)
        cufftDestroy(fft_plan_);
    if (blas_)
        cublasDestroy(blas_);
}

void DeviceBandSolver::set_veff(const Real* v_eff_r) {
    ATV_CUDA_CHECK(cudaMemcpy(d_veff_, v_eff_r, nr_ * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void DeviceBandSolver::set_kpoint(const cpu::pw::PWWorkspace& ws) {
    int ng = ws.n_g;
    const auto& gv = *ws.gvec;
    ng_ = ng;
    omega_ = ws.omega;

    // G → flat c2c FFT-grid index map (same modular fold as pad_to_fft).
    std::vector<int> idx(ng);
    std::vector<double> tg(ng), gx(ng), gy(ng), gz(ng);
    for (int ig = 0; ig < ng; ++ig) {
        int ix = ((gv.h[ig] % nx_) + nx_) % nx_;
        int iy = ((gv.k[ig] % ny_) + ny_) % ny_;
        int iz = ((gv.l[ig] % nz_) + nz_) % nz_;
        idx[ig] = ix + nx_ * (iy + ny_ * iz);
        tg[ig] = static_cast<double>(ws.T_G[ig]);
        // Bloch states live on q = k+G (Y_lm direction, phase).
        gx[ig] = gv.g[ig][0] + gv.kvec[0];
        gy[ig] = gv.g[ig][1] + gv.kvec[1];
        gz[ig] = gv.g[ig][2] + gv.kvec[2];
    }
    device_free(d_idx_);
    device_free(d_tg_);
    d_idx_ = device_alloc<int>(ng);
    d_tg_ = device_alloc<double>(ng);
    ATV_CUDA_CHECK(cudaMemcpy(d_idx_, idx.data(), ng * sizeof(int),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_tg_, tg.data(), ng * sizeof(double),
                              cudaMemcpyHostToDevice));

    // ── KB projector matrix ──
    // Flatten (atom, projector, m) into β columns; skip |D| < 1e-16
    // projectors exactly like the CPU apply_hamiltonian.
    std::vector<int> col_l, col_m, col_fgoff, col_fglen;
    std::vector<double> col_d, col_pos, fg_flat;
    for (size_t ia = 0; ia < ws.kb_proj.size(); ++ia) {
        const auto& pos = ws.atom_pos_bohr[ia];
        for (size_t ip = 0; ip < ws.kb_proj[ia].size(); ++ip) {
            const auto& proj = ws.kb_proj[ia][ip];
            if (std::abs(proj.d) < Real(1e-16))
                continue;
            int fg_off = static_cast<int>(fg_flat.size());
            int fg_len = static_cast<int>(
                std::min<size_t>(proj.f_G.size(), static_cast<size_t>(ng)));
            for (int ig = 0; ig < fg_len; ++ig)
                fg_flat.push_back(static_cast<double>(proj.f_G[ig]));
            int nm = 2 * proj.l + 1;
            for (int m = 0; m < nm; ++m) {
                col_l.push_back(proj.l);
                col_m.push_back(m);
                col_d.push_back(static_cast<double>(proj.d));
                col_pos.push_back(pos[0]);
                col_pos.push_back(pos[1]);
                col_pos.push_back(pos[2]);
                col_fgoff.push_back(fg_off);
                col_fglen.push_back(fg_len);
            }
        }
    }
    n_proj_ = static_cast<int>(col_l.size());

    device_free(d_beta_);
    device_free(d_betad_);
    device_free(d_proj_c_);
    if (n_proj_ > 0) {
        d_beta_ =
            device_alloc<cufftDoubleComplex>(static_cast<size_t>(ng) * n_proj_);
        d_betad_ = device_alloc<double>(n_proj_);
        d_proj_c_ = device_alloc<cufftDoubleComplex>(n_proj_);
        ATV_CUDA_CHECK(cudaMemcpy(d_betad_, col_d.data(),
                                  n_proj_ * sizeof(double),
                                  cudaMemcpyHostToDevice));

        auto* d_gx = device_alloc<double>(ng);
        auto* d_gy = device_alloc<double>(ng);
        auto* d_gz = device_alloc<double>(ng);
        auto* d_cl = device_alloc<int>(n_proj_);
        auto* d_cm = device_alloc<int>(n_proj_);
        auto* d_cp = device_alloc<double>(3 * n_proj_);
        auto* d_co = device_alloc<int>(n_proj_);
        auto* d_cn = device_alloc<int>(n_proj_);
        auto* d_fg = device_alloc<double>(std::max<size_t>(fg_flat.size(), 1));
        ATV_CUDA_CHECK(cudaMemcpy(d_gx, gx.data(), ng * sizeof(double),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_gy, gy.data(), ng * sizeof(double),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_gz, gz.data(), ng * sizeof(double),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_cl, col_l.data(), n_proj_ * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_cm, col_m.data(), n_proj_ * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_cp, col_pos.data(),
                                  3 * n_proj_ * sizeof(double),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_co, col_fgoff.data(), n_proj_ * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ATV_CUDA_CHECK(cudaMemcpy(d_cn, col_fglen.data(), n_proj_ * sizeof(int),
                                  cudaMemcpyHostToDevice));
        if (!fg_flat.empty())
            ATV_CUDA_CHECK(cudaMemcpy(d_fg, fg_flat.data(),
                                      fg_flat.size() * sizeof(double),
                                      cudaMemcpyHostToDevice));

        double pref = 4.0 * M_PI / std::sqrt(static_cast<double>(ws.omega));
        dim3 grid(n_blocks(ng), n_proj_);
        k_build_beta<<<grid, kBlock>>>(d_gx, d_gy, d_gz, d_cl, d_cm, d_cp, d_co,
                                       d_cn, d_fg, pref, ng, n_proj_, d_beta_);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_gx);
        cudaFree(d_gy);
        cudaFree(d_gz);
        cudaFree(d_cl);
        cudaFree(d_cm);
        cudaFree(d_cp);
        cudaFree(d_co);
        cudaFree(d_cn);
        cudaFree(d_fg);
    }
}

void DeviceBandSolver::ensure_subspace_capacity(int ng, int cap) {
    // Per-vector length must cover the largest k-sphere bound so far —
    // k-spheres differ in size, so growth in EITHER dimension reallocates.
    if (cap <= cap_sub_ && ng <= ng_cap_ && d_sub_ != nullptr)
        return;
    device_free(d_sub_);
    device_free(d_hsub_);
    device_free(d_ritz_);
    device_free(d_hritz_);
    device_free(d_hs_);
    device_free(d_x_);
    device_free(d_work_);
    device_free(d_ovlp_);
    cap_sub_ = std::max(cap, cap_sub_);
    ng_cap_ = std::max(ng, ng_cap_);
    cap = cap_sub_;
    size_t ngs = static_cast<size_t>(ng_cap_);
    d_sub_ = device_alloc<cufftDoubleComplex>(ngs * cap);
    d_hsub_ = device_alloc<cufftDoubleComplex>(ngs * cap);
    d_ritz_ = device_alloc<cufftDoubleComplex>(ngs * n_bands_);
    d_hritz_ = device_alloc<cufftDoubleComplex>(ngs * n_bands_);
    d_hs_ = device_alloc<cufftDoubleComplex>(static_cast<size_t>(cap) * cap);
    d_x_ =
        device_alloc<cufftDoubleComplex>(static_cast<size_t>(cap) * n_bands_);
    d_work_ = device_alloc<cufftDoubleComplex>(ngs);
    d_ovlp_ = device_alloc<cufftDoubleComplex>(cap);
}

void DeviceBandSolver::apply_h(const cufftDoubleComplex* psi,
                               cufftDoubleComplex* hpsi) {
    ATV_CUDA_CHECK(cudaMemset(d_grid_, 0, nr_ * sizeof(cufftDoubleComplex)));
    k_scatter<<<n_blocks(ng_), kBlock>>>(psi, d_idx_, ng_, d_grid_);
    check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_INVERSE),
                "Z2Z inverse (H·psi)");
    k_mul_veff<<<n_blocks(nr_), kBlock>>>(d_grid_, d_veff_, nr_);
    check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_FORWARD),
                "Z2Z forward (H·psi)");
    k_gather_combine<<<n_blocks(ng_), kBlock>>>(
        d_grid_, d_idx_, d_tg_, psi, 1.0 / static_cast<double>(nr_), ng_, hpsi);
    ATV_CUDA_CHECK(cudaGetLastError());

    if (n_proj_ > 0) {
        const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
        // c = βᴴ ψ
        check_cublas(
            cublasZgemv(blas_, CUBLAS_OP_C, ng_, n_proj_, &one,
                        reinterpret_cast<const cuDoubleComplex*>(d_beta_), ng_,
                        reinterpret_cast<const cuDoubleComplex*>(psi), 1, &zero,
                        reinterpret_cast<cuDoubleComplex*>(d_proj_c_), 1),
            "Zgemv beta^H psi");
        k_scale_cols<<<n_blocks(n_proj_), kBlock>>>(d_proj_c_, d_betad_,
                                                    n_proj_);
        ATV_CUDA_CHECK(cudaGetLastError());
        // Hψ += β (D∘c)
        check_cublas(
            cublasZgemv(blas_, CUBLAS_OP_N, ng_, n_proj_, &one,
                        reinterpret_cast<const cuDoubleComplex*>(d_beta_), ng_,
                        reinterpret_cast<const cuDoubleComplex*>(d_proj_c_), 1,
                        &one, reinterpret_cast<cuDoubleComplex*>(hpsi), 1),
            "Zgemv beta c");
    }
}

void DeviceBandSolver::solve(cpu::pw::PWWorkspace& ws, Real /*tol*/,
                             int max_iter) {
    int ng = ws.n_g;
    int nb = n_bands_;
    if (ng != ng_)
        throw std::runtime_error(
            "DeviceBandSolver::solve: set_kpoint(ws) must bind the same "
            "G-sphere first");

    ws.eigenvalues.resize({static_cast<std::size_t>(nb)});

    // ── Initial subspace: identical to the CPU solver ──
    std::vector<std::complex<Real>> sub_host;
    int n_sub = cpu::pw::internal::init_subspace(ws, nb, sub_host);

    int max_david = std::min(max_iter, 20);
    int cap = std::min(ng, n_sub + nb * max_david);
    ensure_subspace_capacity(ng, cap);
    ATV_CUDA_CHECK(
        cudaMemcpy(d_sub_, sub_host.data(),
                   static_cast<size_t>(n_sub) * ng * sizeof(cufftDoubleComplex),
                   cudaMemcpyHostToDevice));
    std::vector<bool> h_computed(n_sub, false);

    const cuDoubleComplex c_one{1.0, 0.0}, c_zero{0.0, 0.0}, c_mone{-1.0, 0.0};

    std::vector<std::complex<double>> hs_host;
    std::vector<Real> hs_real, sub_eig, sub_vecs;
    std::vector<std::complex<double>> x_host;

    for (int idavid = 0; idavid < max_david; ++idavid) {
        // H·v for vectors that don't have it yet
        for (int j = 0; j < n_sub; ++j) {
            if (h_computed[j])
                continue;
            apply_h(d_sub_ + static_cast<size_t>(j) * ng,
                    d_hsub_ + static_cast<size_t>(j) * ng);
            h_computed[j] = true;
        }

        // Subspace Hamiltonian Hs = Vᴴ (HV)
        check_cublas(
            cublasZgemm(
                blas_, CUBLAS_OP_C, CUBLAS_OP_N, n_sub, n_sub, ng, &c_one,
                reinterpret_cast<const cuDoubleComplex*>(d_sub_), ng,
                reinterpret_cast<const cuDoubleComplex*>(d_hsub_), ng, &c_zero,
                reinterpret_cast<cuDoubleComplex*>(d_hs_), n_sub),
            "Zgemm V^H HV");
        hs_host.resize(static_cast<size_t>(n_sub) * n_sub);
        ATV_CUDA_CHECK(cudaMemcpy(hs_host.data(), d_hs_,
                                  hs_host.size() * sizeof(cufftDoubleComplex),
                                  cudaMemcpyDeviceToHost));

        // Real symmetric part, row-major — matches the CPU Hs assembly.
        hs_real.assign(static_cast<size_t>(n_sub) * n_sub, Real(0));
        for (int mu = 0; mu < n_sub; ++mu)
            for (int nu = mu; nu < n_sub; ++nu) {
                Real v = static_cast<Real>(
                    hs_host[static_cast<size_t>(nu) * n_sub + mu].real());
                hs_real[mu * n_sub + nu] = hs_real[nu * n_sub + mu] = v;
            }

        sub_eig.assign(n_sub, Real(0));
        sub_vecs.assign(static_cast<size_t>(n_sub) * n_sub, Real(0));
        cpu::pw::internal::jacobi_eigh(n_sub, hs_real.data(), sub_eig.data(),
                                       sub_vecs.data());

        // Ritz rotation X (n_sub × nb, column-major)
        x_host.assign(static_cast<size_t>(n_sub) * nb,
                      std::complex<double>(0, 0));
        for (int b = 0; b < nb; ++b)
            for (int mu = 0; mu < n_sub; ++mu)
                x_host[static_cast<size_t>(b) * n_sub + mu] =
                    std::complex<double>(
                        static_cast<double>(sub_vecs[mu * n_sub + b]), 0.0);
        ATV_CUDA_CHECK(cudaMemcpy(d_x_, x_host.data(),
                                  x_host.size() * sizeof(cufftDoubleComplex),
                                  cudaMemcpyHostToDevice));

        // ritz = V·X ; H·ritz = (HV)·X (H is linear, so no re-apply needed)
        check_cublas(
            cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, ng, nb, n_sub, &c_one,
                        reinterpret_cast<const cuDoubleComplex*>(d_sub_), ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_x_), n_sub,
                        &c_zero, reinterpret_cast<cuDoubleComplex*>(d_ritz_),
                        ng),
            "Zgemm ritz");
        check_cublas(
            cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, ng, nb, n_sub, &c_one,
                        reinterpret_cast<const cuDoubleComplex*>(d_hsub_), ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_x_), n_sub,
                        &c_zero, reinterpret_cast<cuDoubleComplex*>(d_hritz_),
                        ng),
            "Zgemm Hritz");

        for (int b = 0; b < nb; ++b)
            ws.eigenvalues[b] = sub_eig[b];

        // Append preconditioned residuals (same policy/order as CPU).
        if (idavid < max_david - 1 && n_sub < ng - 1) {
            int n_added = 0;
            for (int b = 0; b < nb && n_sub < ng && n_sub < cap_sub_; ++b) {
                double eps = static_cast<double>(sub_eig[b]);
                k_residual<<<n_blocks(ng), kBlock>>>(
                    d_hritz_ + static_cast<size_t>(b) * ng,
                    d_ritz_ + static_cast<size_t>(b) * ng, eps, ng, d_work_);
                ATV_CUDA_CHECK(cudaGetLastError());

                double rnorm = 0;
                check_cublas(
                    cublasDznrm2(
                        blas_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        &rnorm),
                    "Dznrm2 residual");
                // Converged band: appending its residual would only
                // inject rounding noise into the subspace.
                if (rnorm < 1e-6)
                    continue;

                k_precondition<<<n_blocks(ng), kBlock>>>(d_work_, d_tg_, eps,
                                                         ng);
                ATV_CUDA_CHECK(cudaGetLastError());

                // Normalize first (scale-free drop threshold), then TWO
                // Gram-Schmidt passes — same policy as the CPU solver:
                // one pass loses orthogonality at large n_sub and the
                // standard Rayleigh-Ritz drifts below the spectrum.
                double pre_norm = 0;
                check_cublas(
                    cublasDznrm2(
                        blas_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        &pre_norm),
                    "Dznrm2 pre");
                if (pre_norm * pre_norm < 1e-24)
                    continue;
                double inv_pre = 1.0 / pre_norm;
                check_cublas(
                    cublasZdscal(blas_, ng, &inv_pre,
                                 reinterpret_cast<cuDoubleComplex*>(d_work_),
                                 1),
                    "Zdscal pre");
                for (int pass = 0; pass < 2; ++pass) {
                    check_cublas(
                        cublasZgemv(
                            blas_, CUBLAS_OP_C, ng, n_sub, &c_one,
                            reinterpret_cast<const cuDoubleComplex*>(d_sub_),
                            ng,
                            reinterpret_cast<const cuDoubleComplex*>(d_work_),
                            1, &c_zero,
                            reinterpret_cast<cuDoubleComplex*>(d_ovlp_), 1),
                        "Zgemv ovlp");
                    check_cublas(
                        cublasZgemv(
                            blas_, CUBLAS_OP_N, ng, n_sub, &c_mone,
                            reinterpret_cast<const cuDoubleComplex*>(d_sub_),
                            ng,
                            reinterpret_cast<const cuDoubleComplex*>(d_ovlp_),
                            1, &c_one,
                            reinterpret_cast<cuDoubleComplex*>(d_work_), 1),
                        "Zgemv project-out");
                }

                double dnorm = 0;
                check_cublas(
                    cublasDznrm2(
                        blas_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        &dnorm),
                    "Dznrm2 delta");
                // Drop only true rounding-level remainders.
                if (dnorm * dnorm < 1e-24)
                    continue;
                double inv = 1.0 / dnorm;
                check_cublas(
                    cublasZdscal(blas_, ng, &inv,
                                 reinterpret_cast<cuDoubleComplex*>(d_work_),
                                 1),
                    "Zdscal delta");

                ATV_CUDA_CHECK(cudaMemcpy(
                    d_sub_ + static_cast<size_t>(n_sub) * ng, d_work_,
                    ng * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
                h_computed.push_back(false);
                ++n_sub;
                ++n_added;
            }
            if (n_added == 0)
                break; // all bands converged / spanned — done
        }
    }
}

void DeviceBandSolver::apply_h_host(const std::complex<Real>* psi,
                                    std::complex<Real>* hpsi) {
    ensure_subspace_capacity(ng_, 2);
    ATV_CUDA_CHECK(cudaMemcpy(
        d_sub_, psi, static_cast<size_t>(ng_) * sizeof(cufftDoubleComplex),
        cudaMemcpyHostToDevice));
    apply_h(d_sub_, d_hsub_);
    ATV_CUDA_CHECK(cudaMemcpy(
        hpsi, d_hsub_, static_cast<size_t>(ng_) * sizeof(cufftDoubleComplex),
        cudaMemcpyDeviceToHost));
}

void DeviceBandSolver::zero_density() {
    ATV_CUDA_CHECK(cudaMemset(d_rho_, 0, nr_ * sizeof(double)));
}

void DeviceBandSolver::accumulate_density(Real weight, Real omega) {
    double scale =
        static_cast<double>(weight) * 2.0 / static_cast<double>(omega);
    for (int b = 0; b < n_bands_; ++b) {
        ATV_CUDA_CHECK(
            cudaMemset(d_grid_, 0, nr_ * sizeof(cufftDoubleComplex)));
        k_scatter<<<n_blocks(ng_), kBlock>>>(
            d_ritz_ + static_cast<size_t>(b) * ng_, d_idx_, ng_, d_grid_);
        check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_INVERSE),
                    "Z2Z inverse (density)");
        k_accum_density<<<n_blocks(nr_), kBlock>>>(d_grid_, scale, nr_, d_rho_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void DeviceBandSolver::fetch_density_g(
    const cpu::pw::internal::GVectorList& gv0, std::complex<Real>* rho_g) {
    int ng0 = gv0.count();
    k_real_to_cplx<<<n_blocks(nr_), kBlock>>>(d_rho_, nr_, d_grid_);
    ATV_CUDA_CHECK(cudaGetLastError());
    check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_FORWARD),
                "Z2Z forward (density)");

    std::vector<int> idx(ng0);
    for (int ig = 0; ig < ng0; ++ig) {
        int ix = ((gv0.h[ig] % nx_) + nx_) % nx_;
        int iy = ((gv0.k[ig] % ny_) + ny_) % ny_;
        int iz = ((gv0.l[ig] % nz_) + nz_) % nz_;
        idx[ig] = ix + nx_ * (iy + ny_ * iz);
    }
    auto* d_idx0 = device_alloc<int>(ng0);
    auto* d_out = device_alloc<cufftDoubleComplex>(ng0);
    ATV_CUDA_CHECK(cudaMemcpy(d_idx0, idx.data(), ng0 * sizeof(int),
                              cudaMemcpyHostToDevice));
    k_gather_scaled<<<n_blocks(ng0), kBlock>>>(
        d_grid_, d_idx0, 1.0 / static_cast<double>(nr_), ng0, d_out);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpy(rho_g, d_out, ng0 * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_idx0);
    cudaFree(d_out);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
