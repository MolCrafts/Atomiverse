/// @file cuda/planewave/internal/device_solver.cu
/// @brief Device-resident plane-wave band solver (block Davidson).
///
/// Follows cpu/planewave/internal/davidson.cpp step for step: same subspace
/// initialization (shared host helper), same preconditioner and residual
/// policy, eigenpairs ascending. The per-G hot loops run as CUDA kernels;
/// subspace algebra runs as cuBLAS ZGEMM/ZGEMV on column-major matrices whose
/// column j is subspace vector j; the dense subspace eigenproblem runs as
/// cuSOLVER `Zheevd` in place on that same column-major buffer.
///
/// The eigensolver is where this stops being a transcription of the CPU path:
/// Jacobi there, divide-and-conquer here. See the parity note in the header —
/// eigenvalues match to roundoff, eigenvectors need not.

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
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

static_assert(std::is_same<AccReal, double>::value,
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

inline void check_cusolver(cusolverStatus_t st, const char* what) {
    if (st != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuda::pw cuSOLVER error in ")
                                 + what + ": " + std::to_string(st));
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

__global__ void k_scale_cols(cufftDoubleComplex* __restrict__ c,
                             const double* __restrict__ d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i].x *= d[i];
        c[i].y *= d[i];
    }
}

/// inv = 1/norm, or 0 when the vector is at rounding level.
///
/// The zero is what makes the unconditional pipeline safe: a band that will
/// be rejected still gets scaled, and scaling by 0 leaves a clean zero vector
/// instead of inf/NaN propagating into the Gram-Schmidt that follows.
__global__ void k_safe_reciprocal(const double* __restrict__ norm,
                                  double* __restrict__ inv) {
    *inv = (*norm > 0.0) ? 1.0 / *norm : 0.0;
}

/// accept = the three host branches this replaces, evaluated on device.
///
/// Byte-for-byte the CPU predicates: reject a converged band (‖R‖ < 1e-6),
/// and reject rounding-level remainders before and after Gram-Schmidt
/// (‖·‖² < 1e-24).
__global__ void k_accept_flag(const double* __restrict__ nrm,
                              int* __restrict__ accept) {
    const double r = nrm[0], p = nrm[1], d = nrm[2];
    *accept = (r >= 1e-6 && p * p >= 1e-24 && d * d >= 1e-24) ? 1 : 0;
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

/// work[i] = scale[i] * psi[i]  (real scale × complex).
__global__ void k_mul_real_scale(const cufftDoubleComplex* __restrict__ psi,
                                 const double* __restrict__ scale, int n,
                                 cufftDoubleComplex* __restrict__ work) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = scale[i];
        work[i].x = s * psi[i].x;
        work[i].y = s * psi[i].y;
    }
}

/// F[ia] += 2 D pref Im(c* s_α) for each projector column (CPU twin).
__global__ void k_nl_force_from_proj(const cufftDoubleComplex* __restrict__ c,
                                     const cufftDoubleComplex* __restrict__ sx,
                                     const cufftDoubleComplex* __restrict__ sy,
                                     const cufftDoubleComplex* __restrict__ sz,
                                     const double* __restrict__ D,
                                     const int* __restrict__ atom,
                                     double pref, int ncol,
                                     double* __restrict__ forces) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ncol)
        return;
    const double Dj = D[j];
    if (fabs(Dj) < 1e-16)
        return;
    const int ia = atom[j];
    if (ia < 0)
        return;
    // Im(conj(c) * s) = c.x * s.y - c.y * s.x
    const cufftDoubleComplex cj = c[j];
    const double scale = 2.0 * Dj * pref;
    const cufftDoubleComplex sjx = sx[j], sjy = sy[j], sjz = sz[j];
    atomicAdd(forces + 3 * ia + 0,
              scale * (cj.x * sjx.y - cj.y * sjx.x));
    atomicAdd(forces + 3 * ia + 1,
              scale * (cj.x * sjy.y - cj.y * sjy.x));
    atomicAdd(forces + 3 * ia + 2,
              scale * (cj.x * sjz.y - cj.y * sjz.x));
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
                                   int n_bands, cudaStream_t stream)
    : nx_(fft_nx), ny_(fft_ny), nz_(fft_nz), nr_(fft_nx * fft_ny * fft_nz),
      n_bands_(n_bands), stream_(stream) {
    if (!stream_) {
        ATV_CUDA_CHECK(cudaStreamCreate(&stream_));
        owns_stream_ = true;
    }
    // FFTW plans in fft.cpp are (nz, ny, nx) with x fastest; match exactly.
    check_cufft(cufftPlan3d(&fft_plan_, nz_, ny_, nx_, CUFFT_Z2Z),
                "cufftPlan3d");
    check_cufft(cufftSetStream(fft_plan_, stream_), "cufftSetStream");
    check_cublas(cublasCreate(&blas_), "cublasCreate");
    check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream");
    check_cublas(cublasCreate(&blas_dev_), "cublasCreate device-pointer");
    check_cublas(
        cublasSetPointerMode(blas_dev_, CUBLAS_POINTER_MODE_DEVICE),
        "cublasSetPointerMode");
    check_cublas(cublasSetStream(blas_dev_, stream_),
                 "cublasSetStream device-pointer");
    check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
    check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
    d_grid_ = device_alloc<cufftDoubleComplex>(nr_);
    d_nrm_ = device_alloc<double>(3);
    d_inv_ = device_alloc<double>(2);
    d_accept_ = device_alloc<int>(1);
}

void DeviceBandSolver::free_tables() {
    device_free(d_idx_all_);
    device_free(d_tg_all_);
    device_free(d_qx_all_);
    device_free(d_qy_all_);
    device_free(d_qz_all_);
    device_free(d_cl_all_);
    device_free(d_cm_all_);
    device_free(d_co_all_);
    device_free(d_cn_all_);
    device_free(d_cia_all_);
    device_free(d_cd_all_);
    device_free(d_cp_all_);
    device_free(d_fg_all_);
    device_free(d_sub0_all_);
    device_free(d_beta_);
    device_free(d_proj_c_);
    // Views into what was just freed.
    d_idx_ = nullptr;
    d_tg_ = nullptr;
    d_betad_ = nullptr;
    d_sub0_ = nullptr;
    ng_ = 0;
    n_proj_ = 0;
    n_sub0_ = 0;
    bound_k_ = -1;
}

DeviceBandSolver::~DeviceBandSolver() {
    if (stream_)
        cudaStreamSynchronize(stream_);
    device_free(d_grid_);
    free_tables();
    device_free(d_sub_);
    device_free(d_hsub_);
    device_free(d_ritz_);
    device_free(d_hritz_);
    device_free(d_hs_);
    device_free(d_eig_);
    device_free(d_evd_work_);
    device_free(d_evd_info_);
    device_free(d_work_);
    device_free(d_ovlp_);
    device_free(d_nrm_);
    device_free(d_inv_);
    device_free(d_accept_);
    if (fft_plan_)
        cufftDestroy(fft_plan_);
    if (blas_)
        cublasDestroy(blas_);
    if (blas_dev_)
        cublasDestroy(blas_dev_);
    if (solver_)
        cusolverDnDestroy(solver_);
    if (owns_stream_ && stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void DeviceBandSolver::bind_grids(double* d_rho_accum, const double* d_veff_r) {
    rho_accum_ = d_rho_accum;
    veff_r_ = d_veff_r;
}

namespace {
/// Shared guard for every entry point that touches the borrowed grids.
inline void require_bound(const void* rho_accum, const void* veff_r) {
    if (!rho_accum || !veff_r)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceBandSolver: bind_grids() must run "
            "before solve()/accumulate_density()");
}
} // namespace

void DeviceBandSolver::build_kpoints(cpu::pw::internal::PWProblem& prob) {
    const int n_k = static_cast<int>(prob.gv_k.size());
    if (n_k <= 0)
        throw std::runtime_error(
            "DeviceBandSolver::build_kpoints: prob.gv_k is empty — "
            "setup_pw_problem() must run first");

    auto& ws = prob.ws;
    auto* saved_gvec = ws.gvec;
    const int saved_ng = ws.n_g;

    n_kpts_ = n_k;
    pref_ = 4.0 * M_PI / std::sqrt(static_cast<double>(ws.omega));

    k_ng_.assign(n_k, 0);
    k_ncol_.assign(n_k, 0);
    k_nsub_.assign(n_k, 0);
    k_goff_.assign(n_k, 0);
    k_coff_.assign(n_k, 0);
    k_fgoff_.assign(n_k, 0);
    k_soff_.assign(n_k, 0);

    // Host staging, filled once for every k-point. These are the exact
    // arrays the old per-iteration set_kpoint() rebuilt from scratch.
    std::vector<int> h_idx, h_cl, h_cm, h_co, h_cn, h_cia;
    std::vector<double> h_tg, h_qx, h_qy, h_qz, h_cd, h_cp, h_fg;
    std::vector<std::complex<AccReal>> h_sub0;

    int max_ng = 0, max_ncol = 0;

    for (int ik = 0; ik < n_k; ++ik) {
        // Bind this k-sphere exactly as the SCF loop used to, so the host
        // helpers (init_subspace) see what they saw before.
        const int ng = prob.gv_k[ik].count();
        ws.gvec = &prob.gv_k[ik];
        ws.n_g = ng;
        if (!prob.kb_proj_k.empty())
            ws.kb_proj = prob.kb_proj_k[ik];
        if (!prob.atom_wfc_k.empty())
            ws.atom_wfc = prob.atom_wfc_k[ik];
        ws.T_G.resize({static_cast<std::size_t>(ng)});
        for (int ig = 0; ig < ng; ++ig)
            ws.T_G[ig] = prob.gv_k[ik].ekin[ig];

        const auto& gv = *ws.gvec;
        k_ng_[ik] = ng;
        k_goff_[ik] = h_idx.size();
        max_ng = std::max(max_ng, ng);

        // G → flat c2c FFT-grid index map (same modular fold as pad_to_fft).
        for (int ig = 0; ig < ng; ++ig) {
            const int ix = ((gv.h[ig] % nx_) + nx_) % nx_;
            const int iy = ((gv.k[ig] % ny_) + ny_) % ny_;
            const int iz = ((gv.l[ig] % nz_) + nz_) % nz_;
            h_idx.push_back(ix + nx_ * (iy + ny_ * iz));
            h_tg.push_back(static_cast<double>(ws.T_G[ig]));
            // Bloch states live on q = k+G (Y_lm direction, phase).
            h_qx.push_back(gv.g[ig][0] + gv.kvec[0]);
            h_qy.push_back(gv.g[ig][1] + gv.kvec[1]);
            h_qz.push_back(gv.g[ig][2] + gv.kvec[2]);
        }

        // ── KB projector columns ──
        // Flatten (atom, projector, m) into β columns; skip |D| < 1e-16
        // projectors exactly like the CPU apply_hamiltonian. f_G offsets are
        // relative to this k's block, so the kernel receives a base pointer.
        k_coff_[ik] = h_cl.size();
        k_fgoff_[ik] = h_fg.size();
        const std::size_t fg_base = h_fg.size();
        for (std::size_t ia = 0; ia < ws.kb_proj.size(); ++ia) {
            const auto& pos = ws.atom_pos_bohr[ia];
            for (std::size_t ip = 0; ip < ws.kb_proj[ia].size(); ++ip) {
                const auto& proj = ws.kb_proj[ia][ip];
                if (std::abs(proj.d) < AccReal(1e-16))
                    continue;
                const int fg_off = static_cast<int>(h_fg.size() - fg_base);
                const int fg_len = static_cast<int>(std::min<std::size_t>(
                    proj.f_G.size(), static_cast<std::size_t>(ng)));
                for (int ig = 0; ig < fg_len; ++ig)
                    h_fg.push_back(static_cast<double>(proj.f_G[ig]));
                const int nm = 2 * proj.l + 1;
                for (int m = 0; m < nm; ++m) {
                    h_cl.push_back(proj.l);
                    h_cm.push_back(m);
                    h_cia.push_back(static_cast<int>(ia));
                    h_cd.push_back(static_cast<double>(proj.d));
                    h_cp.push_back(pos[0]);
                    h_cp.push_back(pos[1]);
                    h_cp.push_back(pos[2]);
                    h_co.push_back(fg_off);
                    h_cn.push_back(fg_len);
                }
            }
        }
        k_ncol_[ik] = static_cast<int>(h_cl.size() - k_coff_[ik]);
        max_ncol = std::max(max_ncol, k_ncol_[ik]);

        // ── Initial Davidson subspace ──
        // A pure function of T_G / atom_wfc / gvec / atom_pos_bohr, i.e. of
        // the geometry: identical in every SCF iteration, so it is built and
        // uploaded once here instead of once per k per iteration.
        std::vector<std::complex<AccReal>> sub_host;
        k_nsub_[ik] =
            cpu::pw::internal::init_subspace(ws, n_bands_, sub_host);
        k_soff_[ik] = h_sub0.size();
        h_sub0.insert(h_sub0.end(), sub_host.begin(), sub_host.end());
    }

    ws.gvec = saved_gvec;
    ws.n_g = saved_ng;

    // Fail with the figure rather than a bare cudaMalloc error. The subspace
    // block dominates; k-batching is the answer if a cell ever exceeds this.
    const std::size_t bytes =
        h_idx.size() * sizeof(int) + h_tg.size() * sizeof(double)
        + (h_qx.size() + h_qy.size() + h_qz.size()) * sizeof(double)
        + (h_cl.size() + h_cm.size() + h_co.size() + h_cn.size()) * sizeof(int)
        + (h_cd.size() + h_cp.size() + h_fg.size()) * sizeof(double)
        + h_sub0.size() * sizeof(cufftDoubleComplex)
        + static_cast<std::size_t>(max_ng) * max_ncol
              * sizeof(cufftDoubleComplex);
    std::size_t free_b = 0, total_b = 0;
    ATV_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (bytes >= free_b)
        throw std::runtime_error(
            "DeviceBandSolver::build_kpoints: the per-k tables for "
            + std::to_string(n_k) + " k-points need "
            + std::to_string(bytes / (1024 * 1024)) + " MiB but only "
            + std::to_string(free_b / (1024 * 1024)) + " of "
            + std::to_string(total_b / (1024 * 1024))
            + " MiB is free on this device. Reduce k_mesh or ecut_ry; "
              "batching the k-tables is not implemented. (The Davidson "
              "subspace is allocated after this and is not counted here.)");

    auto upload_i = [this](const std::vector<int>& h, int*& d) {
        if (h.empty())
            return;
        d = device_alloc<int>(h.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream_));
    };
    auto upload_d = [this](const std::vector<double>& h, double*& d) {
        if (h.empty())
            return;
        d = device_alloc<double>(h.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
    };

    free_tables();
    upload_i(h_idx, d_idx_all_);
    upload_d(h_tg, d_tg_all_);
    upload_d(h_qx, d_qx_all_);
    upload_d(h_qy, d_qy_all_);
    upload_d(h_qz, d_qz_all_);
    upload_i(h_cl, d_cl_all_);
    upload_i(h_cm, d_cm_all_);
    upload_i(h_co, d_co_all_);
    upload_i(h_cn, d_cn_all_);
    upload_i(h_cia, d_cia_all_);
    upload_d(h_cd, d_cd_all_);
    upload_d(h_cp, d_cp_all_);
    upload_d(h_fg, d_fg_all_);

    if (!h_sub0.empty()) {
        // std::complex<double> and cufftDoubleComplex are both two adjacent
        // doubles; the CPU path already relies on this layout equivalence.
        static_assert(sizeof(std::complex<AccReal>)
                          == sizeof(cufftDoubleComplex),
                      "initial subspace upload assumes layout equivalence");
        d_sub0_all_ = device_alloc<cufftDoubleComplex>(h_sub0.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            d_sub0_all_, h_sub0.data(),
            h_sub0.size() * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice,
            stream_));
    }
    // Host vectors go out of scope after this call — fence before return.
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));

    // β and its ⟨β|ψ⟩ workspace are rebuilt per bind_kpoint, so one
    // allocation at the largest k-point serves every k.
    if (max_ncol > 0) {
        d_beta_ = device_alloc<cufftDoubleComplex>(
            static_cast<std::size_t>(max_ng) * max_ncol);
        d_proj_c_ = device_alloc<cufftDoubleComplex>(max_ncol);
    }
    bound_k_ = -1;
}

void DeviceBandSolver::bind_kpoint(int ik) {
    if (ik < 0 || ik >= n_kpts_)
        throw std::runtime_error(
            "DeviceBandSolver::bind_kpoint: k-point index "
            + std::to_string(ik) + " out of range for "
            + std::to_string(n_kpts_)
            + " built k-points (build_kpoints() must run first)");

    ng_ = k_ng_[ik];
    n_proj_ = k_ncol_[ik];
    n_sub0_ = k_nsub_[ik];
    bound_k_ = ik;

    const std::size_t go = k_goff_[ik], co = k_coff_[ik];
    d_idx_ = d_idx_all_ + go;
    d_tg_ = d_tg_all_ + go;
    d_betad_ = d_cd_all_ ? d_cd_all_ + co : nullptr;
    d_sub0_ = d_sub0_all_ ? d_sub0_all_ + k_soff_[ik] : nullptr;

    if (n_proj_ > 0) {
        // Inputs are already resident: one launch, no alloc, no H2D, and no
        // sync (the following kernels are ordered behind it on stream_).
        const dim3 grid(n_blocks(ng_), n_proj_);
        k_build_beta<<<grid, kBlock, 0, stream_>>>(
            d_qx_all_ + go, d_qy_all_ + go, d_qz_all_ + go, d_cl_all_ + co,
            d_cm_all_ + co, d_cp_all_ + 3 * co, d_co_all_ + co,
            d_cn_all_ + co, d_fg_all_ + k_fgoff_[ik], pref_, ng_, n_proj_,
            d_beta_);
        ATV_CUDA_CHECK(cudaGetLastError());
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
    device_free(d_eig_);
    device_free(d_evd_work_);
    device_free(d_evd_info_);
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
    d_eig_ = device_alloc<double>(cap);
    d_evd_info_ = device_alloc<int>(1);
    d_work_ = device_alloc<cufftDoubleComplex>(ngs);
    d_ovlp_ = device_alloc<cufftDoubleComplex>(cap);

    // Zheevd scratch, sized for the largest subspace this capacity allows.
    // The requirement grows with n, so a buffer for `cap` covers every
    // n_sub <= cap the Davidson loop reaches; solve() re-queries anyway and
    // grows if a cuSOLVER version ever breaks that.
    evd_lwork_ = 0;
    check_cusolver(cusolverDnZheevd_bufferSize(
                       solver_, CUSOLVER_EIG_MODE_VECTOR,
                       CUBLAS_FILL_MODE_UPPER, cap,
                       reinterpret_cast<const cuDoubleComplex*>(d_hs_), cap,
                       d_eig_, &evd_lwork_),
                   "Zheevd_bufferSize");
    d_evd_work_ = device_alloc<cufftDoubleComplex>(
        static_cast<size_t>(std::max(evd_lwork_, 1)));
}

void DeviceBandSolver::apply_h(const cufftDoubleComplex* psi,
                               cufftDoubleComplex* hpsi) {
    ATV_CUDA_CHECK(cudaMemsetAsync(d_grid_, 0,
                                   nr_ * sizeof(cufftDoubleComplex), stream_));
    k_scatter<<<n_blocks(ng_), kBlock, 0, stream_>>>(psi, d_idx_, ng_, d_grid_);
    check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_INVERSE),
                "Z2Z inverse (H·psi)");
    k_mul_veff<<<n_blocks(nr_), kBlock, 0, stream_>>>(d_grid_, veff_r_, nr_);
    check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_FORWARD),
                "Z2Z forward (H·psi)");
    k_gather_combine<<<n_blocks(ng_), kBlock, 0, stream_>>>(
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
        k_scale_cols<<<n_blocks(n_proj_), kBlock, 0, stream_>>>(
            d_proj_c_, d_betad_, n_proj_);
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

void DeviceBandSolver::solve(cpu::pw::internal::PWWorkspace& ws, int max_iter) {
    require_bound(rho_accum_, veff_r_);
    if (bound_k_ < 0)
        throw std::runtime_error(
            "DeviceBandSolver::solve: bind_kpoint() must select a k-point "
            "built by build_kpoints() first");
    int ng = ng_;
    int nb = n_bands_;

    ws.eigenvalues.resize({static_cast<std::size_t>(nb)});

    // ── Initial subspace ──
    // Identical to the CPU solver, but built once per k-point by
    // build_kpoints() rather than per SCF iteration: init_subspace() is a
    // pure function of the geometry, so this copy is device→device and no
    // O(n_G) array crosses PCIe here.
    int n_sub = n_sub0_;
    int max_david = std::min(max_iter, 20);
    int cap = std::min(ng, n_sub + nb * max_david);
    ensure_subspace_capacity(ng, cap);
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_sub_, d_sub0_,
        static_cast<size_t>(n_sub) * ng * sizeof(cufftDoubleComplex),
        cudaMemcpyDeviceToDevice, stream_));
    std::vector<bool> h_computed(n_sub, false);

    const cuDoubleComplex c_one{1.0, 0.0}, c_zero{0.0, 0.0}, c_mone{-1.0, 0.0};

    // Only the nb lowest eigenvalues come back to the host: they are kernel
    // scalar arguments (residual shift, preconditioner) and the caller's
    // eigenvalue output. Everything else stays on device.
    std::vector<double> sub_eig(static_cast<size_t>(nb), 0.0);

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
        // ── Subspace eigenproblem, in place on device ──
        // cuBLAS wrote Hs column-major with ld = n_sub, so element (mu, nu) =
        // ⟨v_mu|H|v_nu⟩ sits at d_hs_[nu*n_sub + mu]. CUBLAS_FILL_MODE_UPPER
        // makes Zheevd read exactly the mu <= nu triangle the CPU assembly
        // read, and LAPACK ignores the imaginary part of the diagonal just as
        // the CPU code forced it to zero — same matrix, no transpose or
        // conjugation to get wrong.
        //
        // Zheevd OVERWRITES d_hs_ with the eigenvectors, column-major, column
        // b being eigenvector b, ascending by eigenvalue. Its first nb columns
        // are therefore the Ritz rotation itself: the ZGEMMs below consume
        // d_hs_ directly and no rotation matrix is ever formed on the host.
        int lwork = 0;
        check_cusolver(cusolverDnZheevd_bufferSize(
                           solver_, CUSOLVER_EIG_MODE_VECTOR,
                           CUBLAS_FILL_MODE_UPPER, n_sub,
                           reinterpret_cast<const cuDoubleComplex*>(d_hs_),
                           n_sub, d_eig_, &lwork),
                       "Zheevd_bufferSize");
        if (lwork > evd_lwork_) {
            // Not expected: the capacity-time query used cap >= n_sub. Grow
            // rather than under-allocate if a cuSOLVER version disagrees.
            device_free(d_evd_work_);
            evd_lwork_ = lwork;
            d_evd_work_ = device_alloc<cufftDoubleComplex>(
                static_cast<size_t>(evd_lwork_));
        }
        check_cusolver(
            cusolverDnZheevd(solver_, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_UPPER, n_sub,
                             reinterpret_cast<cuDoubleComplex*>(d_hs_), n_sub,
                             d_eig_,
                             reinterpret_cast<cuDoubleComplex*>(d_evd_work_),
                             evd_lwork_, d_evd_info_),
            "Zheevd");

        // One 4-byte + nb*8-byte readback per round. devInfo must be checked —
        // a non-zero value means the eigenvectors are garbage, and silently
        // rotating the subspace with them would poison the whole SCF.
        // Stream-scoped: fence only stream_, not the default stream.
        int evd_info = 0;
        ATV_CUDA_CHECK(cudaMemcpyAsync(&evd_info, d_evd_info_, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_));
        ATV_CUDA_CHECK(cudaMemcpyAsync(sub_eig.data(), d_eig_,
                                       static_cast<size_t>(nb) * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (evd_info != 0)
            throw std::runtime_error(
                "cuda::pw DeviceBandSolver::solve: cusolverDnZheevd failed to "
                "converge the "
                + std::to_string(n_sub) + "x" + std::to_string(n_sub)
                + " subspace (devInfo=" + std::to_string(evd_info) + ")");

        // ritz = V·X ; H·ritz = (HV)·X (H is linear, so no re-apply needed)
        check_cublas(
            cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, ng, nb, n_sub, &c_one,
                        reinterpret_cast<const cuDoubleComplex*>(d_sub_), ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_hs_), n_sub,
                        &c_zero, reinterpret_cast<cuDoubleComplex*>(d_ritz_),
                        ng),
            "Zgemm ritz");
        check_cublas(
            cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, ng, nb, n_sub, &c_one,
                        reinterpret_cast<const cuDoubleComplex*>(d_hsub_), ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_hs_), n_sub,
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
                k_residual<<<n_blocks(ng), kBlock, 0, stream_>>>(
                    d_hritz_ + static_cast<size_t>(b) * ng,
                    d_ritz_ + static_cast<size_t>(b) * ng, eps, ng, d_work_);
                ATV_CUDA_CHECK(cudaGetLastError());

                // ── Norms on device, one readback per band ──
                // The CPU solver branches three times here (converged band,
                // rounding-level before and after Gram-Schmidt), and each
                // branch used to cost a host-pointer cublasDznrm2, i.e. a
                // synchronization. All three stages now run unconditionally
                // with device-pointer norms and device-computed reciprocal
                // scales, and the three predicates are folded into one accept
                // flag — the loop's only stream-scoped sync.
                //
                // Predicates and arithmetic are unchanged, so an accepted
                // band is bit-identical to before. A rejected band pays two
                // extra ZGEMVs it would previously have skipped; that is the
                // trade for dropping two syncs per band. k_safe_reciprocal
                // keeps the rejected path finite.
                check_cublas(
                    cublasDznrm2(
                        blas_dev_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        d_nrm_ + 0),
                    "Dznrm2 residual");

                k_precondition<<<n_blocks(ng), kBlock, 0, stream_>>>(
                    d_work_, d_tg_, eps, ng);
                ATV_CUDA_CHECK(cudaGetLastError());

                // Normalize first (scale-free drop threshold), then TWO
                // Gram-Schmidt passes — same policy as the CPU solver:
                // one pass loses orthogonality at large n_sub and the
                // standard Rayleigh-Ritz drifts below the spectrum.
                check_cublas(
                    cublasDznrm2(
                        blas_dev_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        d_nrm_ + 1),
                    "Dznrm2 pre");
                k_safe_reciprocal<<<1, 1, 0, stream_>>>(d_nrm_ + 1, d_inv_ + 0);
                ATV_CUDA_CHECK(cudaGetLastError());
                check_cublas(
                    cublasZdscal(blas_dev_, ng, d_inv_ + 0,
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

                check_cublas(
                    cublasDznrm2(
                        blas_dev_, ng,
                        reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                        d_nrm_ + 2),
                    "Dznrm2 delta");
                k_safe_reciprocal<<<1, 1, 0, stream_>>>(d_nrm_ + 2, d_inv_ + 1);
                ATV_CUDA_CHECK(cudaGetLastError());
                check_cublas(
                    cublasZdscal(blas_dev_, ng, d_inv_ + 1,
                                 reinterpret_cast<cuDoubleComplex*>(d_work_),
                                 1),
                    "Zdscal delta");

                k_accept_flag<<<1, 1, 0, stream_>>>(d_nrm_, d_accept_);
                ATV_CUDA_CHECK(cudaGetLastError());
                int accept = 0;
                ATV_CUDA_CHECK(cudaMemcpyAsync(&accept, d_accept_, sizeof(int),
                                               cudaMemcpyDeviceToHost,
                                               stream_));
                ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
                if (!accept)
                    continue;

                ATV_CUDA_CHECK(cudaMemcpyAsync(
                    d_sub_ + static_cast<size_t>(n_sub) * ng, d_work_,
                    ng * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice,
                    stream_));
                h_computed.push_back(false);
                ++n_sub;
                ++n_added;
            }
            if (n_added == 0)
                break; // all bands converged / spanned — done
        }
    }
}

void DeviceBandSolver::apply_h_host(const std::complex<AccReal>* psi,
                                    std::complex<AccReal>* hpsi) {
    require_bound(rho_accum_, veff_r_);
    ensure_subspace_capacity(ng_, 2);
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_sub_, psi, static_cast<size_t>(ng_) * sizeof(cufftDoubleComplex),
        cudaMemcpyHostToDevice, stream_));
    apply_h(d_sub_, d_hsub_);
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        hpsi, d_hsub_, static_cast<size_t>(ng_) * sizeof(cufftDoubleComplex),
        cudaMemcpyDeviceToHost, stream_));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void DeviceBandSolver::accumulate_density(AccReal weight, AccReal omega) {
    require_bound(rho_accum_, veff_r_);
    // require_bound covers the borrowed grids only. This method also reads
    // d_ritz_ / d_idx_, which stay null until bind_kpoint() + solve() have
    // run — dereferencing them would be a silent out-of-bounds read rather
    // than a named error.
    if (!d_ritz_ || !d_idx_ || ng_ <= 0)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceBandSolver::accumulate_density: "
            "bind_kpoint() and solve() must run first (no Ritz vectors "
            "bound)");
    double scale =
        static_cast<double>(weight) * 2.0 / static_cast<double>(omega);
    for (int b = 0; b < n_bands_; ++b) {
        ATV_CUDA_CHECK(cudaMemsetAsync(
            d_grid_, 0, nr_ * sizeof(cufftDoubleComplex), stream_));
        k_scatter<<<n_blocks(ng_), kBlock, 0, stream_>>>(
            d_ritz_ + static_cast<size_t>(b) * ng_, d_idx_, ng_, d_grid_);
        check_cufft(cufftExecZ2Z(fft_plan_, d_grid_, d_grid_, CUFFT_INVERSE),
                    "Z2Z inverse (density)");
        k_accum_density<<<n_blocks(nr_), kBlock, 0, stream_>>>(
            d_grid_, scale, nr_, rho_accum_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void DeviceBandSolver::accumulate_nl_forces(double weight, int n_atoms,
                                            double* d_forces) {
    if (n_proj_ <= 0 || !d_beta_ || !d_betad_ || !d_cia_all_)
        return; // pure-local potential (e.g. GTH-PBE-q1 H): F_nl = 0
    if (!d_ritz_ || bound_k_ < 0 || ng_ <= 0)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceBandSolver::accumulate_nl_forces: "
            "bind_kpoint() and solve() must run first");
    if (n_atoms <= 0 || !d_forces)
        throw std::runtime_error(
            "cuda::pw::internal::DeviceBandSolver::accumulate_nl_forces: "
            "n_atoms must be > 0 and d_forces non-null");

    // Ensure d_work_ is live (solve always allocates it via
    // ensure_subspace_capacity; force a capacity if a unit test skips solve).
    ensure_subspace_capacity(ng_, std::max(n_bands_ + 2, 2));

    const std::size_t go = k_goff_[bound_k_];
    const std::size_t co = k_coff_[bound_k_];
    const double* qx = d_qx_all_ + go;
    const double* qy = d_qy_all_ + go;
    const double* qz = d_qz_all_ + go;
    const int* cia = d_cia_all_ + co;

    cufftDoubleComplex* d_c = device_alloc<cufftDoubleComplex>(n_proj_);
    cufftDoubleComplex* d_sx = device_alloc<cufftDoubleComplex>(n_proj_);
    cufftDoubleComplex* d_sy = device_alloc<cufftDoubleComplex>(n_proj_);
    cufftDoubleComplex* d_sz = device_alloc<cufftDoubleComplex>(n_proj_);

    const cuDoubleComplex one{1.0, 0.0}, zero{0.0, 0.0};
    // pref = weight * occ with closed-shell f = 2 (CPU twin).
    const double pref = weight * 2.0;

    for (int b = 0; b < n_bands_; ++b) {
        const cufftDoubleComplex* psi =
            d_ritz_ + static_cast<size_t>(b) * ng_;

        // c = βᴴ ψ
        check_cublas(
            cublasZgemv(blas_, CUBLAS_OP_C, ng_, n_proj_, &one,
                        reinterpret_cast<const cuDoubleComplex*>(d_beta_), ng_,
                        reinterpret_cast<const cuDoubleComplex*>(psi), 1, &zero,
                        reinterpret_cast<cuDoubleComplex*>(d_c), 1),
            "Zgemv beta^H psi (nl force c)");

        // s_α = βᴴ (q_α ⊙ ψ)
        auto proj_q = [&](const double* q, cufftDoubleComplex* s) {
            k_mul_real_scale<<<n_blocks(ng_), kBlock>>>(psi, q, ng_, d_work_);
            ATV_CUDA_CHECK(cudaGetLastError());
            check_cublas(
                cublasZgemv(
                    blas_, CUBLAS_OP_C, ng_, n_proj_, &one,
                    reinterpret_cast<const cuDoubleComplex*>(d_beta_), ng_,
                    reinterpret_cast<const cuDoubleComplex*>(d_work_), 1,
                    &zero, reinterpret_cast<cuDoubleComplex*>(s), 1),
                "Zgemv beta^H (q*psi) (nl force s)");
        };
        proj_q(qx, d_sx);
        proj_q(qy, d_sy);
        proj_q(qz, d_sz);

        k_nl_force_from_proj<<<n_blocks(n_proj_), kBlock>>>(
            d_c, d_sx, d_sy, d_sz, d_betad_, cia, pref, n_proj_, d_forces);
        ATV_CUDA_CHECK(cudaGetLastError());
        (void)n_atoms; // bounds already enforced by column atom indices
    }

    device_free(d_c);
    device_free(d_sx);
    device_free(d_sy);
    device_free(d_sz);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
