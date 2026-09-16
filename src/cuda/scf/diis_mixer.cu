/// @file cuda/scf/diis_mixer.cu
/// @brief Device CDIIS (Pulay) Fock mixer + Saunders–Hillier level shift.
///
/// Heavy work (error commutator, history, F linear combination, F′ transform,
/// level-shift projector) is device-side via cuBLAS GEMM/DOT/AXPY. Only the
/// (m+1)×(m+1) Pulay Lagrange system is solved on host — never the Fock matrix.

#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"

#include "atomiverse/cuda/device/launch.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/frame/types.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {
namespace {

/// @brief F' += λ on the diagonal (col-major n×n). Same kernel as rhf_fock.cu.
__global__ void add_diagonal_kernel(prec_t* __restrict__ M, prec_t lambda,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        M[i + i * n] += lambda;
}

/// @brief Host solve of the Pulay system B c = rhs with partial pivoting.
///
/// B is (m+1)×(m+1) row-major: B_ij = Tr(e_i e_j) for i,j < m; last row/col
/// are the Lagrange constraint (−1), B(m,m)=0; rhs = (0…0, −1).
/// On success writes c[0..m) and returns true; on singular/pivot fail false.
bool solve_pulay_host(const std::vector<double>& Bmm, int m,
                      std::vector<double>& c_out) {
    const int dim = m + 1;
    std::vector<double> A(static_cast<size_t>(dim * dim), 0.0);
    std::vector<double> b(static_cast<size_t>(dim), 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            A[static_cast<size_t>(i * dim + j)] =
                Bmm[static_cast<size_t>(i * m + j)];
    for (int i = 0; i < m; ++i) {
        A[static_cast<size_t>(m * dim + i)] = -1.0;
        A[static_cast<size_t>(i * dim + m)] = -1.0;
    }
    A[static_cast<size_t>(m * dim + m)] = 0.0;
    b[static_cast<size_t>(m)] = -1.0;

    // Gaussian elimination with partial pivoting (in-place on A|b).
    for (int k = 0; k < dim; ++k) {
        int piv = k;
        double best = std::abs(A[static_cast<size_t>(k * dim + k)]);
        for (int i = k + 1; i < dim; ++i) {
            const double v = std::abs(A[static_cast<size_t>(i * dim + k)]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-14)
            return false;
        if (piv != k) {
            for (int j = k; j < dim; ++j)
                std::swap(A[static_cast<size_t>(k * dim + j)],
                          A[static_cast<size_t>(piv * dim + j)]);
            std::swap(b[static_cast<size_t>(k)], b[static_cast<size_t>(piv)]);
        }
        const double akk = A[static_cast<size_t>(k * dim + k)];
        for (int i = k + 1; i < dim; ++i) {
            const double f = A[static_cast<size_t>(i * dim + k)] / akk;
            A[static_cast<size_t>(i * dim + k)] = 0.0;
            for (int j = k + 1; j < dim; ++j)
                A[static_cast<size_t>(i * dim + j)] -=
                    f * A[static_cast<size_t>(k * dim + j)];
            b[static_cast<size_t>(i)] -= f * b[static_cast<size_t>(k)];
        }
    }
    // Back-substitution.
    std::vector<double> x(static_cast<size_t>(dim), 0.0);
    for (int i = dim - 1; i >= 0; --i) {
        double s = b[static_cast<size_t>(i)];
        for (int j = i + 1; j < dim; ++j)
            s -= A[static_cast<size_t>(i * dim + j)] * x[static_cast<size_t>(j)];
        const double aii = A[static_cast<size_t>(i * dim + i)];
        if (std::abs(aii) < 1e-14)
            return false;
        x[static_cast<size_t>(i)] = s / aii;
    }
    c_out.resize(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i)
        c_out[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    return true;
}

/// @brief y = sum_i c_i * hist[:, i]  for circular history (device).
///
/// One thread per matrix element; m is small (≤ 8). Skills: boundary (idx<nn),
/// coalesced col-major loads along the element index.
__global__ void weighted_sum_hist_kernel(prec_t* __restrict__ y,
                                         const prec_t* __restrict__ hist,
                                         const prec_t* __restrict__ c, int nn,
                                         int m, int hist_start,
                                         int max_vectors) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nn)
        return;
    work_t acc = work_t(0);
    for (int i = 0; i < m; ++i) {
        const int slot = (hist_start + i) % max_vectors;
        acc += static_cast<work_t>(c[i])
               * static_cast<work_t>(hist[slot * nn + idx]);
    }
    y[idx] = static_cast<prec_t>(acc);
}

} // namespace

DIISMixer::DIISMixer(int max_vectors, int start_iter, double level_shift)
    : max_vectors_(max_vectors > 0 ? max_vectors : 8),
      start_iter_(start_iter), level_shift_(level_shift) {}

void DIISMixer::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    n_ = ws.n;
    if (n_ <= 0) {
        built_ = true;
        return;
    }
    const int nn = n_ * n_;
    const size_t mat_bytes =
        static_cast<size_t>(nn) * static_cast<size_t>(max_vectors_)
        * sizeof(prec_t);

    cudaMalloc(&d_F_hist_, mat_bytes);
    cudaMalloc(&d_e_hist_, mat_bytes);
    cudaMalloc(&d_err_, static_cast<size_t>(nn) * sizeof(prec_t));
    cudaMalloc(&d_temp1_, static_cast<size_t>(nn) * sizeof(prec_t));
    cudaMalloc(&d_temp2_, static_cast<size_t>(nn) * sizeof(prec_t));
    cudaMalloc(&d_B_, static_cast<size_t>(max_vectors_ * max_vectors_)
                          * sizeof(prec_t));
    cudaMalloc(&d_c_, static_cast<size_t>(max_vectors_) * sizeof(prec_t));
    cudaMemsetAsync(d_F_hist_, 0, mat_bytes, stream);
    cudaMemsetAsync(d_e_hist_, 0, mat_bytes, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;

    hist_count_ = 0;
    hist_start_ = 0;
    built_ = true;
}

void DIISMixer::reset() {
    hist_count_ = 0;
    hist_start_ = 0;
}

void DIISMixer::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged || scf.n <= 0 || !built_)
        return;
    // Overlap is required for the AO-basis DIIS error. Without S, keep the
    // Fock path's F′ in H and only apply level shift (should not happen on
    // RHF paths after S upload).
    if (!scf.S || !scf.X || !scf.work || !scf.P)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    cublasSetStream(handle, stream);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    const int n = scf.n;
    const int nn = n * n;
    const work_t one = work_t(1);
    const work_t zero = work_t(0);
    const work_t neg_one = work_t(-1);

    // ── e = Xᵀ (F P S − S P F) X  (unshifted F in scf.work) ───────────
    // t1 = P S
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.P, n,
                 scf.S, n, &zero, d_temp1_, n);
    // t2 = F (P S) = F P S
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.work, n,
                 d_temp1_, n, &zero, d_temp2_, n);
    // t1 = P F
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.P, n,
                 scf.work, n, &zero, d_temp1_, n);
    // err = S (P F) = S P F   then  FPS − SPF → d_err_
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.S, n,
                 d_temp1_, n, &zero, d_err_, n);
    detail::axpy(handle, nn, &neg_one, d_err_, 1, d_temp2_, 1); // t2 = FPS−SPF
    // t1 = (FPS−SPF) X
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_temp2_, n,
                 scf.X, n, &zero, d_temp1_, n);
    // e = Xᵀ t1
    detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, scf.X, n,
                 d_temp1_, n, &zero, d_err_, n);

    // ── Push history + Pulay extrapolation (CPU: iter >= diis_start) ──
    const int iter = ctx.step;
    if (iter >= start_iter_) {
        // Circular push of (F, e): overwrite oldest when full.
        int slot;
        if (hist_count_ < max_vectors_) {
            slot = (hist_start_ + hist_count_) % max_vectors_;
            ++hist_count_;
        } else {
            slot = hist_start_;
            hist_start_ = (hist_start_ + 1) % max_vectors_;
        }
        cudaMemcpyAsync(d_F_hist_ + static_cast<size_t>(slot) * nn, scf.work,
                        static_cast<size_t>(nn) * sizeof(prec_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_e_hist_ + static_cast<size_t>(slot) * nn, d_err_,
                        static_cast<size_t>(nn) * sizeof(prec_t),
                        cudaMemcpyDeviceToDevice, stream);

        const int m = hist_count_;
        if (m >= 2) {
            // B_ij = Tr(e_i e_j) via device dots (host only collects m² scalars).
            std::vector<double> Bmm(static_cast<size_t>(m * m), 0.0);
            for (int i = 0; i < m; ++i) {
                const int si = (hist_start_ + i) % max_vectors_;
                for (int j = 0; j <= i; ++j) {
                    const int sj = (hist_start_ + j) % max_vectors_;
                    prec_t dot_val = 0;
                    detail::dot(handle, nn, d_e_hist_ + si * nn, 1,
                                d_e_hist_ + sj * nn, 1, &dot_val);
                    // cublas DOT is host-mode: result is available after the
                    // call returns (stream sync internal to the reduction for
                    // POINTER_MODE_HOST).
                    const double v = static_cast<double>(dot_val);
                    Bmm[static_cast<size_t>(i * m + j)] = v;
                    Bmm[static_cast<size_t>(j * m + i)] = v;
                }
            }

            std::vector<double> c_host;
            if (solve_pulay_host(Bmm, m, c_host)) {
                std::vector<prec_t> c_prec(static_cast<size_t>(m));
                for (int i = 0; i < m; ++i)
                    c_prec[static_cast<size_t>(i)] =
                        static_cast<prec_t>(c_host[static_cast<size_t>(i)]);
                cudaMemcpyAsync(d_c_, c_prec.data(),
                                static_cast<size_t>(m) * sizeof(prec_t),
                                cudaMemcpyHostToDevice, stream);
                const int blocks = cuda::grid_size(nn, cuda::kDefaultBlock);
                weighted_sum_hist_kernel<<<blocks, cuda::kDefaultBlock, 0,
                                           stream>>>(
                    scf.work, d_F_hist_, d_c_, nn, m, hist_start_,
                    max_vectors_);
            }
            // On singular B: leave scf.work as the latest F (CPU catch path).
        }
    }

    // ── F′ = Xᵀ F X → H / F_prime (EigenSolver input) ────────────────
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, scf.work, n,
                 scf.X, n, &zero, d_temp1_, n);
    detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, scf.X, n,
                 d_temp1_, n, &zero, scf.F_prime, n);
    cudaMemcpyAsync(scf.H, scf.F_prime, static_cast<size_t>(nn) * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);

    // ── Level shift on F′ only (energy keeps unshifted F in work) ────
    // F′ += λ(I − P′_occ), P′_occ = C′_occ C′_occᵀ from previous diagonalize.
    // Skipped on step 0 (no C′) and |λ| ≤ 1e-4 (PySCF threshold).
    if (std::abs(level_shift_) > 1e-4 && iter > 0 && scf.C_prime
        && scf.n_occ > 0) {
        const prec_t lambda = static_cast<prec_t>(level_shift_);
        int dblocks = cuda::grid_size(n, cuda::kDefaultBlock);
        add_diagonal_kernel<<<dblocks, cuda::kDefaultBlock, 0, stream>>>(
            scf.H, lambda, n);
        const work_t neg_lambda = static_cast<work_t>(-level_shift_);
        const work_t one_b = work_t(1);
        detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, scf.n_occ,
                     &neg_lambda, scf.C_prime, n, scf.C_prime, n, &one_b, scf.H,
                     n);
    }
}

void DIISMixer::destroy(const SolverContext& /*ctx*/, SCFWorkspace& /*ws*/) {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
    cudaFree(d_F_hist_);
    d_F_hist_ = nullptr;
    cudaFree(d_e_hist_);
    d_e_hist_ = nullptr;
    cudaFree(d_err_);
    d_err_ = nullptr;
    cudaFree(d_temp1_);
    d_temp1_ = nullptr;
    cudaFree(d_temp2_);
    d_temp2_ = nullptr;
    cudaFree(d_B_);
    d_B_ = nullptr;
    cudaFree(d_c_);
    d_c_ = nullptr;
    hist_count_ = 0;
    hist_start_ = 0;
    n_ = 0;
    built_ = false;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV
