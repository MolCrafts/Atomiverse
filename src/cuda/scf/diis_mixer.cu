/// @file cuda/scf/diis_mixer.cu
/// @brief Device CDIIS (Pulay) Fock mixer + Saunders–Hillier level shift.
///
/// The commutator, history Gram matrix, Pulay LU solve, extrapolation and
/// level shift stay on device. cuBLAS owns dense contractions; cuSOLVER owns
/// pivoted LU. Build reserves all scratch; step transfers no scientific data.

#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/frame/types.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

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

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("CUDA DIIS cuBLAS status "
                                 + std::to_string(static_cast<int>(status)));
}

void check_solver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("CUDA DIIS cuSOLVER status "
                                 + std::to_string(static_cast<int>(status)));
}

/// @brief Assemble [Gram,-1;-1,0] and rhs (0,...,0,-1), column major.
///
/// Elementwise map: one thread per entry, contiguous column-major writes.
/// No reduction or shared storage; cuBLAS computed the error inner products.
/// Physical history slots are sufficient: a permutation of Pulay rows and
/// columns leaves the extrapolation invariant, including circular overwrite.
__global__ void assemble_pulay_kernel(const prec_t* gram, prec_t* A,
                                      prec_t* rhs, int m) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = m + 1;
    if (idx >= dim * dim)
        return;
    const int row = idx % dim;
    const int col = idx / dim;
    A[idx] = row < m && col < m     ? gram[row + col * m]
             : row == m && col == m ? prec_t(0)
                                    : prec_t(-1);
    if (col == 0)
        rhs[row] = row == m ? prec_t(-1) : prec_t(0);
}

/// @brief On a singular/invalid solve select the most recent Fock exactly.
///
/// This tiny control operation inspects at most max_vectors+1 LU pivots and
/// coefficients. A single thread avoids a second scratch reduction/launch;
/// dense matrix science is performed by cuSOLVER and cuBLAS, not here.
__global__ void validate_pulay_kernel(const prec_t* lu, prec_t* c,
                                      const int* info, int m, int latest) {
    const int dim = m + 1;
    bool valid = info[0] == 0 && info[1] == 0;
    for (int i = 0; i < dim; ++i)
        valid = valid && isfinite(c[i]) && isfinite(lu[i + i * dim])
                && fabs(lu[i + i * dim]) >= 1e-14;
    if (!valid)
        for (int i = 0; i < m; ++i)
            c[i] = i == latest ? prec_t(1) : prec_t(0);
}

} // namespace

DIISMixer::DIISMixer(int max_vectors, int start_iter, double level_shift)
    : max_vectors_(max_vectors), start_iter_(start_iter),
      level_shift_(level_shift) {
    if (max_vectors < 1)
        throw std::invalid_argument("CUDA DIIS: max_vectors must be positive");
}

void DIISMixer::build(const SolverContext& ctx, SCFWorkspace& ws) {
    if (built_)
        throw std::logic_error("CUDA DIIS: build called twice without destroy");
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    n_ = ws.n;
    if (n_ <= 0) {
        built_ = true;
        return;
    }
    if (n_ > std::numeric_limits<int>::max() / n_
        || max_vectors_ == std::numeric_limits<int>::max()
        || max_vectors_ + 1
               > std::numeric_limits<int>::max() / (max_vectors_ + 1))
        throw std::invalid_argument(
            "CUDA DIIS: matrix dimensions exceed BLAS int range");
    const int nn = n_ * n_;
    const int dim = max_vectors_ + 1;
    const size_t mat_bytes = static_cast<size_t>(nn)
                             * static_cast<size_t>(max_vectors_)
                             * sizeof(prec_t);
    try {
        ATV_CUDA_CHECK(cudaMalloc(&d_F_hist_, mat_bytes));
        ATV_CUDA_CHECK(cudaMalloc(&d_e_hist_, mat_bytes));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_err_, static_cast<size_t>(nn) * sizeof(prec_t)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_temp1_, static_cast<size_t>(nn) * sizeof(prec_t)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_temp2_, static_cast<size_t>(nn) * sizeof(prec_t)));
        ATV_CUDA_CHECK(cudaMalloc(&d_B_, static_cast<size_t>(max_vectors_)
                                             * max_vectors_ * sizeof(prec_t)));
        ATV_CUDA_CHECK(cudaMalloc(&d_pulay_, static_cast<size_t>(dim) * dim
                                                 * sizeof(prec_t)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_c_, static_cast<size_t>(dim) * sizeof(prec_t)));
        ATV_CUDA_CHECK(
            cudaMalloc(&d_pivots_, static_cast<size_t>(dim) * sizeof(int)));
        ATV_CUDA_CHECK(cudaMalloc(&d_info_, 2 * sizeof(int)));
        ATV_CUDA_CHECK(cudaMemsetAsync(d_F_hist_, 0, mat_bytes, stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(d_e_hist_, 0, mat_bytes, stream));

        cublasHandle_t handle = nullptr;
        check_blas(cublasCreate(&handle));
        cublas_handle_ = handle;
        check_blas(cublasSetStream(handle, stream));
        cusolverDnHandle_t solver = nullptr;
        check_solver(cusolverDnCreate(&solver));
        cusolver_handle_ = solver;
        check_solver(cusolverDnSetStream(solver, stream));
        int lwork = 0;
        // Query each actual dimension: library workspace size need not be
        // monotone in matrix order. This runs only during setup.
        for (int order = 3; order <= dim; ++order) {
            int required = 0;
            check_solver(cusolverDnDgetrf_bufferSize(
                solver, order, order, d_pulay_, order, &required));
            if (required > lwork)
                lwork = required;
        }
        if (lwork > 0)
            ATV_CUDA_CHECK(cudaMalloc(
                &d_solver_work_, static_cast<size_t>(lwork) * sizeof(prec_t)));
        hist_count_ = 0;
        hist_start_ = 0;
        built_ = true;
    } catch (...) {
        destroy(ctx, ws);
        throw;
    }
}

void DIISMixer::reset() {
    hist_count_ = 0;
    hist_start_ = 0;
}

void DIISMixer::measure_error_(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged || scf.n <= 0 || !built_)
        return;
    if (scf.n != n_)
        throw std::invalid_argument(
            "CUDA DIIS: matrix order changed after build");
    if (!scf.S || !scf.X || !scf.work || !scf.P || !scf.H || !scf.F_prime)
        throw std::invalid_argument(
            "CUDA DIIS: missing overlap, density or Fock workspace");

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    check_blas(cublasSetStream(handle, stream));
    check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    const int n = scf.n;
    const int nn = n * n;
    const work_t one = work_t(1);
    const work_t zero = work_t(0);
    const work_t neg_one = work_t(-1);

    // ── e = Xᵀ (F P S − S P F) X  (unshifted F in scf.work) ───────────
    // t1 = P S
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            scf.P, n, scf.S, n, &zero, d_temp1_, n));
    // t2 = F (P S) = F P S
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            scf.work, n, d_temp1_, n, &zero, d_temp2_, n));
    // t1 = P F
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            scf.P, n, scf.work, n, &zero, d_temp1_, n));
    // err = S (P F) = S P F   then  FPS − SPF → d_err_
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            scf.S, n, d_temp1_, n, &zero, d_err_, n));
    check_blas(detail::axpy(handle, nn, &neg_one, d_err_, 1, d_temp2_,
                            1)); // t2 = FPS−SPF
    // t1 = (FPS−SPF) X
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            d_temp2_, n, scf.X, n, &zero, d_temp1_, n));
    // e = Xᵀ t1
    check_blas(detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
                            scf.X, n, d_temp1_, n, &zero, d_err_, n));
}

void DIISMixer::transform_fock_(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    const int n = scf.n;
    const int nn = n * n;
    const work_t one = work_t(1);
    const work_t zero = work_t(0);
    // ── F′ = Xᵀ F X → H / F_prime (EigenSolver input) ────────────────
    check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                            scf.work, n, scf.X, n, &zero, d_temp1_, n));
    check_blas(detail::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
                            scf.X, n, d_temp1_, n, &zero, scf.F_prime, n));
    ATV_CUDA_CHECK(cudaMemcpyAsync(scf.H, scf.F_prime,
                                   static_cast<size_t>(nn) * sizeof(prec_t),
                                   cudaMemcpyDeviceToDevice, stream));
}

void DIISMixer::canonicalize(const SolverContext& ctx, SCFWorkspace& ws) {
    if (ws.converged || ws.n <= 0 || !built_)
        return;
    measure_error_(ctx, ws);
    transform_fock_(ctx, ws);
}

void DIISMixer::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged || scf.n <= 0 || !built_)
        return;
    measure_error_(ctx, ws);
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    const int n = scf.n;
    const int nn = n * n;
    const work_t one = work_t(1);
    const work_t zero = work_t(0);
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
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(d_F_hist_ + static_cast<size_t>(slot) * nn,
                            scf.work, static_cast<size_t>(nn) * sizeof(prec_t),
                            cudaMemcpyDeviceToDevice, stream));
        ATV_CUDA_CHECK(
            cudaMemcpyAsync(d_e_hist_ + static_cast<size_t>(slot) * nn, d_err_,
                            static_cast<size_t>(nn) * sizeof(prec_t),
                            cudaMemcpyDeviceToDevice, stream));

        const int m = hist_count_;
        if (m >= 2) {
            // One dense contraction over physical history slots replaces
            // m(m+1)/2 synchronizing host-result DOT calls.
            check_blas(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, nn,
                                   &one, d_e_hist_, nn, d_e_hist_, nn, &zero,
                                   d_B_, m));
            const int dim = m + 1;
            assemble_pulay_kernel<<<cuda::grid_size(dim * dim,
                                                    cuda::kDefaultBlock),
                                    cuda::kDefaultBlock, 0, stream>>>(
                d_B_, d_pulay_, d_c_, m);
            ATV_CUDA_CHECK(cudaGetLastError());
            auto solver = static_cast<cusolverDnHandle_t>(cusolver_handle_);
            check_solver(cusolverDnSetStream(solver, stream));
            check_solver(cusolverDnDgetrf(solver, dim, dim, d_pulay_, dim,
                                          d_solver_work_, d_pivots_, d_info_));
            check_solver(cusolverDnDgetrs(solver, CUBLAS_OP_N, dim, 1, d_pulay_,
                                          dim, d_pivots_, d_c_, dim,
                                          d_info_ + 1));
            validate_pulay_kernel<<<1, 1, 0, stream>>>(d_pulay_, d_c_, d_info_,
                                                       m, slot);
            ATV_CUDA_CHECK(cudaGetLastError());
            check_blas(cublasDgemv(handle, CUBLAS_OP_N, nn, m, &one, d_F_hist_,
                                   nn, d_c_, 1, &zero, scf.work, 1));
        }
    }

    transform_fock_(ctx, ws);

    // ── Level shift on F′ only (energy keeps unshifted F in work) ────
    // F′ += λ(I − P′_occ), P′_occ = C′_occ C′_occᵀ from previous diagonalize.
    // Skipped on step 0 (no C′) and |λ| ≤ 1e-4 (PySCF threshold).
    if (std::abs(level_shift_) > 1e-4 && iter > 0 && scf.C_prime
        && scf.n_occ > 0) {
        const prec_t lambda = static_cast<prec_t>(level_shift_);
        int dblocks = cuda::grid_size(n, cuda::kDefaultBlock);
        add_diagonal_kernel<<<dblocks, cuda::kDefaultBlock, 0, stream>>>(
            scf.H, lambda, n);
        ATV_CUDA_CHECK(cudaGetLastError());
        const work_t neg_lambda = static_cast<work_t>(-level_shift_);
        const work_t one_b = work_t(1);
        check_blas(detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n,
                                scf.n_occ, &neg_lambda, scf.C_prime, n,
                                scf.C_prime, n, &one_b, scf.H, n));
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
    if (cusolver_handle_) {
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(cusolver_handle_));
        cusolver_handle_ = nullptr;
    }
    cudaFree(d_pulay_);
    d_pulay_ = nullptr;
    cudaFree(d_solver_work_);
    d_solver_work_ = nullptr;
    cudaFree(d_pivots_);
    d_pivots_ = nullptr;
    cudaFree(d_info_);
    d_info_ = nullptr;
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
