/// @file cuda/gpw/internal/ao_diis_eigen.cu
/// @brief Device Pulay DIIS + cuSOLVER eigen for GPW Γ RKS/UKS mid-SCF.

#include "atomiverse/cuda/gpw/internal/ao_diis_eigen.cuh"

#include "atomiverse/cuda/device/launch.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {
namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("AoDiisEigenDevice: ") + what
                                 + ": " + cudaGetErrorString(e));
}

void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("AoDiisEigenDevice: ") + what
                                 + " cublas status=" + std::to_string(int(s)));
}

void check_cusolver(cusolverStatus_t s, const char* what) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("AoDiisEigenDevice: ") + what
                                 + " cusolver status="
                                 + std::to_string(int(s)));
}

/// Host solve of the Pulay system B c = rhs with partial pivoting.
/// B is m×m Tr(e_i e_j); bordered with Lagrange −1. Returns false if singular.
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

/// y = Σ_i c_i hist[:, i] for circular history (one thread per element).
__global__ void weighted_sum_hist_kernel(double* __restrict__ y,
                                         const double* __restrict__ hist,
                                         const double* __restrict__ c, int nn,
                                         int m, int hist_start,
                                         int max_vectors) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nn)
        return;
    double acc = 0.0;
    for (int i = 0; i < m; ++i) {
        const int slot = (hist_start + i) % max_vectors;
        acc += c[i] * hist[static_cast<size_t>(slot) * nn + idx];
    }
    y[idx] = acc;
}

/// F′_ii += λ  (column-major m×m).
__global__ void add_diagonal_kernel(double* __restrict__ M, double lambda,
                                    int m) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < m)
        M[i + i * m] += lambda;
}

/// Undo level shift on virtual eigenvalues: eig[i] -= λ for i ≥ n_occ.
__global__ void unshift_virtual_eig_kernel(double* __restrict__ eig, double lambda,
                                           int n_occ, int m) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_occ && i < m)
        eig[i] -= lambda;
}

} // namespace

void AoDiisEigenDevice::ensure_built_() const {
    if (!built_)
        throw std::logic_error("AoDiisEigenDevice: build() required first");
}

void AoDiisEigenDevice::build(cudaStream_t stream, int n, int m, int max_vectors,
                              int n_occ, double occ_factor,
                              double level_shift) {
    destroy();
    if (n <= 0 || m <= 0 || m > n)
        throw std::invalid_argument(
            "AoDiisEigenDevice::build: need n>=1, 1<=m<=n");
    if (n_occ < 0 || n_occ > m)
        throw std::invalid_argument(
            "AoDiisEigenDevice::build: n_occ out of range");
    stream_ = stream;
    n_ = n;
    m_ = m;
    max_vectors_ = max_vectors > 0 ? max_vectors : 0;
    use_diis_ = max_vectors_ >= 1;
    n_occ_ = n_occ;
    occ_factor_ = occ_factor;
    level_shift_ = level_shift;
    have_C_ = false;
    hist_count_ = 0;
    hist_start_ = 0;

    const size_t nn = static_cast<size_t>(n) * n;
    const size_t nm = static_cast<size_t>(n) * m;
    const size_t mm = static_cast<size_t>(m) * m;
    // Scratch must cover n×n (commutator), n×m (X products), m×m (F′).
    const size_t scratch = std::max(nn, std::max(nm, mm));

    check_cuda(cudaMalloc(&d_S_, nn * sizeof(double)), "malloc S");
    check_cuda(cudaMalloc(&d_X_, nm * sizeof(double)), "malloc X");
    check_cuda(cudaMalloc(&d_P_, nn * sizeof(double)), "malloc P");
    check_cuda(cudaMalloc(&d_H_, nn * sizeof(double)), "malloc H");
    check_cuda(cudaMalloc(&d_C_, nm * sizeof(double)), "malloc C");
    check_cuda(cudaMalloc(&d_Cp_, mm * sizeof(double)), "malloc Cp");
    check_cuda(cudaMalloc(&d_eig_, static_cast<size_t>(m) * sizeof(double)),
               "malloc eig");
    check_cuda(cudaMalloc(&d_err_, mm * sizeof(double)), "malloc err");
    check_cuda(cudaMalloc(&d_t1_, scratch * sizeof(double)), "malloc t1");
    check_cuda(cudaMalloc(&d_t2_, scratch * sizeof(double)), "malloc t2");
    check_cuda(cudaMalloc(&d_t3_, nn * sizeof(double)), "malloc t3");
    check_cuda(cudaMalloc(&d_info_, sizeof(int)), "malloc info");

    if (use_diis_) {
        check_cuda(cudaMalloc(&d_H_hist_,
                              static_cast<size_t>(max_vectors_) * nn
                                  * sizeof(double)),
                   "malloc H_hist");
        check_cuda(cudaMalloc(&d_e_hist_,
                              static_cast<size_t>(max_vectors_) * mm
                                  * sizeof(double)),
                   "malloc e_hist");
        check_cuda(cudaMalloc(&d_c_,
                              static_cast<size_t>(max_vectors_)
                                  * sizeof(double)),
                   "malloc c");
        check_cuda(cudaMemsetAsync(d_H_hist_, 0,
                                   static_cast<size_t>(max_vectors_) * nn
                                       * sizeof(double),
                                   stream_),
                   "zero H_hist");
        check_cuda(cudaMemsetAsync(d_e_hist_, 0,
                                   static_cast<size_t>(max_vectors_) * mm
                                       * sizeof(double),
                                   stream_),
                   "zero e_hist");
    }

    cublasHandle_t blas;
    check_cublas(cublasCreate(&blas), "cublasCreate");
    check_cublas(cublasSetStream(blas, stream_), "cublasSetStream");
    check_cublas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST),
                 "cublasSetPointerMode");
    cublas_ = blas;

    cusolverDnHandle_t sol;
    check_cusolver(cusolverDnCreate(&sol), "cusolverDnCreate");
    check_cusolver(cusolverDnSetStream(sol, stream_), "cusolverDnSetStream");
    cusolver_ = sol;

    int lwork = 0;
    check_cusolver(cusolverDnDsyevd_bufferSize(
                       sol, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m,
                       d_Cp_, m, d_eig_, &lwork),
                   "syevd_bufferSize");
    sy_lwork_ = lwork;
    check_cuda(cudaMalloc(&d_sy_work_,
                          static_cast<size_t>(sy_lwork_) * sizeof(double)),
               "malloc syevd work");

    built_ = true;
}

void AoDiisEigenDevice::destroy() {
    if (cublas_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_));
        cublas_ = nullptr;
    }
    if (cusolver_) {
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(cusolver_));
        cusolver_ = nullptr;
    }
    auto free_p = [](double*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    free_p(d_S_);
    free_p(d_X_);
    free_p(d_P_);
    free_p(d_H_);
    free_p(d_C_);
    free_p(d_Cp_);
    free_p(d_eig_);
    free_p(d_err_);
    free_p(d_t1_);
    free_p(d_t2_);
    free_p(d_t3_);
    free_p(d_H_hist_);
    free_p(d_e_hist_);
    free_p(d_c_);
    free_p(d_sy_work_);
    if (d_info_) {
        cudaFree(d_info_);
        d_info_ = nullptr;
    }
    stream_ = nullptr;
    n_ = m_ = max_vectors_ = n_occ_ = 0;
    hist_count_ = hist_start_ = 0;
    sy_lwork_ = 0;
    use_diis_ = false;
    have_C_ = false;
    built_ = false;
}

void AoDiisEigenDevice::upload_SX(const double* S_host, const double* X_host) {
    ensure_built_();
    const size_t nn = static_cast<size_t>(n_) * n_;
    const size_t nm = static_cast<size_t>(n_) * m_;
    check_cuda(cudaMemcpyAsync(d_S_, S_host, nn * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D S");
    check_cuda(cudaMemcpyAsync(d_X_, X_host, nm * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D X");
}

void AoDiisEigenDevice::upload_P(const double* P_host) {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_P_, P_host,
                               static_cast<size_t>(n_) * n_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D P");
}

void AoDiisEigenDevice::upload_C(const double* C_host) {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_C_, C_host,
                               static_cast<size_t>(n_) * m_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D C");
    have_C_ = true;
}

void AoDiisEigenDevice::download_P(double* P_host) const {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(P_host, d_P_,
                               static_cast<size_t>(n_) * n_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H P");
    check_cuda(cudaStreamSynchronize(stream_), "sync D2H P");
}

void AoDiisEigenDevice::download_C(double* C_host) const {
    ensure_built_();
    if (C_host == nullptr)
        throw std::invalid_argument("AoDiisEigenDevice::download_C: null");
    check_cuda(cudaMemcpyAsync(C_host, d_C_,
                               static_cast<size_t>(n_) * m_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H C");
    check_cuda(cudaStreamSynchronize(stream_), "sync D2H C");
}

void AoDiisEigenDevice::download_eig(double* eig_host) const {
    ensure_built_();
    if (eig_host == nullptr)
        throw std::invalid_argument("AoDiisEigenDevice::download_eig: null");
    check_cuda(cudaMemcpyAsync(eig_host, d_eig_,
                               static_cast<size_t>(m_) * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H eig");
    check_cuda(cudaStreamSynchronize(stream_), "sync D2H eig");
}

void AoDiisEigenDevice::reset() {
    hist_count_ = 0;
    hist_start_ = 0;
    // have_C_ intentionally kept: call upload_C after reset to seed level-shift.
}

void AoDiisEigenDevice::form_error_() {
    auto handle = static_cast<cublasHandle_t>(cublas_);
    const int n = n_;
    const int m = m_;
    const double one = 1.0;
    const double zero = 0.0;
    const double neg = -1.0;

    // t1 = P S   (n×n)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_P_, n, d_S_, n, &zero, d_t1_, n),
                 "gemm PS");
    // t2 = H (P S) = HPS
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_H_, n, d_t1_, n, &zero, d_t2_, n),
                 "gemm HPS");
    // t3 = P H
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_P_, n, d_H_, n, &zero, d_t3_, n),
                 "gemm PH");
    // t1 = S (P H) = SPH
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_S_, n, d_t3_, n, &zero, d_t1_, n),
                 "gemm SPH");
    // t2 = HPS − SPH
    check_cublas(cublasDaxpy(handle, n * n, &neg, d_t1_, 1, d_t2_, 1),
                 "axpy HPS-SPH");
    // t1 = (HPS−SPH) X   (n×m)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one,
                             d_t2_, n, d_X_, n, &zero, d_t1_, n),
                 "gemm CX");
    // err = Xᵀ t1   (m×m)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one,
                             d_X_, n, d_t1_, n, &zero, d_err_, m),
                 "gemm XᵀCX");
}

void AoDiisEigenDevice::diis_extrapolate_() {
    if (!use_diis_)
        return;
    auto handle = static_cast<cublasHandle_t>(cublas_);
    const int nn = n_ * n_;
    const int mm = m_ * m_;

    int slot;
    if (hist_count_ < max_vectors_) {
        slot = (hist_start_ + hist_count_) % max_vectors_;
        ++hist_count_;
    } else {
        slot = hist_start_;
        hist_start_ = (hist_start_ + 1) % max_vectors_;
    }
    check_cuda(cudaMemcpyAsync(d_H_hist_ + static_cast<size_t>(slot) * nn, d_H_,
                               static_cast<size_t>(nn) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream_),
               "push H hist");
    check_cuda(cudaMemcpyAsync(d_e_hist_ + static_cast<size_t>(slot) * mm,
                               d_err_, static_cast<size_t>(mm) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream_),
               "push e hist");

    const int mh = hist_count_;
    if (mh < 2)
        return; // warm-up: keep latest H in d_H_

    std::vector<double> Bmm(static_cast<size_t>(mh * mh), 0.0);
    for (int i = 0; i < mh; ++i) {
        const int si = (hist_start_ + i) % max_vectors_;
        for (int j = 0; j <= i; ++j) {
            const int sj = (hist_start_ + j) % max_vectors_;
            double dot_val = 0.0;
            check_cublas(cublasDdot(handle, mm,
                                    d_e_hist_ + static_cast<size_t>(si) * mm, 1,
                                    d_e_hist_ + static_cast<size_t>(sj) * mm, 1,
                                    &dot_val),
                         "dot e_i e_j");
            Bmm[static_cast<size_t>(i * mh + j)] = dot_val;
            Bmm[static_cast<size_t>(j * mh + i)] = dot_val;
        }
    }
    std::vector<double> c_host;
    if (!solve_pulay_host(Bmm, mh, c_host))
        return; // singular → keep latest H
    check_cuda(cudaMemcpyAsync(d_c_, c_host.data(),
                               static_cast<size_t>(mh) * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D c");
    const int blocks = grid_size(nn, kDefaultBlock);
    weighted_sum_hist_kernel<<<blocks, kDefaultBlock, 0, stream_>>>(
        d_H_, d_H_hist_, d_c_, nn, mh, hist_start_, max_vectors_);
    check_cuda(cudaGetLastError(), "weighted_sum_hist");
}

void AoDiisEigenDevice::diagonalize_and_density_() {
    auto handle = static_cast<cublasHandle_t>(cublas_);
    auto sol = static_cast<cusolverDnHandle_t>(cusolver_);
    const int n = n_;
    const int m = m_;
    const double one = 1.0;
    const double zero = 0.0;

    // F′ = Xᵀ H X → d_Cp_
    // t1 = H X  (n×m)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one,
                             d_H_, n, d_X_, n, &zero, d_t1_, n),
                 "gemm HX");
    // Cp = Xᵀ t1  (m×m)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one,
                             d_X_, n, d_t1_, n, &zero, d_Cp_, m),
                 "gemm XᵀHX");

    // Level shift on F′: F′ += λ(I − P′_occ), P′_occ = C′_occ C′_occᵀ
    // with C′_occ = Xᵀ S C_occ from previous C.
    if (level_shift_ > 0.0 && have_C_ && n_occ_ > 0) {
        // t1 = S C  (n×m) then use first n_occ columns as C_occ via lda n
        check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n,
                                 &one, d_S_, n, d_C_, n, &zero, d_t1_, n),
                     "gemm SC");
        // t2 = Xᵀ (S C)  (m×m) — Cp_all; occ = first n_occ cols
        check_cublas(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n,
                                 &one, d_X_, n, d_t1_, n, &zero, d_t2_, m),
                     "gemm XᵀSC");
        // d_Cp_ += λ I
        const int dblocks = grid_size(m, kDefaultBlock);
        add_diagonal_kernel<<<dblocks, kDefaultBlock, 0, stream_>>>(
            d_Cp_, level_shift_, m);
        // d_Cp_ -= λ * Cp_occ * Cp_occᵀ   gemm m×n_occ × n_occ×m
        const double neg_l = -level_shift_;
        check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n_occ_,
                                 &neg_l, d_t2_, m, d_t2_, m, &one, d_Cp_, m),
                     "gemm level-shift projector");
    }

    // syevd in-place on d_Cp_ → eigenvectors; eig → d_eig_
    check_cusolver(cusolverDnDsyevd(sol, CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER, m, d_Cp_, m, d_eig_,
                                    d_sy_work_, sy_lwork_, d_info_),
                   "syevd");
    int info = 0;
    check_cuda(cudaMemcpyAsync(&info, d_info_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H syevd info");
    check_cuda(cudaStreamSynchronize(stream_), "sync syevd info");
    if (info != 0)
        throw std::runtime_error("AoDiisEigenDevice: syevd info="
                                 + std::to_string(info));

    if (level_shift_ > 0.0 && have_C_ && n_occ_ > 0) {
        const int eblocks = grid_size(m, kDefaultBlock);
        unshift_virtual_eig_kernel<<<eblocks, kDefaultBlock, 0, stream_>>>(
            d_eig_, level_shift_, n_occ_, m);
    }

    // C = X Cp  (n×m)
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one,
                             d_X_, n, d_Cp_, m, &zero, d_C_, n),
                 "gemm X Cp");
    have_C_ = true;

    // P = factor * C_occ * C_occᵀ
    if (n_occ_ <= 0) {
        check_cuda(cudaMemsetAsync(d_P_, 0,
                                   static_cast<size_t>(n) * n * sizeof(double),
                                   stream_),
                   "zero P");
        return;
    }
    // gemm: P = factor * C(:,0:n_occ) * C(:,0:n_occ)^T
    check_cublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n_occ_,
                             &occ_factor_, d_C_, n, d_C_, n, &zero, d_P_, n),
                 "gemm P=Cocc Coccᵀ");
}

namespace {

/// @brief Grid-stride max |x_i| into *out (single block; m*m small for GPW).
__global__ void max_abs_kernel(const double* __restrict__ x, int n,
                               double* __restrict__ out) {
    __shared__ double s[256];
    double local = 0.0;
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x))
        local = fmax(local, fabs(x[i]));
    s[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
         stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = s[0];
}

} // namespace

double AoDiisEigenDevice::err_norm_from_device_() const {
    // Device max-|e| (no full-matrix D2H). Single scalar D2H for host stop.
    const int mm = m_ * m_;
    double* d_nrm = d_t1_; // reuse scratch (first double only)
    max_abs_kernel<<<1, 256, 0, stream_>>>(d_err_, mm, d_nrm);
    check_cuda(cudaGetLastError(), "max_abs_kernel");
    double nrm = 0.0;
    check_cuda(cudaMemcpyAsync(&nrm, d_nrm, sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H residual scalar");
    check_cuda(cudaStreamSynchronize(stream_), "sync residual scalar");
    return nrm;
}

double AoDiisEigenDevice::residual(const double* H_host) {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_H_, H_host,
                               static_cast<size_t>(n_) * n_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D H");
    form_error_();
    return err_norm_from_device_();
}

double AoDiisEigenDevice::residual_device(const double* d_H) {
    ensure_built_();
    if (d_H == nullptr)
        throw std::invalid_argument("AoDiisEigenDevice::residual_device: null d_H");
    const size_t bytes =
        static_cast<size_t>(n_) * static_cast<size_t>(n_) * sizeof(double);
    if (d_H != d_H_) {
        check_cuda(cudaMemcpyAsync(d_H_, d_H, bytes, cudaMemcpyDeviceToDevice,
                                   stream_),
                   "D2D H");
    }
    form_error_();
    return err_norm_from_device_();
}

void AoDiisEigenDevice::advance() {
    ensure_built_();
    diis_extrapolate_();
    diagonalize_and_density_();
}

double AoDiisEigenDevice::step(const double* H_host) {
    const double err_norm = residual(H_host);
    advance();
    return err_norm;
}

void AoDiisEigenDevice::error_to_host(const double* H_host,
                                      double* e_host_mm) {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_H_, H_host,
                               static_cast<size_t>(n_) * n_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D H err");
    form_error_();
    check_cuda(cudaMemcpyAsync(e_host_mm, d_err_,
                               static_cast<size_t>(m_) * m_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H e");
    check_cuda(cudaStreamSynchronize(stream_), "sync e");
}

void AoDiisEigenDevice::eigen_to_host(const double* H_host, double* eig_host,
                                      double* C_host) {
    ensure_built_();
    check_cuda(cudaMemcpyAsync(d_H_, H_host,
                               static_cast<size_t>(n_) * n_ * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D H eigen");
    // Skip DIIS/level-shift for oracle: temporarily zero shift + clear C.
    const double saved_shift = level_shift_;
    const bool saved_have = have_C_;
    level_shift_ = 0.0;
    have_C_ = false;
    diagonalize_and_density_();
    level_shift_ = saved_shift;
    have_C_ = saved_have;
    check_cuda(cudaMemcpyAsync(eig_host, d_eig_,
                               static_cast<size_t>(m_) * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H eig");
    check_cuda(cudaMemcpyAsync(C_host, d_C_,
                               static_cast<size_t>(n_) * m_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H C");
    check_cuda(cudaStreamSynchronize(stream_), "sync eigen");
}

void AoDiisEigenDevice::eigen_from_device(const double* d_H, double* eig_host,
                                          double* C_host) {
    ensure_built_();
    if (d_H == nullptr)
        throw std::invalid_argument(
            "AoDiisEigenDevice::eigen_from_device: null d_H");
    const size_t bytes =
        static_cast<size_t>(n_) * static_cast<size_t>(n_) * sizeof(double);
    if (d_H != d_H_) {
        check_cuda(cudaMemcpyAsync(d_H_, d_H, bytes, cudaMemcpyDeviceToDevice,
                                   stream_),
                   "D2D H eigen");
    }
    // Payload eigen of raw KS H: no DIIS, no level shift (matches host
    // store_converged_state / store_converged_state_uks → diagonalize_fock
    // without Saunders–Hillier; RKS device-payload + UKS uks-payload-device).
    const double saved_shift = level_shift_;
    const bool saved_have = have_C_;
    level_shift_ = 0.0;
    have_C_ = false;
    diagonalize_and_density_();
    level_shift_ = saved_shift;
    have_C_ = saved_have;
    // Boundary D2H only (post-convergence payload).
    if (eig_host != nullptr) {
        check_cuda(cudaMemcpyAsync(eig_host, d_eig_,
                                   static_cast<size_t>(m_) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H eig payload");
    }
    if (C_host != nullptr) {
        check_cuda(cudaMemcpyAsync(C_host, d_C_,
                                   static_cast<size_t>(n_) * m_ * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H C payload");
    }
    check_cuda(cudaStreamSynchronize(stream_), "sync eigen_from_device");
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
