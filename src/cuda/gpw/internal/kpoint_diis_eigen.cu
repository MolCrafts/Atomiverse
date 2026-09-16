/// @file cuda/gpw/internal/kpoint_diis_eigen.cu
/// @brief Device multi-k Pulay DIIS + cuSOLVER complex Hermitian eigen.

#include "atomiverse/cuda/gpw/internal/kpoint_diis_eigen.cuh"

#include "atomiverse/cuda/device/launch.cuh"

#include <cuComplex.h>
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

using Cplx = std::complex<double>;

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("KpointDiisEigenDevice: ") + what
                                 + ": " + cudaGetErrorString(e));
}

void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("KpointDiisEigenDevice: ") + what
                                 + " cublas status=" + std::to_string(int(s)));
}

void check_cusolver(cusolverStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("KpointDiisEigenDevice: ") + what
                                 + " cusolver status="
                                 + std::to_string(int(s)));
}

/// Host solve of the real Pulay system B c = rhs (same as AoDiisEigenDevice).
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

/// y = Σ_i c_i hist[slot_i] for complex circular history (one thread / element).
__global__ void weighted_sum_cplx_hist_kernel(cuDoubleComplex* __restrict__ y,
                                              const cuDoubleComplex* __restrict__ hist,
                                              const double* __restrict__ c,
                                              int nn, int m, int hist_start,
                                              int max_vectors, int nk, int ik) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nn)
        return;
    double re = 0.0, im = 0.0;
    for (int i = 0; i < m; ++i) {
        const int slot = (hist_start + i) % max_vectors;
        // hist layout: [slot][ik][nn]
        const size_t base =
            (static_cast<size_t>(slot) * static_cast<size_t>(nk)
             + static_cast<size_t>(ik))
            * static_cast<size_t>(nn);
        const cuDoubleComplex v = hist[base + static_cast<size_t>(idx)];
        const double ci = c[i];
        re += ci * cuCreal(v);
        im += ci * cuCimag(v);
    }
    y[idx] = make_cuDoubleComplex(re, im);
}

/// Device row-major → column-major transpose for n×n (or rows×cols) complex.
/// One thread per element: cm[i + j*rows] = rm[i*cols + j].
__global__ void cplx_rm_to_cm_kernel(const cuDoubleComplex* __restrict__ rm,
                                     cuDoubleComplex* __restrict__ cm,
                                     int rows, int cols) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int nn = rows * cols;
    if (idx >= nn)
        return;
    const int i = idx / cols;
    const int j = idx % cols;
    cm[i + j * rows] = rm[i * cols + j];
}

void rm_to_cm_host(const Cplx* rm, Cplx* cm, int rows, int cols) {
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            cm[static_cast<size_t>(i) + static_cast<size_t>(j) * rows] =
                rm[static_cast<size_t>(i) * cols + static_cast<size_t>(j)];
}

void cm_to_rm_host(const Cplx* cm, Cplx* rm, int rows, int cols) {
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            rm[static_cast<size_t>(i) * cols + static_cast<size_t>(j)] =
                cm[static_cast<size_t>(i) + static_cast<size_t>(j) * rows];
}

cuDoubleComplex* as_z(void* p) {
    return static_cast<cuDoubleComplex*>(p);
}
const cuDoubleComplex* as_z(const void* p) {
    return static_cast<const cuDoubleComplex*>(p);
}

} // namespace

void KpointDiisEigenDevice::ensure_built_() const {
    if (!built_)
        throw std::logic_error("KpointDiisEigenDevice: build() required first");
}

void KpointDiisEigenDevice::ensure_ik_(int ik) const {
    if (ik < 0 || ik >= nk_)
        throw std::out_of_range("KpointDiisEigenDevice: ik out of range");
}

int KpointDiisEigenDevice::m_k(int ik) const {
    ensure_built_();
    ensure_ik_(ik);
    return m_k_[static_cast<size_t>(ik)];
}

const void* KpointDiisEigenDevice::d_P(int ik) const {
    ensure_built_();
    ensure_ik_(ik);
    return d_P_[static_cast<size_t>(ik)];
}

void KpointDiisEigenDevice::host_rm_to_dev_cm_(const Cplx* rm, void* d_cm,
                                              int rows, int cols) const {
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    stage_.resize(n);
    rm_to_cm_host(rm, stage_.data(), rows, cols);
    check_cuda(cudaMemcpyAsync(d_cm, stage_.data(), n * sizeof(Cplx),
                               cudaMemcpyHostToDevice, stream_),
               "H2D rm→cm");
}

void KpointDiisEigenDevice::dev_cm_to_host_rm_(const void* d_cm, Cplx* rm,
                                              int rows, int cols) const {
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    stage_.resize(n);
    check_cuda(cudaMemcpyAsync(stage_.data(), d_cm, n * sizeof(Cplx),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H cm");
    check_cuda(cudaStreamSynchronize(stream_), "sync D2H cm");
    cm_to_rm_host(stage_.data(), rm, rows, cols);
}

void KpointDiisEigenDevice::build(cudaStream_t stream, int nk, int n, int max_m,
                                  int max_vectors, int n_occ, double occ_factor,
                                  const std::vector<double>& weights) {
    destroy();
    if (nk < 1 || n < 1 || max_m < 1 || max_m > n)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::build: need nk>=1, n>=1, 1<=max_m<=n");
    if (n_occ < 0 || n_occ > max_m)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::build: n_occ out of range");
    if (static_cast<int>(weights.size()) != nk)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::build: weights.size() must equal nk");

    stream_ = stream;
    nk_ = nk;
    n_ = n;
    max_m_ = max_m;
    max_vectors_ = max_vectors > 0 ? max_vectors : 0;
    use_diis_ = max_vectors_ >= 1;
    n_occ_ = n_occ;
    occ_factor_ = occ_factor;
    weights_ = weights;
    m_k_.assign(static_cast<size_t>(nk), 0);
    hist_count_ = 0;
    hist_start_ = 0;

    const size_t nn = static_cast<size_t>(n) * n;
    const size_t nm = static_cast<size_t>(n) * max_m;
    const size_t mm = static_cast<size_t>(max_m) * max_m;
    const size_t scratch = std::max(nn, std::max(nm, mm));

    d_S_.resize(static_cast<size_t>(nk), nullptr);
    d_X_.resize(static_cast<size_t>(nk), nullptr);
    d_P_.resize(static_cast<size_t>(nk), nullptr);
    d_H_.resize(static_cast<size_t>(nk), nullptr);
    d_C_.resize(static_cast<size_t>(nk), nullptr);
    d_Cp_.resize(static_cast<size_t>(nk), nullptr);
    d_err_.resize(static_cast<size_t>(nk), nullptr);
    d_eig_.resize(static_cast<size_t>(nk), nullptr);

    for (int ik = 0; ik < nk; ++ik) {
        const size_t i = static_cast<size_t>(ik);
        check_cuda(cudaMalloc(&d_S_[i], nn * sizeof(Cplx)), "malloc S");
        check_cuda(cudaMalloc(&d_X_[i], nm * sizeof(Cplx)), "malloc X");
        check_cuda(cudaMalloc(&d_P_[i], nn * sizeof(Cplx)), "malloc P");
        check_cuda(cudaMalloc(&d_H_[i], nn * sizeof(Cplx)), "malloc H");
        check_cuda(cudaMalloc(&d_C_[i], nm * sizeof(Cplx)), "malloc C");
        check_cuda(cudaMalloc(&d_Cp_[i], mm * sizeof(Cplx)), "malloc Cp");
        check_cuda(cudaMalloc(&d_err_[i], mm * sizeof(Cplx)), "malloc err");
        check_cuda(cudaMalloc(&d_eig_[i],
                              static_cast<size_t>(max_m) * sizeof(double)),
                   "malloc eig");
        check_cuda(cudaMemsetAsync(d_X_[i], 0, nm * sizeof(Cplx), stream_),
                   "zero X");
        check_cuda(cudaMemsetAsync(d_err_[i], 0, mm * sizeof(Cplx), stream_),
                   "zero err");
    }

    check_cuda(cudaMalloc(&d_t1_, scratch * sizeof(Cplx)), "malloc t1");
    check_cuda(cudaMalloc(&d_t2_, scratch * sizeof(Cplx)), "malloc t2");
    check_cuda(cudaMalloc(&d_t3_, nn * sizeof(Cplx)), "malloc t3");
    check_cuda(cudaMalloc(&d_info_, sizeof(int)), "malloc info");

    if (use_diis_) {
        const size_t h_bytes = static_cast<size_t>(max_vectors_)
                               * static_cast<size_t>(nk) * nn * sizeof(Cplx);
        const size_t e_bytes = static_cast<size_t>(max_vectors_)
                               * static_cast<size_t>(nk) * mm * sizeof(Cplx);
        check_cuda(cudaMalloc(&d_H_hist_, h_bytes), "malloc H_hist");
        check_cuda(cudaMalloc(&d_e_hist_, e_bytes), "malloc e_hist");
        check_cuda(cudaMalloc(&d_c_,
                              static_cast<size_t>(max_vectors_)
                                  * sizeof(double)),
                   "malloc c");
        check_cuda(cudaMemsetAsync(d_H_hist_, 0, h_bytes, stream_),
                   "zero H_hist");
        check_cuda(cudaMemsetAsync(d_e_hist_, 0, e_bytes, stream_),
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

    // Workspace for largest possible m (max_m).
    int lwork = 0;
    check_cusolver(
        cusolverDnZheevd_bufferSize(
            sol, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, max_m,
            as_z(d_Cp_[0]), max_m, d_eig_[0], &lwork),
        "zheevd_bufferSize");
    sy_lwork_ = lwork;
    check_cuda(cudaMalloc(&d_sy_work_,
                          static_cast<size_t>(sy_lwork_) * sizeof(Cplx)),
               "malloc zheevd work");

    built_ = true;
}

void KpointDiisEigenDevice::destroy() {
    if (cublas_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_));
        cublas_ = nullptr;
    }
    if (cusolver_) {
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(cusolver_));
        cusolver_ = nullptr;
    }
    auto free_v = [](void*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    auto free_vec = [&](std::vector<void*>& v) {
        for (void*& p : v)
            free_v(p);
        v.clear();
    };
    free_vec(d_S_);
    free_vec(d_X_);
    free_vec(d_P_);
    free_vec(d_H_);
    free_vec(d_C_);
    free_vec(d_Cp_);
    free_vec(d_err_);
    for (double*& p : d_eig_) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    }
    d_eig_.clear();
    free_v(d_t1_);
    free_v(d_t2_);
    free_v(d_t3_);
    free_v(d_H_hist_);
    free_v(d_e_hist_);
    if (d_c_) {
        cudaFree(d_c_);
        d_c_ = nullptr;
    }
    free_v(d_sy_work_);
    if (d_info_) {
        cudaFree(d_info_);
        d_info_ = nullptr;
    }
    stream_ = nullptr;
    nk_ = n_ = max_m_ = max_vectors_ = n_occ_ = 0;
    hist_count_ = hist_start_ = 0;
    sy_lwork_ = 0;
    use_diis_ = false;
    built_ = false;
    m_k_.clear();
    weights_.clear();
    stage_.clear();
}

void KpointDiisEigenDevice::upload_SX_k(int ik, const Cplx* S_host,
                                        const Cplx* X_host, int m_k) {
    ensure_built_();
    ensure_ik_(ik);
    if (m_k < 1 || m_k > max_m_)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::upload_SX_k: m_k out of range");
    m_k_[static_cast<size_t>(ik)] = m_k;
    host_rm_to_dev_cm_(S_host, d_S_[static_cast<size_t>(ik)], n_, n_);
    host_rm_to_dev_cm_(X_host, d_X_[static_cast<size_t>(ik)], n_, m_k);
}

void KpointDiisEigenDevice::upload_P_k(int ik, const Cplx* P_host) {
    ensure_built_();
    ensure_ik_(ik);
    host_rm_to_dev_cm_(P_host, d_P_[static_cast<size_t>(ik)], n_, n_);
}

void KpointDiisEigenDevice::download_P_k(int ik, Cplx* P_host) const {
    ensure_built_();
    ensure_ik_(ik);
    dev_cm_to_host_rm_(d_P_[static_cast<size_t>(ik)], P_host, n_, n_);
}

void KpointDiisEigenDevice::reset() {
    hist_count_ = 0;
    hist_start_ = 0;
}

void KpointDiisEigenDevice::form_error_k_(int ik) {
    auto handle = static_cast<cublasHandle_t>(cublas_);
    const int n = n_;
    const int m = m_k_[static_cast<size_t>(ik)];
    if (m < 1)
        throw std::logic_error(
            "KpointDiisEigenDevice: upload_SX_k required before residual");
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    const cuDoubleComplex neg = make_cuDoubleComplex(-1.0, 0.0);

    auto* d_S = as_z(d_S_[static_cast<size_t>(ik)]);
    auto* d_X = as_z(d_X_[static_cast<size_t>(ik)]);
    auto* d_P = as_z(d_P_[static_cast<size_t>(ik)]);
    auto* d_H = as_z(d_H_[static_cast<size_t>(ik)]);
    auto* d_err = as_z(d_err_[static_cast<size_t>(ik)]);
    auto* t1 = as_z(d_t1_);
    auto* t2 = as_z(d_t2_);
    auto* t3 = as_z(d_t3_);

    // t1 = P S
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_P, n, d_S, n, &zero, t1, n),
                 "zgemm PS");
    // t2 = H (P S)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_H, n, t1, n, &zero, t2, n),
                 "zgemm HPS");
    // t3 = P H
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_P, n, d_H, n, &zero, t3, n),
                 "zgemm PH");
    // t1 = S (P H)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                             d_S, n, t3, n, &zero, t1, n),
                 "zgemm SPH");
    // t2 = HPS − SPH
    check_cublas(cublasZaxpy(handle, n * n, &neg, t1, 1, t2, 1),
                 "zaxpy HPS-SPH");
    // t1 = (HPS−SPH) X   (n×m)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one,
                             t2, n, d_X, n, &zero, t1, n),
                 "zgemm CX");
    // err = X† t1   (m×m)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, &one,
                             d_X, n, t1, n, &zero, d_err, m),
                 "zgemm X†CX");
}

double KpointDiisEigenDevice::err_norm_from_device_() const {
    double nrm = 0.0;
    for (int ik = 0; ik < nk_; ++ik) {
        const int m = m_k_[static_cast<size_t>(ik)];
        const int mm = m * m;
        stage_.resize(static_cast<size_t>(mm));
        check_cuda(cudaMemcpyAsync(stage_.data(),
                                   d_err_[static_cast<size_t>(ik)],
                                   static_cast<size_t>(mm) * sizeof(Cplx),
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H err");
        check_cuda(cudaStreamSynchronize(stream_), "sync err");
        for (const Cplx& v : stage_)
            nrm = std::max(nrm, std::abs(v));
    }
    return nrm;
}

double KpointDiisEigenDevice::residual(
    const std::vector<const Cplx*>& H_hosts) {
    ensure_built_();
    if (static_cast<int>(H_hosts.size()) != nk_)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::residual: H_hosts.size() must equal nk");
    for (int ik = 0; ik < nk_; ++ik) {
        if (!H_hosts[static_cast<size_t>(ik)])
            throw std::invalid_argument(
                "KpointDiisEigenDevice::residual: null H_hosts[ik]");
        host_rm_to_dev_cm_(H_hosts[static_cast<size_t>(ik)],
                           d_H_[static_cast<size_t>(ik)], n_, n_);
        form_error_k_(ik);
    }
    return err_norm_from_device_();
}

void KpointDiisEigenDevice::ingest_H_k_device_rm(int ik, const void* d_H_rm) {
    ensure_built_();
    ensure_ik_(ik);
    if (d_H_rm == nullptr)
        throw std::invalid_argument(
            "KpointDiisEigenDevice::ingest_H_k_device_rm: null d_H_rm");
    const int nn = n_ * n_;
    const int blocks = grid_size(nn, kDefaultBlock);
    cplx_rm_to_cm_kernel<<<blocks, kDefaultBlock, 0, stream_>>>(
        as_z(d_H_rm), as_z(d_H_[static_cast<size_t>(ik)]), n_, n_);
    check_cuda(cudaGetLastError(), "cplx_rm_to_cm_kernel");
}

double KpointDiisEigenDevice::residual_from_device() {
    ensure_built_();
    for (int ik = 0; ik < nk_; ++ik)
        form_error_k_(ik);
    return err_norm_from_device_();
}

void KpointDiisEigenDevice::diis_extrapolate_() {
    if (!use_diis_)
        return;
    auto handle = static_cast<cublasHandle_t>(cublas_);
    const int nn = n_ * n_;
    const int mm_pad = max_m_ * max_m_;

    int slot;
    if (hist_count_ < max_vectors_) {
        slot = (hist_start_ + hist_count_) % max_vectors_;
        ++hist_count_;
    } else {
        slot = hist_start_;
        hist_start_ = (hist_start_ + 1) % max_vectors_;
    }

    // Push (H_k, e_k) for every k into circular history.
    for (int ik = 0; ik < nk_; ++ik) {
        const size_t h_base =
            (static_cast<size_t>(slot) * static_cast<size_t>(nk_)
             + static_cast<size_t>(ik))
            * static_cast<size_t>(nn);
        const size_t e_base =
            (static_cast<size_t>(slot) * static_cast<size_t>(nk_)
             + static_cast<size_t>(ik))
            * static_cast<size_t>(mm_pad);
        check_cuda(cudaMemcpyAsync(as_z(d_H_hist_) + h_base,
                                   d_H_[static_cast<size_t>(ik)],
                                   static_cast<size_t>(nn) * sizeof(Cplx),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "push H hist");
        const int m = m_k_[static_cast<size_t>(ik)];
        const int mm = m * m;
        // Zero pad then copy m×m residual (col-major lda=m into mm_pad slot).
        check_cuda(cudaMemsetAsync(as_z(d_e_hist_) + e_base, 0,
                                   static_cast<size_t>(mm_pad) * sizeof(Cplx),
                                   stream_),
                   "zero e slot");
        check_cuda(cudaMemcpyAsync(as_z(d_e_hist_) + e_base,
                                   d_err_[static_cast<size_t>(ik)],
                                   static_cast<size_t>(mm) * sizeof(Cplx),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "push e hist");
    }

    const int mh = hist_count_;
    if (mh < 2)
        return; // warm-up: keep latest H

    // B_ij = Σ_k w_k Re Tr(e_i† e_j) = Σ_k w_k Re zdotc(e_i, e_j)
    std::vector<double> Bmm(static_cast<size_t>(mh * mh), 0.0);
    for (int i = 0; i < mh; ++i) {
        const int si = (hist_start_ + i) % max_vectors_;
        for (int j = 0; j <= i; ++j) {
            const int sj = (hist_start_ + j) % max_vectors_;
            double val = 0.0;
            for (int ik = 0; ik < nk_; ++ik) {
                const int m = m_k_[static_cast<size_t>(ik)];
                const int mm = m * m;
                const size_t bi =
                    (static_cast<size_t>(si) * static_cast<size_t>(nk_)
                     + static_cast<size_t>(ik))
                    * static_cast<size_t>(mm_pad);
                const size_t bj =
                    (static_cast<size_t>(sj) * static_cast<size_t>(nk_)
                     + static_cast<size_t>(ik))
                    * static_cast<size_t>(mm_pad);
                cuDoubleComplex zdot = make_cuDoubleComplex(0.0, 0.0);
                check_cublas(cublasZdotc(handle, mm, as_z(d_e_hist_) + bi, 1,
                                         as_z(d_e_hist_) + bj, 1, &zdot),
                             "zdotc e_i e_j");
                val += weights_[static_cast<size_t>(ik)] * cuCreal(zdot);
            }
            Bmm[static_cast<size_t>(i * mh + j)] = val;
            Bmm[static_cast<size_t>(j * mh + i)] = val;
        }
    }
    std::vector<double> c_host;
    if (!solve_pulay_host(Bmm, mh, c_host))
        return; // singular → keep latest H
    check_cuda(cudaMemcpyAsync(d_c_, c_host.data(),
                               static_cast<size_t>(mh) * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "H2D c");
    for (int ik = 0; ik < nk_; ++ik) {
        const int blocks = grid_size(nn, kDefaultBlock);
        weighted_sum_cplx_hist_kernel<<<blocks, kDefaultBlock, 0, stream_>>>(
            as_z(d_H_[static_cast<size_t>(ik)]), as_z(d_H_hist_), d_c_, nn, mh,
            hist_start_, max_vectors_, nk_, ik);
        check_cuda(cudaGetLastError(), "weighted_sum_cplx_hist");
    }
}

void KpointDiisEigenDevice::diagonalize_and_density_k_(int ik) {
    auto handle = static_cast<cublasHandle_t>(cublas_);
    auto sol = static_cast<cusolverDnHandle_t>(cusolver_);
    const int n = n_;
    const int m = m_k_[static_cast<size_t>(ik)];
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    auto* d_X = as_z(d_X_[static_cast<size_t>(ik)]);
    auto* d_H = as_z(d_H_[static_cast<size_t>(ik)]);
    auto* d_C = as_z(d_C_[static_cast<size_t>(ik)]);
    auto* d_Cp = as_z(d_Cp_[static_cast<size_t>(ik)]);
    auto* d_P = as_z(d_P_[static_cast<size_t>(ik)]);
    double* d_eig = d_eig_[static_cast<size_t>(ik)];
    auto* t1 = as_z(d_t1_);

    // t1 = H X  (n×m)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one,
                             d_H, n, d_X, n, &zero, t1, n),
                 "zgemm HX");
    // Cp = X† t1  (m×m)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, &one,
                             d_X, n, t1, n, &zero, d_Cp, m),
                 "zgemm X†HX");

    check_cusolver(cusolverDnZheevd(
                       sol, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m,
                       d_Cp, m, d_eig, as_z(d_sy_work_), sy_lwork_, d_info_),
                   "zheevd");
    int info = 0;
    check_cuda(cudaMemcpyAsync(&info, d_info_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H zheevd info");
    check_cuda(cudaStreamSynchronize(stream_), "sync zheevd info");
    if (info != 0)
        throw std::runtime_error("KpointDiisEigenDevice: zheevd info="
                                 + std::to_string(info) + " at k="
                                 + std::to_string(ik));

    // C = X Cp  (n×m)
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one,
                             d_X, n, d_Cp, m, &zero, d_C, n),
                 "zgemm X Cp");

    // P = occ * C_occ * C_occ†
    if (n_occ_ <= 0) {
        check_cuda(cudaMemsetAsync(d_P, 0,
                                   static_cast<size_t>(n) * n * sizeof(Cplx),
                                   stream_),
                   "zero P");
        return;
    }
    const int nfill = std::min(n_occ_, m);
    const cuDoubleComplex fac = make_cuDoubleComplex(occ_factor_, 0.0);
    check_cublas(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_C, n, n, nfill,
                             &fac, d_C, n, d_C, n, &zero, d_P, n),
                 "zgemm P=Cocc Cocc†");
}

void KpointDiisEigenDevice::advance() {
    ensure_built_();
    diis_extrapolate_();
    for (int ik = 0; ik < nk_; ++ik)
        diagonalize_and_density_k_(ik);
}

double KpointDiisEigenDevice::step(const std::vector<const Cplx*>& H_hosts) {
    const double err_norm = residual(H_hosts);
    advance();
    return err_norm;
}

void KpointDiisEigenDevice::error_to_host(int ik, const Cplx* H_host,
                                          Cplx* e_host_mm) {
    ensure_built_();
    ensure_ik_(ik);
    host_rm_to_dev_cm_(H_host, d_H_[static_cast<size_t>(ik)], n_, n_);
    form_error_k_(ik);
    const int m = m_k_[static_cast<size_t>(ik)];
    // Device err is col-major m×m → host row-major.
    dev_cm_to_host_rm_(d_err_[static_cast<size_t>(ik)], e_host_mm, m, m);
}

void KpointDiisEigenDevice::eigen_to_host(int ik, const Cplx* H_host,
                                          double* eig_host, Cplx* C_host) {
    ensure_built_();
    ensure_ik_(ik);
    host_rm_to_dev_cm_(H_host, d_H_[static_cast<size_t>(ik)], n_, n_);
    diagonalize_and_density_k_(ik);
    const int m = m_k_[static_cast<size_t>(ik)];
    check_cuda(cudaMemcpyAsync(eig_host, d_eig_[static_cast<size_t>(ik)],
                               static_cast<size_t>(m) * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_),
               "D2H eig");
    check_cuda(cudaStreamSynchronize(stream_), "sync eig");
    // C is n×m col-major on device; host wants row-major n×m.
    dev_cm_to_host_rm_(d_C_[static_cast<size_t>(ik)], C_host, n_, m);
}

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV
