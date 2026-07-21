/// @file cuda/scf/cublas_dispatch.cuh
/// @brief Precision-dispatched cuBLAS/cuSOLVER wrappers for SCF components.
///
/// Avoids if-constexpr-in-non-template issues: only the matching overload compiles.
#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "atomiverse/core/real.h"

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace scf {
namespace detail {

// ── cuBLAS axpy ──

inline cublasStatus_t axpy(cublasHandle_t h, int n, const float* a,
                           const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(h, n, a, x, incx, y, incy);
}
inline cublasStatus_t axpy(cublasHandle_t h, int n, const double* a,
                           const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(h, n, a, x, incx, y, incy);
}

// ── cuBLAS nrm2 ──

inline cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, int incx,
                           float* result) {
    return cublasSnrm2(h, n, x, incx, result);
}
inline cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, int incx,
                           double* result) {
    return cublasDnrm2(h, n, x, incx, result);
}

// ── cuBLAS dot ──

inline cublasStatus_t dot(cublasHandle_t h, int n, const float* x, int incx,
                          const float* y, int incy, float* result) {
    return cublasSdot(h, n, x, incx, y, incy, result);
}
inline cublasStatus_t dot(cublasHandle_t h, int n, const double* x, int incx,
                          const double* y, int incy, double* result) {
    return cublasDdot(h, n, x, incx, y, incy, result);
}

// ── cuBLAS gemm ──

inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C,
                           int ldc) {
    return cublasSgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const double* alpha, const double* A, int lda,
                           const double* B, int ldb, const double* beta,
                           double* C, int ldc) {
    return cublasDgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ── cuSOLVER syevd buffer size ──

inline cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                         cusolverEigMode_t mode,
                                         cublasFillMode_t fill, int n, float* A,
                                         int lda, float* W, int* lwork) {
    return cusolverDnSsyevd_bufferSize(h, mode, fill, n, A, lda, W, lwork);
}
inline cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                         cusolverEigMode_t mode,
                                         cublasFillMode_t fill, int n,
                                         double* A, int lda, double* W,
                                         int* lwork) {
    return cusolverDnDsyevd_bufferSize(h, mode, fill, n, A, lda, W, lwork);
}

// ── cuSOLVER syevd ──

inline cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t mode,
                              cublasFillMode_t fill, int n, float* A, int lda,
                              float* W, float* work, int lwork, int* info) {
    return cusolverDnSsyevd(h, mode, fill, n, A, lda, W, work, lwork, info);
}
inline cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t mode,
                              cublasFillMode_t fill, int n, double* A, int lda,
                              double* W, double* work, int lwork, int* info) {
    return cusolverDnDsyevd(h, mode, fill, n, A, lda, W, work, lwork, info);
}

} // namespace detail
} // namespace scf
} // namespace cuda
} // namespace ATV
