/// @file cpu/linalg/internal/blas.h
/// @brief **PRIVATE** Fortran BLAS level-3 bindings on raw column-major
///        buffers, for contractions whose operands are reinterpreted slices
///        of larger tensors (no xtensor view can express them without a copy).
///
/// Thread model: called outside OpenMP regions the OpenMP-flavour OpenBLAS
/// threads internally; called inside a region it runs on the calling thread.
/// Never set OMP_PROC_BIND / OMP_PLACES around these (see
/// .claude/notes/openblas-thread-binding.md).
#pragma once

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
}

namespace ATV::cpu::linalg::internal {

/// C = alpha op(A) op(B) + beta C, column-major (BLAS dgemm).
inline void gemm(char ta, char tb, int m, int n, int k, double alpha,
                 const double* a, int lda, const double* b, int ldb,
                 double beta, double* c, int ldc) {
    if (m <= 0 || n <= 0)
        return;
    dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

/// C = alpha A Aᵀ + beta C (trans = 'N') or alpha Aᵀ A + beta C ('T'),
/// lower triangle of column-major C written (BLAS dsyrk).
inline void syrk_lower(char trans, int n, int k, double alpha, const double* a,
                       int lda, double beta, double* c, int ldc) {
    if (n <= 0)
        return;
    const char uplo = 'L';
    dsyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

} // namespace ATV::cpu::linalg::internal
