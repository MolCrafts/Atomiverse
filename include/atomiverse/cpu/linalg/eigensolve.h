/// @file cpu/linalg/eigensolve.h
/// @brief Symmetric / generalized eigenproblem for the SCF Fock matrix:
///        Löwdin orthogonalization X = S^{-1/2} and the solutions of F C = S C ε
///        (both the two-step X^T F X path and the direct LAPACK dsygvd path).
///
/// Method-agnostic — shared by `cpu::scf` and `cpu::gpw`.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/linalg/matrix.h"

extern "C" {
// Direct LAPACK binding for the generalized symmetric eigensolver. We do NOT go
// through xt::linalg::eigh(F, S): its workspace query (LWORK=-1) is broken in the
// LAPACK 3.12.1 bundled with OpenBLAS 0.3.32 (xSYGVD writes WORK(1)=0 on query,
// so the follow-up call aborts with "parameter number 11 had an illegal value").
// Passing the documented workspace sizes explicitly sidesteps the query entirely.
void dsygvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
             double* a, const int* lda, double* b, const int* ldb, double* w,
             double* work, const int* lwork, int* iwork, const int* liwork,
             int* info);
}

namespace ATV {
namespace cpu {
namespace linalg {

/// @brief Canonical orthogonalization of an overlap matrix with
///        eigenvalue thresholding. Direct instantiation — no factory
///        function: construction performs the eigendecomposition.
///
/// Eigendecompose S = U s Uᵀ (dimensionless overlap eigenvalues,
/// ascending), keep the directions with s_i >= tau, and form the
/// rectangular transform X = Ũ s̃^{-1/2} (n × m, m <= n) satisfying
/// XᵀSX = I_m. Near-linear-dependent directions (s_i → 0) are dropped
/// instead of amplified, which is what keeps Roothaan–Hall solvable when
/// the basis approaches linear dependence.
struct CanonicalOrth {
    RealMatrix X;      ///< n × m transform, XᵀSX = I_m (dimensionless).
    int n_dropped = 0; ///< Number of eigen-directions dropped (s_i < tau).
    double s_min = 0;  ///< Raw smallest eigenvalue of S, before filtering.
    /// @brief Löwdin S^{-1/2} = U s^{-1/2} Uᵀ (n × n, dimensionless), built
    ///        from the SAME eigendecomposition so callers that want the
    ///        symmetric orthogonalizer on a healthy basis pay one eigh, not
    ///        two. Populated only when n_dropped == 0 (empty otherwise —
    ///        a filtered basis has no meaningful full-rank S^{-1/2}).
    ///        Bit-identical to `build_orthogonalization_matrix(S)`.
    RealMatrix lowdin;

    /// @brief Eigendecompose and threshold.
    /// @param S   Symmetric overlap matrix (dimensionless), n × n. Read-only.
    /// @param tau Keep-threshold on the raw S eigenvalues (dimensionless);
    ///            s_i >= tau is kept. tau <= 0 disables filtering (all n
    ///            columns kept). PySCF >= 2.13 uses 1e-6 on S eigenvalues;
    ///            Atomiverse's SCF default is 1e-8 (cpu::scf::SCFConfig::lindep_threshold).
    /// @throws std::runtime_error when every column would be dropped
    ///         (no fallback values).
    CanonicalOrth(const RealMatrix& S, double tau) {
        const int n = static_cast<int>(S.shape(0));
        auto [s_eig, U] = xt::linalg::eigh(S); // ascending eigenvalues
        s_min = s_eig(0);

        int first_kept = 0;
        if (tau > 0) {
            while (first_kept < n && s_eig(first_kept) < tau)
                ++first_kept;
        }
        n_dropped = first_kept;
        const int m = n - n_dropped;
        if (m == 0)
            throw std::runtime_error(
                "CanonicalOrth: every overlap eigenvalue below tau="
                + std::to_string(tau) + " (s_max="
                + std::to_string(s_eig(n - 1)) + ") — no basis left");

        X = xt::zeros<double>({n, m});
        for (int j = 0; j < m; ++j) {
            const double inv_sqrt = 1.0 / std::sqrt(s_eig(first_kept + j));
            for (int i = 0; i < n; ++i)
                X(i, j) = U(i, first_kept + j) * inv_sqrt;
        }
        // Healthy basis: X == U s^{-1/2} (column scaling == the diagonal
        // gemm, bitwise — one nonzero product per accumulator), so one
        // more gemm yields the Löwdin matrix from the same factors.
        if (n_dropped == 0)
            lowdin = xt::linalg::dot(X, xt::transpose(U));
    }
};

/// @brief Build the symmetric (Löwdin) orthogonalization matrix X = S^{-1/2}.
inline RealMatrix build_orthogonalization_matrix(const RealMatrix& S) {
    int n = static_cast<int>(S.shape(0));
    auto [s_eig, U] = xt::linalg::eigh(S);
    RealMatrix s_inv_sqrt = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        s_inv_sqrt(i, i) = 1.0 / std::sqrt(s_eig(i));
    return xt::linalg::dot(xt::linalg::dot(U, s_inv_sqrt), xt::transpose(U));
}

/// @brief Diagonalize Fock in the orthogonal basis (two-step path).
/// @return (orbital energies [n], AO MO coefficients [n x n]).
inline std::pair<RealVector, RealMatrix> diagonalize_fock(const RealMatrix& F,
                                                          const RealMatrix& X) {
    RealMatrix F_prime =
        xt::linalg::dot(xt::transpose(X), xt::linalg::dot(F, X));
    auto [eig, C_prime] = xt::linalg::eigh(F_prime);
    RealMatrix C = xt::linalg::dot(X, C_prime);
    return {eig, C};
}

/// @brief Solve the generalized symmetric eigenproblem F C = S C ε (LAPACK dsygvd).
/// @return (orbital energies [n], AO MO coefficients [n x n]).
inline std::pair<RealVector, RealMatrix>
diagonalize_fock_generalized(const RealMatrix& F, const RealMatrix& S) {
    const int n = static_cast<int>(F.shape(0));
    RealMatrix C = F; // dsygvd overwrites A with the eigenvectors
    RealMatrix B = S; // ... and B with its Cholesky factor
    RealVector eig = xt::zeros<double>({n});

    // Documented minimum workspace for ITYPE=1, JOBZ='V' (LAPACK dsygvd):
    // LWORK >= 1 + 6N + 2N^2, LIWORK >= 3 + 5N (the LWORK=-1 query is broken).
    const int itype = 1;
    const char jobz = 'V';
    const char uplo = 'L';
    const int lwork = 1 + 6 * n + 2 * n * n;
    const int liwork = 3 + 5 * n;
    std::vector<double> work(static_cast<std::size_t>(lwork));
    std::vector<int> iwork(static_cast<std::size_t>(liwork));
    int info = 0;

    dsygvd_(&itype, &jobz, &uplo, &n, C.data(), &n, B.data(), &n, eig.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        throw std::runtime_error(
            "diagonalize_fock_generalized: dsygvd failed, info="
            + std::to_string(info));
    return {eig, C};
}

} // namespace linalg
} // namespace cpu
} // namespace ATV
