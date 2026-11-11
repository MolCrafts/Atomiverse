/// @file src/cpu/semiempirical/nddo_matrix.cpp
/// @brief Packed matrix conversion and diagonalization for NDDO solvers.

#include "atomiverse/cpu/semiempirical/nddo_matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

extern "C" {
// Fortran BLAS/LAPACK, the vocabulary of cpu/linalg/orbital_tangent.cpp.
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
             int* info);
void dtrsm_(const char* side, const char* uplo, const char* transa,
            const char* diag, const int* m, const int* n, const double* alpha,
            const double* a, const int* lda, double* b, const int* ldb);
}

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

/// n² element loops run on the OpenMP threads from this many orbitals on.
constexpr int kParallelOrbitals = 256;

void validate_packed(int n, const std::vector<double>& packed) {
    const std::size_t expected =
        static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1u) / 2u;
    if (n <= 0 || packed.size() != expected) {
        throw std::runtime_error(
            "packed NDDO matrix size does not match dimension");
    }
}

} // namespace

void NDDOMatrixBuilder::packed_to_dense(const std::vector<double>& packed,
                                        int n,
                                        cpu::linalg::RealMatrix& dense) const {
    validate_packed(n, packed);
    if (dense.shape(0) != static_cast<std::size_t>(n)
        || dense.shape(1) != static_cast<std::size_t>(n))
        throw std::runtime_error(
            "NDDO dense output size does not match dimension");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            const double value = packed[packed_index(i, j)];
            dense(i, j) = value;
            dense(j, i) = value;
        }
    }
}

std::vector<double>
NDDOMatrixBuilder::dense_to_packed(const cpu::linalg::RealMatrix& dense) const {
    if (dense.dimension() != 2 || dense.shape(0) != dense.shape(1)) {
        throw std::runtime_error("dense NDDO matrix must be square");
    }
    const int n = static_cast<int>(dense.shape(0));
    std::vector<double> packed(static_cast<std::size_t>(n)
                                   * static_cast<std::size_t>(n + 1) / 2u,
                               0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            packed[packed_index(i, j)] = dense(i, j);
        }
    }
    return packed;
}

void NDDODiagonalizer::build(int n) {
    // syevd('V') needs at least 1 + 6*n + 2*n*n double workspace entries.
    // Reject dimensions that overflow the same integer ABI as xtensor-blas.
    if (n <= 0
        || 1.0 + 6.0 * n + 2.0 * n * n > static_cast<double>(
               std::numeric_limits<xt::blas_index_t>::max()))
        throw std::invalid_argument(
            "NDDO diagonalizer dimension exceeds LAPACK limits");
    if (n_ == n)
        return;
    // Publish the usable size only after every allocation and query succeeds.
    n_ = 0;
    has_orbitals_ = false;
    const auto dim = static_cast<std::size_t>(n);
    matrix_.resize({dim, dim});
    eigenvalues_.resize(dim);
    const auto lapack_n = static_cast<xt::blas_index_t>(n);
    double work_query = 0.0;
    xt::blas_index_t iwork_query = 0;
    const auto query = static_cast<xt::blas_index_t>(-1);
    const int info = cxxlapack::syevd<xt::blas_index_t>(
        'V', 'L', lapack_n, matrix_.data(), lapack_n, eigenvalues_.data(),
        &work_query, query, &iwork_query, query);
    if (info != 0 || !std::isfinite(work_query) || work_query < 1.0
        || work_query > static_cast<double>(
               std::numeric_limits<xt::blas_index_t>::max())
        || std::trunc(work_query) != work_query || iwork_query < 1)
        throw std::runtime_error("NDDO syevd workspace query failed, info="
                                 + std::to_string(info));
    work_.resize(static_cast<std::size_t>(work_query));
    iwork_.resize(static_cast<std::size_t>(iwork_query));
    n_ = n;
}

void NDDODiagonalizer::diagonalize_packed(
    const std::vector<double>& packed,
    std::vector<double>& coefficients_row_major) {
    if (n_ <= 0)
        throw std::runtime_error("NDDO diagonalizer used before build");
    validate_packed(n_, packed);
    for (double value : packed)
        if (!std::isfinite(value))
            throw std::runtime_error("NDDO non-finite Fock matrix");
    NDDOMatrixBuilder{}.packed_to_dense(packed, n_, matrix_);
    const auto n = static_cast<xt::blas_index_t>(n_);
    // Exactly the former xt::linalg::eigh algorithm, layout, triangle and
    // queried workspace sizes, without its per-iteration matrix/work copies.
    const int info = cxxlapack::syevd<xt::blas_index_t>(
        'V', 'L', n, matrix_.data(), n, eigenvalues_.data(), work_.data(),
        static_cast<xt::blas_index_t>(work_.size()), iwork_.data(),
        static_cast<xt::blas_index_t>(iwork_.size()));
    if (info != 0)
        throw std::runtime_error("NDDO syevd failed, info="
                                 + std::to_string(info));
    for (double value : eigenvalues_)
        if (!std::isfinite(value))
            throw std::runtime_error("NDDO non-finite orbital energy");
    has_orbitals_ = true;
    publish_row_major_(coefficients_row_major);
}

void NDDODiagonalizer::publish_row_major_(
    std::vector<double>& coefficients_row_major) const {
    coefficients_row_major.resize(static_cast<std::size_t>(n_) * n_);
    const int n = n_;
    const double* c = matrix_.data(); // Column-major.
    bool finite = true;
#pragma omp parallel for schedule(static)                                      \
    reduction(&& : finite) if (n >= kParallelOrbitals)
    for (int i = 0; i < n; ++i) {
        double* row =
            coefficients_row_major.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            const double value = c[i + static_cast<std::size_t>(j) * n];
            finite = finite && std::isfinite(value);
            row[j] = value;
        }
    }
    if (!finite)
        throw std::runtime_error("NDDO non-finite orbital coefficient");
}

bool NDDODiagonalizer::pseudo_diagonalize_packed(
    const std::vector<double>& packed, int n_occupied, double min_gap,
    std::vector<double>& coefficients_row_major) {
    if (n_ <= 0 || !has_orbitals_ || n_occupied <= 0 || n_occupied >= n_)
        return false;
    validate_packed(n_, packed);
    const int n = n_, no = n_occupied, nv = n_ - n_occupied;
    const auto nn = static_cast<std::size_t>(n) * n;
    fock_.resize(nn);
    product_.resize(nn);
    rotated_.resize(nn);
    mixing_.resize(static_cast<std::size_t>(no) * nv);
    bool finite = true;
#pragma omp parallel for schedule(static)                                      \
    reduction(&& : finite) if (n >= kParallelOrbitals)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            const double value = packed[packed_index(i, j)];
            finite = finite && std::isfinite(value);
            fock_[static_cast<std::size_t>(i) * n + j] =
                fock_[static_cast<std::size_t>(j) * n + i] = value;
        }
    if (!finite)
        throw std::runtime_error("NDDO non-finite Fock matrix");
    const double* c = matrix_.data(); // Column-major orbitals, ld n.
    const double* c_vir = c + static_cast<std::size_t>(no) * n;
    const double one = 1.0, zero = 0.0, minus = -1.0;
    const char no_t = 'N', trans = 'T', upper = 'U', right = 'R';
    // G = F C, then ε_p = c_pᵀ F c_p and F_ia = C_occᵀ G_vir.
    dgemm_(&no_t, &no_t, &n, &n, &n, &one, fock_.data(), &n, c, &n, &zero,
           product_.data(), &n);
    eigenvalues_.resize(static_cast<std::size_t>(n));
    for (int p = 0; p < n; ++p) {
        double e = 0.0;
        const double* cp = c + static_cast<std::size_t>(p) * n;
        const double* gp = product_.data() + static_cast<std::size_t>(p) * n;
        for (int mu = 0; mu < n; ++mu)
            e += cp[mu] * gp[mu];
        eigenvalues_[static_cast<std::size_t>(p)] = e;
    }
    dgemm_(&trans, &no_t, &no, &nv, &n, &one, c, &n,
           product_.data() + static_cast<std::size_t>(no) * n, &n, &zero,
           mixing_.data(), &no);
    for (int a = 0; a < nv; ++a)
        for (int i = 0; i < no; ++i) {
            const double gap = eigenvalues_[static_cast<std::size_t>(no + a)]
                               - eigenvalues_[static_cast<std::size_t>(i)];
            if (!(gap >= min_gap))
                return false;
            mixing_[static_cast<std::size_t>(a) * no + i] /= -gap;
        }
    // C_occ' = C_occ + C_vir κᵀ ; C_vir' = C_vir − C_occ κ.
    std::copy(c, c + nn, rotated_.begin());
    double* r_occ = rotated_.data();
    double* r_vir = rotated_.data() + static_cast<std::size_t>(no) * n;
    dgemm_(&no_t, &trans, &n, &no, &nv, &one, c_vir, &n, mixing_.data(), &no,
           &one, r_occ, &n);
    dgemm_(&no_t, &no_t, &n, &nv, &no, &minus, c, &n, mixing_.data(), &no, &one,
           r_vir, &n);
    // Orthonormalize the occupied set: C_occ'' = C_occ' U⁻¹, Uᵀ U = C'ᵀ C'.
    auto orthonormalize = [&](double* block, int k) {
        overlap_.resize(static_cast<std::size_t>(k) * k);
        dsyrk_(&upper, &trans, &k, &n, &one, block, &n, &zero, overlap_.data(),
               &k);
        int info = 0;
        dpotrf_(&upper, &k, overlap_.data(), &k, &info);
        if (info != 0)
            return false;
        dtrsm_(&right, &upper, &no_t, &no_t, &n, &k, &one, overlap_.data(), &k,
               block, &n);
        return true;
    };
    if (!orthonormalize(r_occ, no))
        return false;
    // Project the occupied space out of the virtual block, then orthonormalize.
    mixing_.resize(static_cast<std::size_t>(no) * nv);
    dgemm_(&trans, &no_t, &no, &nv, &n, &one, r_occ, &n, r_vir, &n, &zero,
           mixing_.data(), &no);
    dgemm_(&no_t, &no_t, &n, &nv, &no, &minus, r_occ, &n, mixing_.data(), &no,
           &one, r_vir, &n);
    if (!orthonormalize(r_vir, nv))
        return false;
    for (std::size_t q = 0; q < nn; ++q)
        if (!std::isfinite(rotated_[q]))
            return false;
    std::copy(rotated_.begin(), rotated_.end(), matrix_.data());
    publish_row_major_(coefficients_row_major);
    return true;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
