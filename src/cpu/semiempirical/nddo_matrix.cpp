/// @file src/cpu/semiempirical/nddo_matrix.cpp
/// @brief Packed matrix conversion and diagonalization for NDDO solvers.

#include "atomiverse/cpu/semiempirical/nddo_matrix.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

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
    coefficients_row_major.resize(static_cast<std::size_t>(n_) * n_);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            const double value = matrix_(i, j);
            if (!std::isfinite(value))
                throw std::runtime_error("NDDO non-finite orbital coefficient");
            coefficients_row_major[static_cast<std::size_t>(i) * n_ + j] =
                value;
        }
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
