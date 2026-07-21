/// @file src/cpu/semiempirical/nddo_matrix.cpp
/// @brief Packed matrix conversion and diagonalization for NDDO solvers.

#include "atomiverse/cpu/semiempirical/nddo_matrix.h"

#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

void validate_packed(int n, const std::vector<double>& packed) {
    const std::size_t expected =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) / 2u;
    if (n <= 0 || packed.size() != expected) {
        throw std::runtime_error(
            "packed NDDO matrix size does not match dimension");
    }
}

} // namespace

cpu::linalg::RealMatrix
NDDOMatrixBuilder::packed_to_dense(const std::vector<double>& packed,
                                   int n) const {
    validate_packed(n, packed);
    cpu::linalg::RealMatrix dense = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            const double value = packed[packed_index(i, j)];
            dense(i, j) = value;
            dense(j, i) = value;
        }
    }
    return dense;
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

NDDOOrbitalResult
NDDODiagonalizer::diagonalize_packed(const std::vector<double>& packed,
                                     int n) const {
    const auto dense = NDDOMatrixBuilder{}.packed_to_dense(packed, n);
    auto [eig, coeff] = xt::linalg::eigh(dense);

    NDDOOrbitalResult result;
    result.eigenvalues.resize(static_cast<std::size_t>(n));
    result.coefficients_row_major.resize(static_cast<std::size_t>(n)
                                         * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        result.eigenvalues[static_cast<std::size_t>(i)] = eig(i);
        for (int j = 0; j < n; ++j) {
            result.coefficients_row_major[static_cast<std::size_t>(i)
                                              * static_cast<std::size_t>(n)
                                          + static_cast<std::size_t>(j)] =
                coeff(i, j);
        }
    }
    return result;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
