/// @file cpu/semiempirical/nddo_matrix.h
/// @brief Packed matrix conversion and diagonalization for NDDO solvers.
#pragma once

#include <vector>

#include <xtensor-blas/xblas_config.hpp>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Matrix utilities for NDDO reference packed lower-triangle storage.
class NDDOMatrixBuilder {
  public:
    /// @brief Expand packed symmetric values into caller-owned column-major storage.
    /// @param packed Lower triangle in row order.
    /// @param n Positive matrix dimension.
    /// @param dense Already-sized n by n matrix, overwritten without allocation.
    void packed_to_dense(const std::vector<double>& packed, int n,
                         cpu::linalg::RealMatrix& dense) const;
    std::vector<double>
    dense_to_packed(const cpu::linalg::RealMatrix& dense) const;
};

/// @brief Diagonalizes a packed symmetric NDDO Fock matrix.
class NDDODiagonalizer {
  public:
    /// @brief Allocate/query the existing LAPACK syevd workspace for this size.
    /// @param n Positive matrix dimension within LAPACK workspace integer limits.
    void build(int n);

    /// @brief Solve with retained buffers using syevd('V', 'L').
    /// @param packed Symmetric Fock lower triangle [eV].
    /// @param coefficients_row_major Eigenvectors in columns, row-major n by n.
    ///        Existing capacity is reused; the owner reserves this at build.
    /// @throws std::runtime_error for unbuilt state, non-finite data, or solver failure.
    void diagonalize_packed(const std::vector<double>& packed,
                            std::vector<double>& coefficients_row_major);

  private:
    int n_ = 0;
    cpu::linalg::RealMatrix matrix_;
    std::vector<double> eigenvalues_;
    std::vector<double> work_;
    std::vector<xt::blas_index_t> iwork_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
