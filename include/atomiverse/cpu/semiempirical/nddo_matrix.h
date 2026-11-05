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

    /// @brief One BLAS-3 pseudo-diagonalization step (MOPAC's idea, done as
    ///        matrix products): rotate the orbitals of the previous solve by
    ///        first-order occupied-virtual mixing κ_ia = F_ia / (ε_i − ε_a)
    ///        in that orbital basis, then re-orthonormalize (Cholesky) the
    ///        occupied set and the virtual complement.
    ///
    /// The orbitals are not eigenvectors, but their occupied space moves
    /// toward the one that makes F_ov vanish, which is all the density needs;
    /// at self-consistency F_ov = 0 and the step is the identity.
    /// @param packed Symmetric Fock lower triangle [eV].
    /// @param n_occupied Doubly occupied orbital count, 0 < n_occupied < n.
    /// @param min_gap Smallest admitted ε_a − ε_i [eV]; a smaller gap refuses.
    /// @param coefficients_row_major Rotated orbitals, row-major n by n.
    /// @return false, leaving every output untouched, when no orbitals are
    ///         held yet, a gap is below min_gap, or an orthonormalization
    ///         fails; the caller then diagonalizes.
    bool pseudo_diagonalize_packed(const std::vector<double>& packed,
                                   int n_occupied, double min_gap,
                                   std::vector<double>& coefficients_row_major);

    /// @brief Forget the held orbitals (the next pseudo step refuses).
    void reset_orbitals() noexcept { has_orbitals_ = false; }

  private:
    void publish_row_major_(std::vector<double>& coefficients_row_major) const;

    int n_ = 0;
    bool has_orbitals_ = false;
    std::vector<double> fock_, product_, mixing_, overlap_, rotated_;
    cpu::linalg::RealMatrix matrix_;
    std::vector<double> eigenvalues_;
    std::vector<double> work_;
    std::vector<xt::blas_index_t> iwork_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
