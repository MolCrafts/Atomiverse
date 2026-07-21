/// @file cpu/semiempirical/nddo_matrix.h
/// @brief Packed matrix conversion and diagonalization for NDDO solvers.
#pragma once

#include <vector>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOOrbitalResult {
    std::vector<double> eigenvalues;
    std::vector<double> coefficients_row_major;
};

/// @brief Matrix utilities for NDDO reference packed lower-triangle storage.
class NDDOMatrixBuilder {
  public:
    cpu::linalg::RealMatrix packed_to_dense(const std::vector<double>& packed,
                                            int n) const;
    std::vector<double>
    dense_to_packed(const cpu::linalg::RealMatrix& dense) const;
};

/// @brief Diagonalizes a packed symmetric NDDO Fock matrix.
class NDDODiagonalizer {
  public:
    NDDOOrbitalResult diagonalize_packed(const std::vector<double>& packed,
                                         int n) const;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
