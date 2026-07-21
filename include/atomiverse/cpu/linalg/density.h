/// @file cpu/linalg/density.h
/// @brief Density matrix from occupied molecular orbitals: P = factor C_occ C_occ^T.
///        Method-agnostic — shared by `cpu::scf` and `cpu::gpw`.
#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace linalg {

/// @brief Build density matrix P = factor * C_occ * C_occ^T.
/// @param C MO coefficient matrix [n x n].
/// @param n_occ Number of occupied orbitals.
/// @param n Basis dimension.
/// @param factor Occupancy: 2 for RHF (closed-shell), 1 per spin (UHF/UKS).
inline RealMatrix build_density_matrix(const RealMatrix& C, int n_occ, int n,
                                       double factor = 1.0) {
    if (n_occ == 0)
        return xt::zeros<double>({n, n});
    auto C_occ = xt::view(C, xt::all(), xt::range(0, n_occ));
    return factor * xt::linalg::dot(C_occ, xt::transpose(C_occ));
}

} // namespace linalg
} // namespace cpu
} // namespace ATV
