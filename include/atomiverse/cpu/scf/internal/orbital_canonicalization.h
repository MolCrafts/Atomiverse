/// @file cpu/scf/internal/orbital_canonicalization.h
/// @brief Density-preserving canonicalization of occupied/virtual subspaces.
#pragma once
#include "atomiverse/cpu/scf/dense_matrix.h"
namespace ATV::cpu::scf::internal {
/// @brief Rotate occupied and virtual spaces separately to diagonalize C^T F C.
/// @param F Physical AO Fock [Ha], n x n.
/// @param C Metric-orthonormal MO coefficients [n x m], rotated in place.
/// @param eig Physical orbital energies [Ha], length m, overwritten.
/// @param n_occ Occupied count in this channel [0,m].
/// @return Rotated F_ai block [Ha], shape (m-n_occ) x n_occ.
/// @throws std::invalid_argument for incompatible shapes or occupations.
RealMatrix semicanonicalize(const RealMatrix& F, RealMatrix& C, RealVector& eig,
                            int n_occ);
} // namespace ATV::cpu::scf::internal
