/// @file cpu/scf/solve.h
/// @brief CPU SCF solver using xtensor + xtensor-blas.
///
/// Standalone reference implementation for correctness verification
/// and CPU/CUDA parity tests. Uses xt::linalg for eigendecomposition
/// and matrix operations -- no manual linear algebra.
///
/// @see the concrete method (cuda::scf::RHF) private cycle for the CUDA inner-solver implementation
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace toy {

/// @brief Toy SCF solver result (synthetic reference solver only).
///
/// Lives in `scf::toy` so it cannot collide with the former production
/// `ATV::cpu::scf::SCFResult` in `RhfScfCycle::drive.h` (same name in the same
/// namespace would be an ODR violation across the library).
struct SCFResult {
    bool converged = false; ///< Whether SCF converged within max_iter.
    int iterations = 0;     ///< Number of iterations performed.
    double energy = 0;      ///< Final SCF energy = trace(H * P).
    double residual = 0;    ///< Final ||P_new - P_old||_F.
    xt::xtensor<double, 1> eigenvalues; ///< Final eigenvalues (ascending).
    xt::xtensor<double, 2> density;     ///< Final density matrix P.
};

/// @brief Run the toy SCF loop on CPU using xtensor-blas.
///
/// Algorithm: H(P) = H0 + alpha * P, iteratively diagonalize H,
/// rebuild P from occupied eigenvectors, check convergence.
///
/// H0 is a deterministic diagonally-dominant symmetric matrix:
///   H0[i,j] = 1 / (1 + |i-j|)  for off-diagonal
///   H0[i,i] = n + 1 - i          for diagonal
///
/// @param n Matrix dimension.
/// @param n_occ Number of occupied orbitals.
/// @param alpha Density coupling parameter.
/// @param tol Convergence tolerance (Frobenius norm of density change).
/// @param max_iter Maximum number of SCF iterations.
/// @return SCFResult with convergence info, energy, eigenvalues, density.
SCFResult solve(int n, int n_occ, double alpha, double tol, int max_iter);

} // namespace toy
} // namespace scf
} // namespace cpu
} // namespace ATV
