/// @file cpu/scf/internal/eigen_solver.h
/// @brief internal::EigenSolver — generalized symmetric eigensolve (F C = S C ε).
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Thin stateful wrapper around `diagonalize_fock_generalized` (LAPACK
/// dsygvd via xtensor-blas). Held by the diagonalize stage inside the
/// driver-ized RHF/UHF/RKS/UKS loop so future LAPACK workspace
/// pre-allocation can land in `prepare()` without touching call sites.
#pragma once

#include <utility>

#include "atomiverse/cpu/linalg/eigensolve.h" // diagonalize_fock_generalized
#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

class EigenSolver {
  public:
    EigenSolver() = default;

    /// @brief Pre-allocate workspaces sized for `n_basis`. Currently a
    /// no-op (xtensor-blas allocates LAPACK workspaces internally per
    /// call); reserved for Phase 4 in-place dsygvd integration.
    void prepare(int /*n_basis*/) {}

    /// @brief Solve F C = S C ε.
    /// @return (eigenvalues [n], eigenvectors [n×n]).
    std::pair<RealVector, RealMatrix> step(const RealMatrix& F,
                                           const RealMatrix& S) const {
        return linalg::diagonalize_fock_generalized(F, S);
    }

    /// @brief Two-step path for a (possibly rectangular) orthogonalization
    ///        matrix: F' = XᵀFX (m×m) → eigh → C = X C' (n×m). Used when
    ///        canonical orthogonalization dropped near-linear-dependent
    ///        columns and the dsygvd Cholesky path would be unstable.
    /// @param F Fock matrix [Hartree], n × n.
    /// @param X Orthogonalization transform (dimensionless), n × m.
    /// @return (eigenvalues [m], AO MO coefficients [n×m]).
    std::pair<RealVector, RealMatrix>
    step_orthogonal(const RealMatrix& F, const RealMatrix& X) const {
        return linalg::diagonalize_fock(F, X);
    }
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
