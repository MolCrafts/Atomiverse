/// @file cpu/scf/internal/stability.h
/// @brief **PRIVATE** Seeger-Pople stability analysis over the OrbitalHessian
///        kernel: the lowest eigenvalue of (A + B) per excitation channel —
///        dense eigensolve for small rotation spaces, matrix-free Davidson
///        lowest-root otherwise. Deterministic (fixed start vector, fixed
///        expansion rule, no randomness).
///
/// Classification (real closed-shell reference, Seeger & Pople 1977,
/// DOI 10.1063/1.434318): internal (RHF→RHF) stable ⇔ λ_min[¹(A+B)] ≥ 0;
/// external / triplet (RHF→UHF) stable ⇔ λ_min[³(A+B)] ≥ 0.
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Lowest-root solver of the (A + B) channel blocks of a borrowed
///        OrbitalHessian. Construction is wiring only; each query solves.
class StabilityAnalysis {
  public:
    /// @brief Bind the kernel whose (A + B) channels are analyzed.
    /// @param hessian Orbital Hessian kernel (borrowed; must outlive this).
    explicit StabilityAnalysis(const OrbitalHessian& hessian)
        : hessian_(hessian) {}

    /// @brief Rotation-space size at or below which `lowest` uses the dense
    ///        eigensolve; above it, the matrix-free Davidson path
    ///        (dimensionless; member, not a config bag).
    int dense_threshold = 512;

    /// @brief Lowest eigenvalue of (A + B)_ch — dense for
    ///        n_ov ≤ dense_threshold, Davidson otherwise.
    /// @param ch Excitation channel.
    /// @return λ_min [Hartree].
    double lowest(Channel ch) const;

    /// @brief Normalized eigenvector of λ_min[(A + B)_ch] — the instability
    ///        mode when λ_min < 0 (composite index a·n_occ + i).
    /// @param ch Excitation channel.
    /// @return Unit-norm mode vector, length n_ov (dimensionless).
    RealVector lowest_vector(Channel ch) const;

    /// @brief Dense-path lowest eigenvalue (always builds the n_ov × n_ov
    ///        matrix; parity anchor for the Davidson path).
    /// @param ch Excitation channel.
    /// @return λ_min [Hartree].
    double lowest_dense(Channel ch) const;

    /// @brief Matrix-free Davidson lowest root (σ-vector products only;
    ///        subspace diagonalization via `xt::linalg::eigh`).
    /// @param ch Excitation channel.
    /// @return λ_min [Hartree].
    /// @throws std::runtime_error when Davidson fails to converge within
    ///         the iteration cap (no fallback values).
    double lowest_davidson(Channel ch) const;

  private:
    /// Eigenpair of one solve: value [Hartree] + unit-norm vector.
    struct Root {
        double value = 0.0;
        RealVector vector;
    };

    Root solve_(Channel ch) const;          ///< Dispatch on dense_threshold.
    Root solve_dense_(Channel ch) const;    ///< Explicit build + eigh.
    Root solve_davidson_(Channel ch) const; ///< Matrix-free lowest root.

    const OrbitalHessian& hessian_; ///< Borrowed kernel.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
