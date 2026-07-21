/// @file cpu/scf/internal/tdhf_pack.h
/// @brief **PRIVATE** TDA and TDHF/RPA eigen-solvers over the link-08
///        `OrbitalHessian` A/B kernel — the kernel's fourth configuration
///        (after stability analysis, SOSCF, and CPHF).
///
/// TDA (Hirata & Head-Gordon 1999, DOI 10.1016/S0009-2614(99)01149-5):
/// drop B and solve the Hermitian A X = ω X (CIS on an HF reference), with
/// A·v = ½(σ₊(v) + σ₋(v)) assembled from the kernel's two σ products.
///
/// TDHF/RPA (Dreuw & Head-Gordon 2005, DOI 10.1021/cr0505627; Szabo &
/// Ostlund §7): [[A,B],[−B,−A]](X;Y) = ω(X;Y), solved for real orbitals via
/// the Hermitian reduced form (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = ω² Z. This
/// reduction is well-defined only while (A−B) is positive definite; a
/// negative ω² is exactly the link-08 instability of the channel's (A+B)
/// block (triplet instability ⇒ imaginary ω) and throws — no fallback.
///
/// Small rotation spaces solve densely; above `dense_threshold` both solvers
/// run a deterministic matrix-free Davidson over σ± products only (the RPA
/// path is the reduced-subspace scheme of Stratmann, Scuseria & Frisch,
/// J. Chem. Phys. 109, 8218 (1998), DOI 10.1063/1.477483: (A−B) is
/// eigen-decomposed only in the m-dimensional subspace).
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Lowest-roots excitation solver bound to one channel of a borrowed
///        OrbitalHessian. Construction is wiring only; each call solves.
class TDHFPack {
  public:
    /// @brief Bind the A/B kernel and the excitation channel to solve.
    /// @param hessian Orbital Hessian kernel (borrowed; must outlive this).
    /// @param ch      Excitation channel (singlet or triplet).
    TDHFPack(const OrbitalHessian& hessian, Channel ch)
        : hessian_(hessian), ch_(ch) {}

    /// @brief Rotation-space size at or below which the solvers build the
    ///        dense matrices; above it, the matrix-free Davidson path
    ///        (dimensionless; member, not a config bag —
    ///        `StabilityAnalysis::dense_threshold` precedent).
    int dense_threshold = 512;

    /// @brief Lowest excitation roots of one solve.
    struct Roots {
        /// @brief Excitation energies ω [Hartree], ascending, size n_states.
        RealVector omega;
        /// @brief Excitation amplitudes X, shape [n_states, n_ov], composite
        ///        column index a·n_occ + i (dimensionless). TDA: unit-norm
        ///        rows; RPA: normalized with Y so that XᵀX − YᵀY = 1 per root.
        RealMatrix X;
        /// @brief De-excitation amplitudes Y, shape [n_states, n_ov]
        ///        (dimensionless). Exactly zero for TDA.
        RealMatrix Y;
    };

    /// @brief Tamm-Dancoff lowest roots: A X = ω X with
    ///        A·v = ½(σ₊(v) + σ₋(v)). Real spectrum on any reference
    ///        (A is Hermitian), including instability-carrying ones.
    /// @param n_states Number of lowest roots (1 ≤ n_states ≤ n_ov).
    /// @return {ω [Hartree], X, Y = 0}.
    /// @throws std::invalid_argument when n_states is outside [1, n_ov].
    /// @throws std::runtime_error on Davidson non-convergence (no fallback).
    Roots tda(int n_states) const;

    /// @brief Full TDHF/RPA lowest roots via the Hermitian reduced form
    ///        (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = ω² Z, back-substituted to
    ///        normalized (X, Y) with XᵀX − YᵀY = 1.
    /// @param n_states Number of lowest roots (1 ≤ n_states ≤ n_ov).
    /// @return {ω [Hartree], X, Y}.
    /// @throws std::invalid_argument when n_states is outside [1, n_ov].
    /// @throws std::runtime_error when (A−B) is not positive definite, when
    ///         a requested ω² is negative beyond the numerical-zero guard
    ///         (channel instability ⇒ imaginary ω — the link-08 trap; the
    ///         message carries the offending ω² [Hartree²]), or on Davidson
    ///         non-convergence. No fallback values.
    Roots rpa(int n_states) const;

  private:
    Roots tda_dense_(int n_states) const;     ///< Explicit A build + eigh.
    Roots tda_davidson_(int n_states) const;  ///< Matrix-free block Davidson.
    Roots rpa_dense_(int n_states) const;     ///< Dense reduced form.
    Roots rpa_davidson_(int n_states) const;  ///< Reduced-subspace scheme.
    void check_n_states_(int n_states) const; ///< [1, n_ov] gate (throws).

    const OrbitalHessian& hessian_; ///< Borrowed kernel.
    Channel ch_;                    ///< Bound excitation channel.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
