/// @file cpu/scf/internal/density_fitting.h
/// @brief internal::DensityFitting — RI-J / RI-JK workspace over the def2
///        universal JKFIT auxiliary basis.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Robust Coulomb-metric density fitting (resolution of the identity):
///   (μν|λσ) ≈ Σ_PQ (μν|P) [V⁻¹]_PQ (Q|λσ),   V_PQ = (P|Q),
/// which makes the Coulomb-metric fit error in J second order in the
/// fitting residual (Vahtras, Almlöf, Feyereisen, Chem. Phys. Lett. 213,
/// 514 (1993), DOI 10.1016/0009-2614(93)89151-7). The exchange fit uses
/// the same auxiliary expansion through the half-transformed B tensor
/// (Weigend, PCCP 4, 4285 (2002), DOI 10.1039/B204199P); J-fit accuracy
/// discussion: Weigend, PCCP 8, 1057 (2006), DOI 10.1039/b515623h.
///
/// The workspace lives in the CARTESIAN engine AO space (the same space
/// `ShellBlockPairList` works in): the constructor builds
///   * (μν|P): 3-center Coulomb integrals via `rys_eri_block` with a
///     unit zero-exponent s dummy shell on the 4th center (coefficient 1,
///     unnormalized — a zero-exponent s Gaussian is the constant 1, so
///     the 4-center kernel collapses to the 3-center integral; the same
///     trick PySCF/libint use),
///   * V_PQ: the 2-center Coulomb metric (dummy shells on both sides),
///   * the Cholesky factorization V = L·Lᵀ and its whitening factor
///     W = L⁻¹ (W·V·Wᵀ = I, V⁻¹ = Wᵀ·W) — the "V^{-1/2}" of the RI-K
///     B route; a near-singular metric throws per `lindep_threshold`,
///   * B^P_μν = Σ_Q W_PQ (μν|Q), stored (n_aux × n_ao²).
///
/// Auxiliary angular convention: def2-universal-jkfit is a SPHERICAL
/// basis family, so the auxiliary index P is transformed to real
/// spherical harmonics (2l+1 per shell, self-normalized against the
/// aux Cartesian overlap exactly like the orbital `IntegralPack::c2s`).
/// This matters beyond bookkeeping: a Cartesian d/f/g aux shell spans
/// extra lower-l functions (e.g. the x²+y²+z² combination of a d shell),
/// so fitting in the raw Cartesian aux span — while still variational —
/// would NOT reproduce PySCF's def2-universal-jkfit numbers. The orbital
/// indices μν stay Cartesian; `HamiltonianBuilder` bridges the fitted
/// J/K back to the working (possibly spherical) AO dimension through its
/// existing `to_ao` sandwich.
///
/// All quantities in atomic units (Bohr, Hartree).
#pragma once

#include <vector>

#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

class DensityFitting {
  public:
    /// @brief Build the full RI workspace: (μν|P), V_PQ, Cholesky
    ///        whitening factor, and the contracted B tensor.
    /// @param orbital_shells Cartesian engine AO shells (one entry per
    ///        Cartesian component, `IntegralPack::shells` layout);
    ///        centers in Bohr.
    /// @param atoms Atom descriptors (Z + position in Bohr) — the
    ///        auxiliary shells are placed on these centers.
    /// @param aux Auxiliary basis identifier; an atom without a table
    ///        row throws std::runtime_error (no fallback).
    /// @param lindep_threshold Relative metric linear-dependence bound
    ///        (dimensionless): the Cholesky of V throws when
    ///        min(diag L)² < lindep_threshold · max(diag L)² (pivot² is
    ///        the metric-eigenvalue scale).
    DensityFitting(const std::vector<native::Shell>& orbital_shells,
                   const std::vector<native::Atom>& atoms, AuxBasis aux,
                   double lindep_threshold = 1e-10);

    /// @brief Cartesian engine AO dimension of the orbital index.
    /// @return n_ao (dimensionless count).
    int n_ao() const { return n_ao_; }

    /// @brief Spherical auxiliary dimension (Σ 2l+1 over aux shells).
    /// @return n_aux (dimensionless count).
    int n_aux() const { return n_aux_; }

    /// @brief The 2-center Coulomb metric V_PQ = (P|Q) in the spherical
    ///        auxiliary basis [Hartree].
    const RealMatrix& metric() const { return V_; }

    /// @brief The W-contracted 3-center tensor B^P_μν, stored
    ///        (n_ao² × n_aux) with the (μ·n_ao + ν) pair index on the
    ///        rows [Hartree^1/2] — `RealMatrix` is column-major, so each
    ///        auxiliary column is one contiguous n_ao² slice (the M_P
    ///        matrix). Σ_P B_{μν,P} B_{λσ,P} is the fitted (μν|λσ).
    ///        Exposed for unit-level inspection.
    const RealMatrix& b_tensor() const { return B_; }

    /// @brief The whitening factor W = L⁻¹ of V = L·Lᵀ: lower-triangular,
    ///        W·V·Wᵀ = I and V⁻¹ = Wᵀ·W [Hartree^-1/2]. This is the
    ///        Cholesky realization of "V^{-1/2}" in the RI-K B route —
    ///        any factor with W·V·Wᵀ = I yields the identical fit.
    const RealMatrix& v_inv_sqrt() const { return W_; }

    /// @brief Fitted Coulomb matrix J[P] via the γ_P route:
    ///        γ = V⁻¹·((μν|Q)·P), J_μν = Σ_P (μν|P) γ_P — evaluated as
    ///        J = Bᵀ·(B·vec(P)) since V⁻¹ = Wᵀ·W.
    /// @param P_total Total density matrix in the Cartesian engine AO
    ///        space [n_ao × n_ao, dimensionless] (RHF: includes the
    ///        factor 2).
    /// @return Fitted Coulomb matrix [n_ao × n_ao, Hartree].
    RealMatrix j_fitted(const RealMatrix& P_total) const;

    /// @brief Fitted exchange matrix via the B half-transform:
    ///        B^P_μi = Σ_ν B^P_μν C_νi, K_μν = Σ_{P,i} B^P_μi B^P_νi.
    /// @param C_occ Occupied orbital coefficients (or any factor with
    ///        P = C·Cᵀ) [n_ao × n_occ, dimensionless].
    /// @return Fitted exchange K[C·Cᵀ] [n_ao × n_ao, Hartree]; with the
    ///         total RHF density factorization (C√occ scaling included)
    ///         this matches `ShellBlockPairList::k`'s normalization
    ///         (F = Hcore + J − 0.5·K for RHF).
    RealMatrix k_fitted(const RealMatrix& C_occ) const;

    /// @brief Fitted exchange for a density without factor form: P is
    ///        eigen-decomposed P = Σ_i λ_i v_i v_iᵀ and K accumulates
    ///        sign(λ_i)·K[√|λ_i| v_i] — exact for any symmetric P, and
    ///        the route the Fock build uses (mid-SCF densities from
    ///        MINAO / damping / ASPC need not be idempotent or carry
    ///        coefficients).
    /// @param P Symmetric density matrix in the Cartesian engine AO
    ///        space [n_ao × n_ao, dimensionless].
    /// @return Fitted exchange K[P] [n_ao × n_ao, Hartree].
    RealMatrix k_fitted_from_density(const RealMatrix& P) const;

  private:
    int n_ao_ = 0;
    int n_aux_ = 0;
    double lindep_threshold_;
    RealMatrix V_; ///< (n_aux × n_aux) spherical Coulomb metric [Hartree].
    RealMatrix W_; ///< (n_aux × n_aux) whitening factor L⁻¹ [Hartree^-1/2].
    RealMatrix B_; ///< (n_ao² × n_aux) W-contracted 3-center tensor
                   ///< [Hartree^1/2]; column-major ⇒ contiguous per-P
                   ///< slices.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
