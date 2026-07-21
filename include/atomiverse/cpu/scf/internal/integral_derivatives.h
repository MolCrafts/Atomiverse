/// @file cpu/scf/integral_derivatives.h
/// @brief Analytic derivatives of molecular integrals w.r.t. nuclear positions.
///
/// Computes dS/dR, dT/dR, dV/dR, d(ERI)/dR, and dV_nuc/dR for contracted
/// Gaussians of any l <= kMaxL (s, p, d shells). Works with any basis the
/// SCF path resolves into an `IntegralPack`.
/// Uses the angular momentum raising formula for derivatives:
///   d/dA_d <a|O|b> = 2*alpha_a * <a+1_d|O|b> - a_d * <a-1_d|O|b>
///
/// All internal computation in atomic units (Bohr, Hartree).
/// Output in Hartree/Bohr.
///
/// @see cpu::scf::native for the undifferentiated integrals
/// @see cpu::scf::internal::ScfGradient for the gradient assembly
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"  // MeanFieldType / SpinType
#include "atomiverse/cpu/scf/slater_determinant.h" // SlaterDeterminant
#include "atomiverse/cpu/scf/xc_functional.h" // XCType

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Derivatives of all molecular integrals w.r.t. nuclear positions.
///
/// Each derivative tensor is indexed as [atom_index, cart_component, mu, nu]
/// where cart_component = 0,1,2 for x,y,z. All in Hartree/Bohr.
struct IntegralDerivPack {
    xt::xtensor<double, 4>
        dS; ///< dS/dR [n_atoms, 3, n_basis, n_basis] [1/Bohr].
    xt::xtensor<double, 4>
        dT; ///< dT/dR [n_atoms, 3, n_basis, n_basis] [Hartree/Bohr].
    xt::xtensor<double, 4>
        dV; ///< dV/dR [n_atoms, 3, n_basis, n_basis] [Hartree/Bohr].
    xt::xtensor<double, 2> dV_nuc; ///< dV_nuc/dR [n_atoms, 3] [Hartree/Bohr].
    int n_atoms = 0;               ///< Number of atoms.
    int n_basis = 0;               ///< Number of basis functions.

    /// @brief Empty pack.
    IntegralDerivPack() = default;

    /// @brief Compute one-electron integral derivatives using the shells
    ///        already resolved inside `pack`.
    ///
    /// The returned dS / dT / dV tensors have the same n_basis dimension as
    /// `pack.S`, so gradient contractions align regardless of basis.
    /// Supports any l ≤ kMaxL — the primitive helpers delegate to
    /// `native::overlap_prim_3d` / `kinetic_prim_3d` / `nuclear_prim`,
    /// which handle s, p, and d shells identically.
    ///
    /// @param pack Integral pack (provides atoms in Bohr + resolved shells).
    explicit IntegralDerivPack(const IntegralPack& pack);
};

/// @brief Contract two-electron derivative integrals directly into the gradient.
///
/// Avoids materializing the full dERI tensor, which scales as O(n_atoms * n_basis^4)
/// in memory. The contraction is performed on the fly using the same analytic
/// primitive derivative formulas as IntegralDerivPack's constructor.
///
/// Works for any basis: the libcint engine handles STO-3G (when compiled in)
/// and the native Rys-quadrature block-derivative engine handles all l ≤ kMaxL.
///
/// Both engines assemble the Coulomb (J) channel always and the exchange (K)
/// channel scaled by the exact-exchange fraction `alpha_hf`: `1.0` for pure
/// HF, `0.0` for a pure (semi-local) functional — in which case the K loop
/// is skipped entirely — and the global-hybrid admixture
/// (`hybrid_exchange_fraction(xc_type)`, 0.20 for B3LYP, 0.25 for
/// PBE0) for a KS hybrid. `alpha_hf` multiplies the existing closed-shell
/// exchange prefactors and density symmetrization as a pure scalar.
///
/// @param[out] grad      Gradient accumulator [n_atoms, 3] in Hartree/Bohr.
/// @param frame          Frame supplying nuclear positions for libcint setup.
/// @param integrals      Integral pack for the current geometry.
/// @param P_total        Total density matrix P^alpha + P^beta (dimensionless).
/// @param det            Converged mean-field wavefunction; supplies the
///                       per-spin factor-1 densities for the unrestricted
///                       exchange contraction.
/// @param method         HF or KS mean field (selects the exchange fraction).
/// @param spin           Restricted or unrestricted spin treatment.
/// @param xc_type        XC functional (KS hybrid exact-exchange fraction).
/// @param force_native_two_electron_deriv Force the native derivative engine
///                       even when libcint is available (test knob).
void accumulate_two_electron_gradient(
    xt::xtensor<double, 2>& grad, const Frame& frame,
    const IntegralPack& integrals, const xt::xtensor<double, 2>& P_total,
    const SlaterDeterminant& det, MeanFieldType method, SpinType spin,
    XCType xc_type, bool force_native_two_electron_deriv);

// ── Second derivatives (hf-capability-gaps-13-cphf) ──────────────────

/// @brief Second derivatives of the one-electron integrals and the nuclear
///        repulsion w.r.t. nuclear positions.
///
/// Built by double application of the angular-momentum raising formula
///   d/dA_d <a|O|b> = 2·alpha_a <a+1_d|O|b> − a_d <a−1_d|O|b>
/// (one center raised by up to +2, or two centers by +1 each), with the
/// remaining center pairs completed by translational invariance
/// (Σ_centers d/dX = 0 applied to each derivative slot). Same shell
/// workspace, ordering, and l ≤ kMaxLOrbital coverage as
/// `IntegralDerivPack`; the shared primitive kernels handle the internal
/// l+2 raise (see native_integrals.cpp buffer sizing).
///
/// All values in atomic units: d2S in 1/Bohr², d2T/d2V/d2V_nuc in
/// Hartree/Bohr².
struct IntegralSecondDerivPack {
    /// d²S/dR_{A,i}dR_{B,j} [n_atoms, 3, n_atoms, 3, n_basis, n_basis]
    /// [1/Bohr²].
    xt::xtensor<double, 6> d2S;
    /// d²T/dR² — same layout [Hartree/Bohr²].
    xt::xtensor<double, 6> d2T;
    /// d²V/dR² — same layout [Hartree/Bohr²].
    xt::xtensor<double, 6> d2V;
    /// d²V_nuc/dR² [n_atoms, 3, n_atoms, 3] [Hartree/Bohr²].
    xt::xtensor<double, 4> d2V_nuc;
    int n_atoms = 0; ///< Number of atoms.
    int n_basis = 0; ///< Number of basis functions.

    /// @brief Empty pack.
    IntegralSecondDerivPack() = default;

    /// @brief Compute all one-electron second-derivative integrals from the
    ///        shells already resolved inside `pack`.
    /// @param pack Integral pack (atoms in Bohr + resolved Cartesian shells).
    /// @throws std::runtime_error on a spherical-convention pack (the
    ///         derivative c2s sandwich is a follow-up link — no fallback).
    explicit IntegralSecondDerivPack(const IntegralPack& pack);
};

/// @brief Accumulate the skeleton two-electron derivative Fock matrices
///        G^{(x)}_{μν} = Σ_{λσ} P_{λσ} [(μν|λσ)^{(x)} − ½(μλ|νσ)^{(x)}]
///        (closed-shell RHF) into `dF` for every nuclear coordinate x.
///
/// The derivative ERIs come from `native::rys_eri_block_deriv` per shell-
/// block quartet (full ordered enumeration — Hessian-assembly scale, run
/// once per `RHF::hessian`), never materialized as a 5-index tensor.
///
/// @param[out] dF     Accumulator [n_atoms, 3, n_basis, n_basis]
///                    [Hartree/Bohr]; the caller pre-fills it with the
///                    one-electron part dT + dV.
/// @param integrals   Integral pack for the current geometry.
/// @param P_total     Total (factor-2 closed-shell) density (dimensionless).
void accumulate_two_electron_fock_derivative(
    xt::xtensor<double, 4>& dF, const IntegralPack& integrals,
    const xt::xtensor<double, 2>& P_total);

/// @brief Accumulate the skeleton two-electron second-derivative energy
///        contraction  ½ Σ_{μνλσ} P_{μν} P_{λσ} [(μν|λσ)^{xy}
///        − ½(μλ|νσ)^{xy}]  (closed-shell RHF) into `hess`.
///
/// Contracted on the fly per primitive quartet — the four-index second-
/// derivative tensor is never materialized. One shared McMurchie-Davidson
/// R-table (built to Ltot+2) and per-axis Hermite E tables (angular
/// momenta raised by up to +2) serve all 45 unique second-derivative
/// elements of a quartet; center D derivatives follow from translational
/// invariance. OpenMP over canonical block pairs.
///
/// @param[out] hess   Accumulator [3·n_atoms, 3·n_atoms] [Hartree/Bohr²],
///                    row/column index c = 3·atom + axis.
/// @param integrals   Integral pack for the current geometry.
/// @param P_total     Total (factor-2 closed-shell) density (dimensionless).
void accumulate_two_electron_hessian(xt::xtensor<double, 2>& hess,
                                     const IntegralPack& integrals,
                                     const xt::xtensor<double, 2>& P_total);

} // namespace scf
} // namespace cpu
} // namespace ATV
