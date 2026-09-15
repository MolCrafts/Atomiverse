/// @file cpu/scf/internal/integral_derivatives.h
/// @brief Analytic derivatives of molecular integrals w.r.t. nuclear positions.
///
/// Computes dS/dR, dT/dR, dV/dR, d(ERI)/dR, and dV_nuc/dR for contracted
/// Gaussians of any l <= kMaxL (s, p, d shells). Works with any basis the
/// SCF path resolves into an `internal::IntegralWorkspace`.
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
#include "atomiverse/cpu/scf/scf_enums.h"          // MeanFieldType / SpinType
#include "atomiverse/cpu/scf/slater_determinant.h" // SlaterDeterminant
#include "atomiverse/cpu/scf/xc_functional.h"      // XCType

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
    ///        already resolved inside `pack` [Hartree/Bohr, 1/Bohr for dS].
    ///
    /// The returned dS / dT / dV tensors have the same n_basis dimension as
    /// `pack.S`, so gradient contractions align regardless of basis.
    /// Supports any l ≤ kMaxL — the primitive helpers delegate to
    /// `native::overlap_prim_3d` / `kinetic_prim_3d` / `nuclear_prim`,
    /// which handle s, p, and d shells identically.
    ///
    /// For a spherical-convention pack (`pack.spherical()`, def2 family) the
    /// Cartesian derivative blocks are sandwiched down to the spherical AO
    /// basis with the SAME stored `pack.c2s` the energy path used —
    /// `dO^sph[A,d] = c2sᵀ · dO^cart[A,d] · c2s` — reusing it (never rebuilt).
    /// The transform is geometry-independent, so no extra terms arise; dV_nuc
    /// is basis-independent and untouched.
    ///
    /// @param pack Integral pack (provides atoms in Bohr + resolved shells).
    explicit IntegralDerivPack(const internal::IntegralView& pack);
};

/// @brief Contract two-electron derivative integrals directly into the gradient.
///
/// Avoids materializing the full dERI tensor, which scales as O(n_atoms * n_basis^4)
/// in memory. The contraction is performed on the fly using the same analytic
/// primitive derivative formulas as IntegralDerivPack's constructor.
///
/// Works for any basis: the libcint engine handles STO-3G (when compiled in)
/// and the native Rys-quadrature block-derivative engine handles all l ≤ kMaxL.
/// For a spherical-convention pack the density is back-transformed to Cartesian
/// (`P^cart = c2s · P^sph · c2sᵀ`, reusing `integrals.c2s`) before the Cartesian
/// on-the-fly contraction — the density dual of the operator sandwich, so no
/// dERI tensor is ever materialized.
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
/// @param integrals      Integral pack for the current geometry.
/// @param P_total        Total density matrix P^alpha + P^beta (dimensionless).
/// @param det            Converged mean-field wavefunction; supplies the
///                       per-spin factor-1 densities for the unrestricted
///                       exchange contraction, and all four spin blocks for
///                       `SpinType::Generalized`.
/// @param method         HF or KS mean field (selects the exchange fraction).
/// @param spin           Restricted or unrestricted spin treatment.
/// @param xc_type        XC functional (KS hybrid exact-exchange fraction).
void accumulate_two_electron_gradient(xt::xtensor<double, 2>& grad,
                                      const internal::IntegralView& integrals,
                                      const xt::xtensor<double, 2>& P_total,
                                      const SlaterDeterminant& det,
                                      MeanFieldType method, SpinType spin,
                                      XCType xc_type);

/// @brief Same contraction from explicit spin densities (no determinant).
///
/// The Car–Parrinello force is taken at a propagated electronic state, which
/// is not a converged determinant — only the densities exist. Collinear spin
/// treatments only: `SpinType::Generalized` needs the spin-off-diagonal blocks
/// and must use the determinant overload above.
///
/// @param[out] grad Gradient accumulator [n_atoms, 3] in Hartree/Bohr.
/// @param integrals Integral pack for the current geometry.
/// @param P_total   Total density P^alpha + P^beta (dimensionless).
/// @param P_alpha   Per-spin factor-1 alpha density (unrestricted only; the
///                  restricted branches read `P_total` alone).
/// @param P_beta    Per-spin factor-1 beta density.
/// @param method    HF or KS mean field (selects the exchange fraction).
/// @param spin      Restricted or unrestricted spin treatment.
/// @param xc_type   XC functional (KS hybrid exact-exchange fraction).
/// @throws std::invalid_argument when `spin == SpinType::Generalized`.
void accumulate_two_electron_gradient(xt::xtensor<double, 2>& grad,
                                      const internal::IntegralView& integrals,
                                      const xt::xtensor<double, 2>& P_total,
                                      const xt::xtensor<double, 2>& P_alpha,
                                      const xt::xtensor<double, 2>& P_beta,
                                      MeanFieldType method, SpinType spin,
                                      XCType xc_type);

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
    explicit IntegralSecondDerivPack(const internal::IntegralView& pack);
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
/// @param alpha_hf    Exact-exchange fraction in [0, 1]: 1 for pure HF,
///                    0 for pure semi-local KS, hybrid_exchange_fraction(xc)
///                    for global hybrids. Scales the −½K channel only.
void accumulate_two_electron_fock_derivative(
    xt::xtensor<double, 4>& dF, const internal::IntegralView& integrals,
    const xt::xtensor<double, 2>& P_total, double alpha_hf = 1.0);

/// @brief Unrestricted skeleton Fock derivatives into separate α/β
///        accumulators: F^{α(x)} += J^{(x)}[P] − K^{(x)}[P^α] and
///        F^{β(x)} += J^{(x)}[P] − K^{(x)}[P^β] with P = P^α + P^β.
///
/// Same on-the-fly derivative-ERI engine as the closed-shell overload;
/// exchange uses the factor-1 spin densities (no closed-shell ½).
///
/// @param[out] dFa    α Fock accumulator [n_atoms, 3, n, n] [Hartree/Bohr]
///                    (caller pre-fills with dT + dV).
/// @param[out] dFb    β Fock accumulator (same layout; also pre-filled).
/// @param integrals   Integral pack for the current geometry.
/// @param Pa          α density (factor-1, dimensionless).
/// @param Pb          β density (factor-1, dimensionless).
void accumulate_two_electron_fock_derivative_uhf(
    xt::xtensor<double, 4>& dFa, xt::xtensor<double, 4>& dFb,
    const internal::IntegralView& integrals, const xt::xtensor<double, 2>& Pa,
    const xt::xtensor<double, 2>& Pb);

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
/// @param alpha_hf    Exact-exchange fraction in [0, 1] (see
///                    `accumulate_two_electron_fock_derivative`); scales the
///                    exchange second-derivative channel only.
void accumulate_two_electron_hessian(xt::xtensor<double, 2>& hess,
                                     const internal::IntegralView& integrals,
                                     const xt::xtensor<double, 2>& P_total,
                                     double alpha_hf = 1.0);

/// @brief Unrestricted skeleton two-electron second-derivative energy
///        contraction into `hess`:
///        ½ Σ P P ĵ^{xy} − ½ Σ_σ P^σ P^σ k̂^{xy}  (P = P^α + P^β).
///
/// Same quartet engine as the closed-shell overload; only the density
/// prefactors change (per-spin exchange with factor-1 densities).
///
/// @param[out] hess   Accumulator [3·n_atoms, 3·n_atoms] [Hartree/Bohr²].
/// @param integrals   Integral pack for the current geometry.
/// @param Pa          α density (factor-1, dimensionless).
/// @param Pb          β density (factor-1, dimensionless).
void accumulate_two_electron_hessian_uhf(
    xt::xtensor<double, 2>& hess, const internal::IntegralView& integrals,
    const xt::xtensor<double, 2>& Pa, const xt::xtensor<double, 2>& Pb);

// ── Primitive nuclear-attraction derivatives (internal; unit-testable) ─
// -Z / -q prefactor is NOT included; caller multiplies.

/// @brief d/dA of primitive nuclear attraction (center A = Gaussian center).
/// @param a_ang Cartesian angular momenta on A [3].
/// @param b_ang Cartesian angular momenta on B [3].
/// @param alpha_a Exponent on A.
/// @param A Center A [Bohr].
/// @param alpha_b Exponent on B.
/// @param B Center B [Bohr].
/// @param C Nuclear / charge position [Bohr].
/// @param[out] dV_dA dV/dA [3] (bare integral; no −Z).
void nuclear_deriv_prim_center_general(const int a_ang[3], const int b_ang[3],
                                       double alpha_a, const double A[3],
                                       double alpha_b, const double B[3],
                                       const double C[3], double dV_dA[3]);

/// @brief d/dC of primitive nuclear attraction via translational invariance.
/// @param a_ang Cartesian angular momenta on A [3].
/// @param b_ang Cartesian angular momenta on B [3].
/// @param alpha_a Exponent on A.
/// @param A Center A [Bohr].
/// @param alpha_b Exponent on B.
/// @param B Center B [Bohr].
/// @param C Nuclear / charge position [Bohr].
/// @param[out] dV_dC dV/dC [3] (bare integral; no −Z).
void nuclear_deriv_prim_nuc_general(const int a_ang[3], const int b_ang[3],
                                    double alpha_a, const double A[3],
                                    double alpha_b, const double B[3],
                                    const double C[3], double dV_dC[3]);

} // namespace scf
} // namespace cpu
} // namespace ATV
