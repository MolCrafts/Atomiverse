/// @file cpu/scf/internal/gradient.h
/// @brief Analytic nuclear gradients for mean-field methods (RHF, UHF, ROHF,
///        GHF, RKS, UKS).
///
/// Entry point: `internal::ScfGradient` — constructed from the discrete method
/// parameters (method / spin / xc_type / grid) the gradient assembly reads, its
/// `compute(...)` member dispatches to HF (Coulomb + exchange) or KS (Coulomb +
/// XC grid). GHF uses the spinor energy-weighted density and the unsymmetrized
/// four-block exchange contraction (P^{αβ} is not 8-fold symmetric).
///
/// `compute_at_density(...)` is the same assembly at an electronic state that
/// does NOT diagonalize the Fock matrix — the Car–Parrinello force, whose Pulay
/// weight is the Lagrange-multiplier matrix W = occ·C(CᵀFC)Cᵀ instead of the
/// energy-weighted density.
///
/// The assembly covers **every** term the method folds into the total energy it
/// reports — including the geometry-only DFT-D3(BJ)/DFT-D4 dispersion legs,
/// which have no electronic structure at all but do have a gradient. A term in
/// the energy without its leg here is a non-conservative force.
///
/// References:
///   Pulay, Mol. Phys. 17, 197 (1969) -- Pulay forces
///   Szabo & Ostlund, Appendix C -- RHF gradient derivation
///   Helgaker, Jorgensen, Olsen, Ch. 9 -- molecular gradients
///   Fukutome, Int. J. Quantum Chem. 20, 955 (1981) -- GHF / spinor HF
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Unified analytic nuclear gradient for all mean-field methods.
///
/// Holds the discrete method parameters the gradient assembly reads;
/// `compute(...)` dispatches on them:
///   - HF (Restricted or Unrestricted): Coulomb + exchange two-electron terms
///   - KS (Restricted or Unrestricted): Coulomb only + XC grid contribution
///
/// The UHF two-electron formula naturally reduces to the RHF fused form for
/// restricted cases (P_alpha = P_beta = P/2).
class ScfGradient {
  public:
    /// @brief Capture the discrete method parameters that drive the gradient.
    /// @param method    HF or KS mean field.
    /// @param spin      Restricted or unrestricted spin treatment.
    /// @param xc_type   Exchange-correlation functional (used only for KS).
    /// @param grid_level DFT integration-grid quality level (KS only).
    /// @param n_radial  Radial grid points per atom (SG0 grid only).
    /// @param n_angular Angular grid points per atom (SG0 grid only).
    /// @param xc_grad_project_translation Project out the net translational XC
    ///        gradient (debug / fixed-grid comparison; production leaves false).
    /// @param d3bj    The method adds PBE-D3(BJ) dispersion to its total
    ///        energy, so the gradient must carry the matching leg.
    /// @param d4      The method adds PBE-D4 dispersion to its total energy
    ///        (wins over `d3bj` when both are set, mirroring the energy).
    ScfGradient(MeanFieldType method, SpinType spin, XCType xc_type,
                GridLevel grid_level, int n_radial, int n_angular,
                bool xc_grad_project_translation, bool d3bj, bool d4);

    /// @brief Unified analytic nuclear gradient at the current geometry.
    /// @param frame     System geometry (positions in Angstrom, type_id = Z).
    /// @param integrals Precomputed integrals at current geometry (atomic units).
    /// @param det       Converged mean-field wavefunction (per-spin densities,
    ///                  MO coefficients, orbital energies).
    /// @param e_scf_ha  Converged SCF total energy [Hartree] (reported on the
    ///                  returned GradientResult in eV).
    /// @return Forces on all atoms (eV/Angstrom) and total energy (eV).
    GradientResult compute(const Frame& frame, const internal::IntegralView& integrals,
                           const SlaterDeterminant& det, double e_scf_ha) const;

    /// @brief Analytic nuclear gradient at an arbitrary (non-stationary)
    ///        electronic state — the Car–Parrinello force.
    ///
    /// `compute` assumes the coefficients diagonalize the Fock matrix, so its
    /// Pulay weight is the energy-weighted density
    /// \(\sum_i \varepsilon_i c_i c_i^\top\). Propagated Car–Parrinello
    /// orbitals satisfy \(C^\top S C = I\) but are **not** eigenvectors, so the
    /// weight must be the full Lagrange-multiplier form
    /// \(W_\sigma = \mathrm{occ}\cdot C_\sigma (C_\sigma^\top F_\sigma
    /// C_\sigma) C_\sigma^\top\) — which collapses to the energy-weighted
    /// density exactly when \(C_\sigma^\top F_\sigma C_\sigma\) is diagonal.
    /// Restricted folds occ = 2 into one channel
    /// (\(W = 2\,C(C^\top F C)C^\top\), \(P = 2\,CC^\top\)); unrestricted
    /// carries occ = 1 per channel and sums. Same assembly as `compute`
    /// otherwise (one-electron, Coulomb/exchange or XC grid, ECP, embedding,
    /// field, solvent). Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP
    /// 2009) §2.5; Pulay, Mol. Phys. **17**, 197 (1969).
    ///
    /// @param frame       System geometry (positions in Å, type_id = Z).
    /// @param integrals   Integrals at that geometry (atomic units).
    /// @param C_occ_alpha Occupied α coefficients [n_basis × n_occ^α]
    ///                    (dimensionless, S-orthonormal).
    /// @param F_alpha     α Fock/KS matrix at this density [Hartree], n × n.
    /// @param C_occ_beta  Occupied β coefficients [n_basis × n_occ^β]; **must
    ///                    be empty** when the method is restricted, and may be
    ///                    empty (zero columns) for an unrestricted channel
    ///                    with no β electrons.
    /// @param F_beta      β Fock/KS matrix [Hartree] (unrestricted only;
    ///                    empty when restricted).
    /// @param e_ha        Total energy at this density [Hartree] (reported on
    ///                    the result in eV).
    /// @return Forces on all atoms [eV/Å] and total energy [eV].
    /// @throws std::invalid_argument for ROHF/ROKS/GHF spin treatments (their
    ///         weights are not this one), on a shape mismatch, or when a
    ///         restricted call supplies β matrices.
    GradientResult
    compute_at_density(const Frame& frame, const internal::IntegralView& integrals,
                       const RealMatrix& C_occ_alpha, const RealMatrix& F_alpha,
                       const RealMatrix& C_occ_beta, const RealMatrix& F_beta,
                       double e_ha) const;

  private:
    /// @brief The gradient assembly shared by both entry points.
    ///
    /// Everything after "what are P and W": one-electron + Pulay, two-electron
    /// (or Coulomb + XC grid), ECP, QM/MM embedding, external field, and
    /// continuum solvent.
    /// @param frame     Geometry (Å).
    /// @param integrals Integrals at that geometry (atomic units).
    /// @param P_total   Total density \(P^\alpha + P^\beta\) (dimensionless).
    /// @param P_alpha   Per-spin factor-1 α density (unused when restricted HF).
    /// @param P_beta    Per-spin factor-1 β density.
    /// @param W         Energy-weighted / multiplier density [Hartree].
    /// @param e_ha      Total energy [Hartree] carried onto the result.
    /// @param generalized_det GHF spinor wavefunction — required (and used)
    ///        only by the `SpinType::Generalized` exchange contraction, which
    ///        needs all four spin blocks; null for every other spin treatment.
    /// @return Forces [eV/Å] + energy [eV].
    GradientResult assemble_(const Frame& frame, const internal::IntegralView& integrals,
                             const xt::xtensor<double, 2>& P_total,
                             const xt::xtensor<double, 2>& P_alpha,
                             const xt::xtensor<double, 2>& P_beta,
                             const xt::xtensor<double, 2>& W, double e_ha,
                             const SlaterDeterminant* generalized_det) const;

    MeanFieldType method_;                 ///< HF or KS mean field.
    SpinType spin_;                        ///< Restricted / unrestricted.
    XCType xc_type_;                       ///< XC functional (KS only).
    GridLevel grid_level_;                 ///< DFT grid quality (KS only).
    int n_radial_;                         ///< Radial grid points (SG0 only).
    int n_angular_;                        ///< Angular grid points (SG0 only).
    bool xc_grad_project_translation_;     ///< Project net translational XC.
    bool d3bj_;                            ///< Add the D3(BJ) dispersion leg.
    bool d4_;                              ///< Add the D4 dispersion leg.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
