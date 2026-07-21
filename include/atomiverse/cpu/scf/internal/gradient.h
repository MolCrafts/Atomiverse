/// @file cpu/scf/internal/gradient.h
/// @brief Analytic nuclear gradients for mean-field methods (RHF, UHF, RKS, UKS).
///
/// Entry point: `internal::ScfGradient` — constructed from the discrete method
/// parameters (method / spin / xc_type / grid) the gradient assembly reads, its
/// `compute(...)` member dispatches to HF (Coulomb + exchange) or KS (Coulomb +
/// XC grid).
///
/// References:
///   Pulay, Mol. Phys. 17, 197 (1969) -- Pulay forces
///   Szabo & Ostlund, Appendix C -- RHF gradient derivation
///   Helgaker, Jorgensen, Olsen, Ch. 9 -- molecular gradients
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/gradient_result.h"
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
    /// @param force_native_two_electron_deriv Force the native 2e derivative
    ///        engine even when libcint is available (native-vs-libcint tests).
    /// @param xc_grad_project_translation Project out the net translational XC
    ///        gradient (debug / fixed-grid comparison; production leaves false).
    ScfGradient(MeanFieldType method, SpinType spin, XCType xc_type,
                GridLevel grid_level, int n_radial, int n_angular,
                bool force_native_two_electron_deriv,
                bool xc_grad_project_translation);

    /// @brief Unified analytic nuclear gradient at the current geometry.
    /// @param frame     System geometry (positions in Angstrom, type_id = Z).
    /// @param integrals Precomputed integrals at current geometry (atomic units).
    /// @param det       Converged mean-field wavefunction (per-spin densities,
    ///                  MO coefficients, orbital energies).
    /// @param e_scf_ha  Converged SCF total energy [Hartree] (reported on the
    ///                  returned GradientResult in eV).
    /// @return Forces on all atoms (eV/Angstrom) and total energy (eV).
    GradientResult compute(const Frame& frame, const IntegralPack& integrals,
                           const SlaterDeterminant& det,
                           double e_scf_ha) const;

  private:
    MeanFieldType method_;                 ///< HF or KS mean field.
    SpinType spin_;                        ///< Restricted / unrestricted.
    XCType xc_type_;                       ///< XC functional (KS only).
    GridLevel grid_level_;                 ///< DFT grid quality (KS only).
    int n_radial_;                         ///< Radial grid points (SG0 only).
    int n_angular_;                        ///< Angular grid points (SG0 only).
    bool force_native_two_electron_deriv_; ///< Force native 2e deriv engine.
    bool xc_grad_project_translation_;     ///< Project net translational XC.
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
