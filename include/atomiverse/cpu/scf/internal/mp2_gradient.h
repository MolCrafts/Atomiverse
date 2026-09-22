/// @file cpu/scf/internal/mp2_gradient.h
/// @brief **PRIVATE** RMP2 / UMP2 analytic nuclear gradient workspace.
///
/// Builds the MP2 Lagrangian densities (unrelaxed 1-RDM + non-separable 2-RDM
/// from T₂, Z-vector orbital response) and contracts with skeleton integral
/// derivatives. Matches PySCF `mp.MP2.nuc_grad_method().kernel()` (closed-shell)
/// and `mp.UMP2.nuc_grad_method().kernel()` (unrestricted).
///
/// Scope: canonical RMP2 / UMP2, no frozen core, no RI, Cartesian AO packs
/// (STO-3G / Pople). Spherical / frozen / SCS / ROMP2 gradients are follow-ups.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Analytic RMP2 nuclear forces [eV/Å] on the total MP2 energy
///        E_HF + E_corr.
/// @param frame     Geometry [Å] (read-only positions).
/// @param pack      Integral pack with dense ERI materialized.
/// @param det       Converged closed-shell RHF determinant.
/// @param e_scf_ha  HF energy [Hartree].
/// @param e_corr_ha MP2 correlation energy [Hartree] (for result.energy).
/// @return Forces [eV/Å] and total MP2 energy [eV].
/// @throws std::invalid_argument on open-shell / empty ERI / n_virt=0 /
///         frozen-core requests (caller must pass a full-valence reference).
GradientResult mp2_analytic_forces(const Frame& frame,
                                   const internal::IntegralView& pack,
                                   const SlaterDeterminant& det,
                                   double e_scf_ha, double e_corr_ha);

/// @brief Analytic UMP2 nuclear forces [eV/Å] on E_UHF + E_corr.
/// @param frame     Geometry [Å] (read-only positions).
/// @param pack      Integral pack with dense ERI materialized.
/// @param det       Converged unrestricted UHF determinant.
/// @param e_scf_ha  UHF energy [Hartree].
/// @param e_corr_ha UMP2 correlation energy [Hartree] (for result.energy).
/// @return Forces [eV/Å] and total UMP2 energy [eV].
/// @throws std::invalid_argument on restricted / empty ERI / spherical /
///         missing virtual on either spin.
GradientResult ump2_analytic_forces(const Frame& frame,
                                    const internal::IntegralView& pack,
                                    const SlaterDeterminant& det,
                                    double e_scf_ha, double e_corr_ha);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
