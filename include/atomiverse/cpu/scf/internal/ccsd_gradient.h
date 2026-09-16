/// @file cpu/scf/internal/ccsd_gradient.h
/// @brief **PRIVATE** closed-shell CCSD nuclear gradient workspace.
///
/// This link: central finite difference of the analytic CCSD total energy
/// (h = 1e-4 Bohr) on the same RHF→`ccsd_energy` path. Matches PySCF
/// `cc.CCSD.nuc_grad_method().kernel()` force values to the regression gate.
/// Density-based analytic (λ + Lagrangian 2-RDM + Z-vector) is a follow-up.
///
/// Scope: canonical closed-shell CCSD, no frozen core, Cartesian AO packs
/// (STO-3G / Pople). Spherical / frozen / UCCSD are follow-up links.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Analytic CCSD nuclear forces [eV/Å] on the total energy
///        E_HF + E_CCSD.
/// @param frame     Geometry [Å] (read-only positions).
/// @param pack      Integral pack with dense ERI materialized.
/// @param det       Converged closed-shell RHF determinant.
/// @param e_scf_ha  HF energy [Hartree].
/// @param e_corr_ha CCSD correlation energy [Hartree] (for result.energy).
/// @return Forces [eV/Å] and total CCSD energy [eV].
/// @throws std::invalid_argument on open-shell / empty ERI / n_virt=0 /
///         non-converged amplitudes / spherical packs (no fallback).
GradientResult ccsd_analytic_forces(const Frame& frame,
                                    const internal::IntegralView& pack,
                                    const SlaterDeterminant& det,
                                    double e_scf_ha, double e_corr_ha);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
