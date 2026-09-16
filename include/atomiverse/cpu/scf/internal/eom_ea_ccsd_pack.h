/// @file cpu/scf/internal/eom_ea_ccsd_pack.h
/// @brief Private pack-based EOM-EA-CCSD entry.
#pragma once

#include "atomiverse/cpu/scf/eom_ea_ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell EOM-EA-CCSD electron-attachment energies.
///
/// Solves spin-orbital CCSD, builds EOM intermediates (PySCF
/// `gintermediates` / `eom_gccsd`), diagonalizes the non-symmetric EA
/// Jacobian in the packed 1p+2p1h space, and returns the lowest unique
/// attachment roots.
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged closed-shell determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_states Number of lowest unique EA roots (dimensionless, ≥ 1).
/// @param tol      CCSD amplitude residual tolerance.
/// @param max_iter CCSD Jacobi budget.
/// @return Attachment ω [eV] plus CCSD energetics.
/// @throws std::invalid_argument on open-shell / missing ERI / bad n_states /
///         non-convergence (no silent fallback).
EOMEACCSDResult eom_ea_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states = 1,
                                     double tol = 1e-10, int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
