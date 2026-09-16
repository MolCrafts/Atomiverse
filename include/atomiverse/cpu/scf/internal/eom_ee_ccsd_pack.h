/// @file cpu/scf/internal/eom_ee_ccsd_pack.h
/// @brief Private pack-based EOM-EE-CCSD entry.
#pragma once

#include "atomiverse/cpu/scf/eom_ee_ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell EOM-EE-CCSD singlet excitation energies.
///
/// Solves spin-orbital CCSD, builds EOM-EE intermediates (PySCF
/// `eom_gccsd` / gintermediates), diagonalizes the non-symmetric Jacobian in
/// the packed singles+doubles space, and returns the lowest singlet roots.
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged closed-shell determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_states Number of lowest singlet roots (dimensionless, ≥ 1).
/// @param tol      CCSD amplitude residual tolerance.
/// @param max_iter CCSD Jacobi budget.
/// @return Singlet ω [eV] plus CCSD energetics.
/// @throws std::invalid_argument on open-shell / missing ERI / bad n_states /
///         non-convergence (no silent fallback).
EOMEECCSDResult eom_ee_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states = 1,
                                     double tol = 1e-10, int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
