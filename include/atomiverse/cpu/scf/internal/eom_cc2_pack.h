/// @file cpu/scf/internal/eom_cc2_pack.h
/// @brief Private pack-based EOM-CC2 entry.
#pragma once

#include "atomiverse/cpu/scf/eom_cc2.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell EOM-CC2 singlet excitation energies.
///
/// Solves spin-orbital CC2 (`ccsd_solve(..., cc2=true)`), builds the same
/// EOM-EE intermediates as EOM-EE-CCSD (PySCF `eom_gccsd` / gintermediates)
/// on those amplitudes, diagonalizes the non-symmetric Jacobian, and returns
/// the lowest singlet roots. Matches PySCF
/// `RCCSD(cc2=True).eomee_ccsd_singlet()`.
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged closed-shell determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_states Number of lowest singlet roots (dimensionless, ≥ 1).
/// @param tol      CC2 amplitude residual tolerance.
/// @param max_iter CC2 Jacobi budget.
/// @return Singlet ω [eV] plus CC2 energetics.
/// @throws std::invalid_argument on open-shell / missing ERI / bad n_states /
///         non-convergence (no silent fallback).
EOMCC2Result eom_cc2_energies(const internal::IntegralView& ints,
                              const SlaterDeterminant& det, double e_scf_ha,
                              int n_states = 1, double tol = 1e-10,
                              int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
