/// @file cpu/scf/internal/eom_sf_ccsd_pack.h
/// @brief Private pack-based EOM-SF-CCSD entry.
#pragma once

#include "atomiverse/cpu/scf/eom_sf_ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Unrestricted EOM-SF-CCSD excitation energies.
///
/// Solves spin-orbital UCCSD on a UHF reference, builds EOM-EE intermediates
/// (same Gauss–Stanton / Wang–Tu–Wang σ equations as EOM-EE-CCSD), diagonalizes
/// the non-symmetric Jacobian, and returns the lowest spin-flip roots
/// (R1 dominated by α→β when n_α ≥ n_β, else β→α).
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged unrestricted determinant.
/// @param e_scf_ha Converged UHF total energy [Hartree].
/// @param n_states Number of lowest SF roots (dimensionless, ≥ 1).
/// @param tol      UCCSD amplitude residual tolerance.
/// @param max_iter UCCSD Jacobi budget.
/// @return SF ω [eV] plus UCCSD energetics.
/// @throws std::invalid_argument on restricted / missing ERI / bad n_states /
///         non-convergence (no silent fallback).
EOMSFCCSDResult eom_sf_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states = 1,
                                     double tol = 1e-10, int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
