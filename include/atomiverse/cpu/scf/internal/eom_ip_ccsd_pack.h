/// @file cpu/scf/internal/eom_ip_ccsd_pack.h
/// @brief Private pack-based EOM-IP-CCSD entry.
#pragma once

#include "atomiverse/cpu/scf/eom_ip_ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell EOM-IP-CCSD ionization potentials.
///
/// Solves spin-orbital CCSD, builds EOM intermediates (PySCF
/// `gintermediates` / `eom_gccsd`), diagonalizes the non-symmetric IP
/// Jacobian in the packed 1h+2h1p space, and returns the lowest unique
/// ionization roots.
///
/// @param ints     Integral pack with dense ERI materialized.
/// @param det      Converged closed-shell determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_states Number of lowest unique IP roots (dimensionless, ≥ 1).
/// @param tol      CCSD amplitude residual tolerance.
/// @param max_iter CCSD Jacobi budget.
/// @return Ionization ω [eV] plus CCSD energetics.
/// @throws std::invalid_argument on open-shell / missing ERI / bad n_states /
///         non-convergence (no silent fallback).
EOMIPCCSDResult eom_ip_ccsd_energies(const internal::IntegralView& ints,
                                     const SlaterDeterminant& det,
                                     double e_scf_ha, int n_states = 1,
                                     double tol = 1e-10, int max_iter = 100);

} // namespace scf
} // namespace cpu
} // namespace ATV
