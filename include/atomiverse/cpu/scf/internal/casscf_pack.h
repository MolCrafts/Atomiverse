/// @file cpu/scf/internal/casscf_pack.h
/// @brief Private pack-based CASSCF entry (active-space FCI + orbital Newton).
#pragma once

#include "atomiverse/cpu/scf/casscf.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CASSCF ground-state energy (dense active-space CI +
///        orbital optimization).
///
/// Active orbitals are the contiguous block `[ncore, ncore+ncas)` with
/// `ncore = n_occ − nelecas/2`. Non-redundant rotations are core–active,
/// active–virtual, and core–virtual; active–active rotations are redundant for
/// a complete active space and are never stepped.
///
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree] (for `e_corr`).
/// @param ncas     Number of active spatial orbitals (dimensionless).
/// @param nelecas  Number of active electrons (even; closed-shell).
/// @param orb_tol  Orbital-gradient convergence [Hartree] (max |g_pq|).
/// @param max_macro Maximum (CI + orbital) macro-iterations.
/// @return CASSCF total / correlation energies [Hartree] and bookkeeping.
/// @throws std::invalid_argument on open-shell, missing ERI, or invalid
///         active-space dimensions (no silent fallback).
CASSCFResult casscf_energy(const internal::IntegralView& ints,
                           const SlaterDeterminant& det, double e_scf_ha,
                           int ncas, int nelecas, double orb_tol = 1e-8,
                           int max_macro = 50);

} // namespace scf
} // namespace cpu
} // namespace ATV
