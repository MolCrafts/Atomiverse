/// @file cpu/scf/internal/casci_pack.h
/// @brief Private pack-based CASCI entry (active-space FCI on frozen core).
#pragma once

#include "atomiverse/cpu/scf/casci.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CASCI ground-state energy (dense active-space CI).
///
/// Active orbitals are the contiguous block
/// `[ncore, ncore+ncas)` with `ncore = n_occ − nelecas/2` (doubly occupied
/// core below the active window; virtuals above are discarded).
///
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree] (for `e_corr`).
/// @param ncas     Number of active spatial orbitals (dimensionless).
/// @param nelecas  Number of active electrons (even; closed-shell).
/// @return CASCI total / correlation energies [Hartree].
/// @throws std::invalid_argument on open-shell, missing ERI, or invalid
///         active-space dimensions (no silent fallback).
CASCIResult casci_energy(const internal::IntegralView& ints,
                         const SlaterDeterminant& det, double e_scf_ha,
                         int ncas, int nelecas);

} // namespace scf
} // namespace cpu
} // namespace ATV
