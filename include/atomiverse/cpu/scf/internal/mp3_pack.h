/// @file cpu/scf/internal/mp3_pack.h
/// @brief Private pack-based MP3 entry (method implementation / unit tests).
#pragma once

#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/mp3.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell canonical MP3 energy from a converged RHF reference.
///
/// Spin-orbital third-order PT (Szabo & Ostlund; Psi4NumPy MP3 diagrams):
/// first-order doubles \(t_{ij}^{ab}=\langle ij\|ab\rangle/D_{ijab}\), then
/// \(E^{(2)}=\tfrac14\sum t\langle ij\|ab\rangle\) and
/// \(E^{(3)}=\tfrac14\sum t\,u\) with the standard ladder + ring residual
/// \(u_{ijab}\). Frozen-core: lowest `n_frozen` spatial occupied MOs (both
/// spins) are excluded from the correlated SO set.
///
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference (MO coeffs + energies).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_frozen Lowest occupied spatial MOs frozen (0 = full correlation).
/// @return MP3 correlation pieces and total energy [Hartree].
MP3Result mp3_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                     double e_scf_ha, int n_frozen = 0);

} // namespace scf
} // namespace cpu
} // namespace ATV
