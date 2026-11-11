/// @file cpu/scf/internal/mp4_pack.h
/// @brief Private pack-based MP4(SDTQ) entry (method implementation / unit tests).
#pragma once

#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/mp4.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell canonical MP4(SDTQ) energy from a converged RHF reference.
///
/// Spin-orbital fourth-order PT:
/// - first-order doubles \(t_{ij}^{ab}=\langle ij\|ab\rangle/D_{ijab}\);
/// - \(E^{(2)}=\tfrac14\sum t\langle ij\|ab\rangle\);
/// - ladder+ring residual \(u_{ijab}\) → \(E^{(3)}=\tfrac14\sum t\,u\);
/// - second-order singles, doubles residual, triples, and renormalization
///   \(E_Q^{(4)}=-E^{(2)}\|\Psi^{(1)}\|^2\) complete \(E^{(4)}\).
///
/// Frozen-core: lowest `n_frozen` spatial occupied MOs (both spins) are
/// excluded from the correlated SO set.
///
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference (MO coeffs + energies).
/// @param e_scf_ha Converged SCF total energy [Hartree].
/// @param n_frozen Lowest occupied spatial MOs frozen (0 = full correlation).
/// @return MP4 correlation pieces and total energy [Hartree].
MP4Result mp4_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen = 0);

} // namespace scf
} // namespace cpu
} // namespace ATV
