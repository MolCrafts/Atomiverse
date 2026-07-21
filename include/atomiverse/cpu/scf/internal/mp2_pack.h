/// @file cpu/scf/internal/mp2_pack.h
/// @brief Private pack-based MP2 entry (unit tests / method implementation).
#pragma once

#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/slater_determinant.h" // SlaterDeterminant

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell canonical MP2 energy from a converged reference.
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant (MO
///                 coefficients, orbital energies, electron counts).
/// @param e_scf_ha Converged SCF total energy [Hartree] (reported as the
///                 reference part of `MP2Result::e_tot`).
/// @return MP2 correlation and total energies [Hartree].
MP2Result mp2_energy(const IntegralPack& ints, const SlaterDeterminant& det,
                     double e_scf_ha);

} // namespace scf
} // namespace cpu
} // namespace ATV
