/// @file cpu/scf/internal/fci_pack.h
/// @brief Private pack-based FCI entry.
#pragma once

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h" // SlaterDeterminant

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Dense CI Hamiltonian in the determinant basis (TINY systems only).
/// @param ints Integral pack with the ERI tensor materialized.
/// @param det  Converged closed-shell reference determinant (MO coefficients,
///             electron counts).
/// @return CI matrix [N_det x N_det] [Hartree], V_nuc included.
xt::xtensor<double, 2> fci_hamiltonian(const internal::IntegralView& ints,
                                       const SlaterDeterminant& det);

/// @brief Full-CI ground-state energy from the dense CI Hamiltonian.
/// @param ints     Integral pack with the ERI tensor materialized.
/// @param det      Converged closed-shell reference determinant.
/// @param e_scf_ha Converged SCF total energy [Hartree] (the reference for
///                 `FCIResult::e_corr`).
/// @return Full-CI correlation and total energies [Hartree].
FCIResult fci_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha);

} // namespace scf
} // namespace cpu
} // namespace ATV
