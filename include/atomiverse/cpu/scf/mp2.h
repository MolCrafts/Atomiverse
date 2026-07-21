/// @file cpu/scf/mp2.h
/// @brief Closed-shell canonical MP2 correlation energy (public API).
///
/// User path: `RHF::mp2(frame)` (see methods.h). Free functions below are
/// thin wrappers / BLAS kernels — they do **not** introduce IntegralPack as a
/// user type (that type is private under `scf/internal/`).
#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell canonical MP2 energy.
struct MP2Result {
    double e_corr; ///< MP2 correlation energy [Hartree] (<= 0).
    double e_tot;  ///< Total MP2 energy = scf.energy + e_corr [Hartree].
};

/// @brief Transform AO two-electron integrals to the MO (ia|jb) ovov block.
xt::xtensor<double, 4> ao2mo_ovov(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_occ,
                                  int n_mo);

/// @brief Transform AO two-electron integrals to the full (pq|rs) MO tensor.
xt::xtensor<double, 4> ao2mo_full(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_mo);

/// @brief Transform the AO core Hamiltonian to the MO basis.
xt::xtensor<double, 2> mo_hcore(const xt::xtensor<double, 2>& Hcore,
                                const xt::xtensor<double, 2>& C, int n_mo);

// Pack-based entry is private implementation — declared in
// internal/mp2_pack.h if needed by unit tests. Public: RHF::mp2(frame).

} // namespace scf
} // namespace cpu
} // namespace ATV
