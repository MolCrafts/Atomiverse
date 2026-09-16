/// @file cpu/scf/internal/ao2mo.h
/// @brief Private AO→MO integral transforms shared across the post-HF methods.
///
/// These free functions are BLAS kernels (chemist-notation four-index / core
/// Hamiltonian transforms), reused by MP2/MP3/MP4, CCSD, CISD, ADC, GF2,
/// G0W0, BSE, CASSCF and FCI. They are private post-HF workspace: the public
/// MP2 header exposes only `MP2Result` and the `RHF/UHF/ROHF::mp2(frame)`
/// entry, never these transforms. Definitions live in `src/cpu/scf/mp2.cpp`.
#pragma once

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

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

} // namespace scf
} // namespace cpu
} // namespace ATV
