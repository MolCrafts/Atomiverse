/// @file cpu/semiempirical/internal/dftb_spin.h
/// @brief **PRIVATE WORKSPACE** — HOAO collapse of shell-resolved DFTB W.
///
/// Users never call this; `DFTB2::launch` / `DFTB2::td_casida` consume
/// `DFTB2::spin_w_ha` and pass the collapsed per-atom scalars [eV] down the
/// existing spin channel. Frame is the only I/O currency.
#pragma once

#include <vector>

#include "atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief Collapse shell-resolved \(W^A_{ll'}\) [Ha] to a per-atom HOAO
///        scalar [eV].
///
/// H (`n_shell == 1` or \(Z = 1\)) → \(W_{ss} = w_{ha}[0][0]\);
/// C/N/O (`n_shell == 2`) → \(W_{pp} = w_{ha}[1][1]\). Converted once with
/// `units::kHartreeInEv`. Missing Z, empty table, or unsupported `n_shell`
/// throws `std::invalid_argument` (no fallback).
///
/// @param spin_w_ha Shell-resolved constants [Ha], typically
///        `basis_data::kDftbSpinConstantsPbeHa`.
/// @param z_per_atom Atomic number Z of each atom (dimensionless).
/// @return Per-atom HOAO \(W_A\) [eV], size `z_per_atom.size()`.
/// @throws std::invalid_argument when the table is empty, a requested Z is
///         absent, or `n_shell` is not 1 or 2.
std::vector<double>
collapse_spin_w_ev(const std::vector<basis_data::DftbSpinConstant>& spin_w_ha,
                   const std::vector<int>& z_per_atom);

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
