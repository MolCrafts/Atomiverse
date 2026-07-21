/// @file cpu/scf/fci.h
/// @brief Tiny full CI (public API) — use `RHF::fci(frame)`.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Exact closed-shell full-CI energy.
struct FCIResult {
    double e_fci;  ///< Exact (full-CI) total energy [Hartree].
    double e_corr; ///< Correlation energy e_fci - scf.energy [Hartree] (<= 0).
};

// Pack-based entry is private — see internal/fci_pack.h. Public: RHF::fci.

} // namespace scf
} // namespace cpu
} // namespace ATV
