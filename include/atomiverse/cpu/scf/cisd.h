/// @file cpu/scf/cisd.h
/// @brief CIS(D) doubles correction to TDA/CIS excitation energies (public API).
///
/// User path: `RHF::cisd(n_states)` on a converged closed-shell RHF reference.
/// Head-Gordon, Rico, Oumi & Lee, Chem. Phys. Lett. **219**, 21 (1994),
/// DOI 10.1016/0009-2614(94)00070-0 — spin-orbital second-order doubles
/// correction on a CIS zeroth-order root (size-consistent, O(N⁵) per state).
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief CIS(D) excitation energies and the parent CIS roots they correct.
struct CISDResult {
    /// @brief CIS(D) excitation energies ω_CIS(D) = ω_CIS + U + V [eV],
    ///        ascending in the parent CIS order (same root indexing as
    ///        `RHF::tda`).
    std::vector<double> energy_ev;

    /// @brief Parent CIS / TDA excitation energies ω_CIS [eV].
    std::vector<double> cis_energy_ev;

    /// @brief Doubles correction U + V per root [eV]
    ///        (ω_CIS(D) − ω_CIS; may raise or lower the root).
    std::vector<double> doubles_correction_ev;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
