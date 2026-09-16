/// @file cpu/scf/adc.h
/// @brief ADC(2) / ADC(2)-x excitation energies (public API, result-only).
///
/// User paths on a converged closed-shell RHF:
/// - `RHF::adc2` / `RHF::adc2x` — electronic excitation (EE)
/// - `RHF::ip_adc2` — ionization potential (IP)
/// - `RHF::ea_adc2` — electron attachment (EA)
///
/// EE path: `RHF::adc2(n_states)` or `RHF::adc2x(n_states)` on a converged
/// closed-shell RHF reference. Intermediate-state representation of the PP
/// propagator through second order (Schirmer, Phys. Rev. A **26**, 2395
/// (1982); Trofimov & Schirmer, J. Phys. B **28**, 2299 (1995)).
///
/// - Strict ADC(2): second-order singles block, first-order singles–doubles
///   coupling, **zeroth-order** doubles diagonal only.
/// - ADC(2)-x: same singles + coupling, plus **first-order** doubles–doubles
///   (vvvv / oovv / ovvo / oooo) — PySCF `method='adc(2)-x'`.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief ADC(2) / ADC(2)-x electronic excitation energies on a closed-shell
///        RHF reference (singlet EE manifold; PySCF `adc.ADC` /
///        `method_type='ee'` convention). Same result shape for both methods;
///        the method is chosen by the caller (`adc2` vs `adc2x`).
struct ADC2Result {
    /// @brief Excitation energies ω [eV], ascending (Davidson Ritz).
    std::vector<double> energy_ev;

    /// @brief MP2 ground-state correlation energy of the same reference
    ///        [Hartree] (PySCF `e_corr` / doubles-only MP2).
    double e_corr_mp2_ha = 0.0;
};

/// @brief Strict ADC(2) ionization potentials on a closed-shell RHF
///        reference (PySCF `adc.ADC` / `method='adc(2)'` /
///        `method_type='ip'` convention).
struct IPADC2Result {
    /// @brief ADC(2) ionization potentials IP [eV], ascending.
    std::vector<double> energy_ev;

    /// @brief MP2 ground-state correlation energy of the same reference
    ///        [Hartree] (PySCF `e_corr` / doubles-only MP2).
    double e_corr_mp2_ha = 0.0;
};

/// @brief Strict EA-ADC(2) electron-attachment energies on a closed-shell RHF
///        reference (PySCF `adc.ADC` / `method='adc(2)'` / `method_type='ea'`).
///
/// Singles space is the virtual (1p) manifold; doubles are 2p–1h. Reported
/// energies are the EA eigenvalues ω [eV] of the EA-ADC matrix (positive for
/// endothermic attachment, e.g. H₂/STO-3G LUMO).
struct EAADC2Result {
    /// @brief EA-ADC(2) attachment energies ω [eV], ascending (Davidson Ritz).
    std::vector<double> energy_ev;

    /// @brief MP2 ground-state correlation energy of the same N-electron
    ///        reference [Hartree] (PySCF `e_corr` / doubles-only MP2).
    double e_corr_mp2_ha = 0.0;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
