/// @file cpu/scf/gf2.h
/// @brief One-shot second-order Green's function (G0F2 / GF2) quasiparticle
///        energies (public, result-only header — g0w0.h / mp2.h precedent).
///
/// User path: `RHF::gf2()` on a converged closed-shell RHF reference.
/// Builds the second-order self-energy in the auxiliary (Lehmann) representation
/// from the HF Green's function (2h1p + 2p1h), solves the Dyson equation once
/// with the HF Fock matrix, and reports the physical poles as quasiparticle
/// energies. Matches the one-shot (non-self-consistent) path of PySCF
/// `agf2.RAGF2` — SE from `init_gf()` then `se.get_greens_function(F_HF)`.
/// Self-consistent AGF2 is a separate roadmap item.
///
/// References: Backhouse, Holzmann & Booth, JCTC **17**, 3414 (2021)
/// (PySCF AGF2); Galitskii & Migdal, Sov. Phys. JETP **7**, 96 (1958).
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief One-shot GF2 quasiparticle spectrum of a closed-shell RHF reference.
struct GF2Result {
    /// @brief Quasiparticle pole energies ε_p^QP [Hartree] for the physical
    ///        poles with the largest weight near each HF MO (HOMO/LUMO pair
    ///        always present; full qmo spectrum may be larger).
    std::vector<double> qp_energy_ha;

    /// @brief Mean-field (RHF) orbital energies ε_p [Hartree].
    std::vector<double> mf_energy_ha;

    /// @brief Full Dyson-spectrum pole energies [Hartree] (physical +
    ///        satellite), sorted ascending — length n_mo + n_aux.
    std::vector<double> pole_energy_ha;

    /// @brief HOMO→LUMO quasiparticle gap (ε_LUMO^QP − ε_HOMO^QP) [eV]
    ///        equivalently IP + EA from the charged poles.
    double gap_ev = 0.0;

    /// @brief Vertical ionization potential −ε_HOMO^QP [eV].
    double ip_ev = 0.0;

    /// @brief Vertical electron affinity ε_LUMO^QP [eV].
    double ea_ev = 0.0;

    /// @brief Galitskii–Migdal correlation energy from the one-shot SE
    ///        (equals canonical MP2 for a pure HF reference) [Hartree].
    double e_corr = 0.0;

    /// @brief Total energy E_HF + e_corr [Hartree].
    double e_tot = 0.0;

    /// @brief Number of occupied spatial MOs (closed-shell).
    int n_occ = 0;

    /// @brief true when the Dyson spectrum yielded a well-defined HOMO/LUMO
    ///        chemical-potential cut (N-electron Aufbau).
    bool converged = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
