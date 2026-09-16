/// @file cpu/scf/g0w0.h
/// @brief G0W0 quasiparticle energies (public, result-only header — mp2.h /
///        tdhf.h precedent).
///
/// User path: `RHF::g0w0()` on a converged closed-shell RHF reference.
/// Spectral (frequency-exact) G0W0 with direct-RPA (dRPA) screening of W —
/// Berkelbach / PySCF `gw.gw_exact.GWExact` twin (HF reference via
/// `xc = 'hf'`). Quasiparticle equation solved orbit-by-orbit (non-linear
/// Newton, not the perturbative linearized form).
///
/// References: Hybertsen & Louie, Phys. Rev. B 34, 5390 (1986),
/// DOI 10.1103/PhysRevB.34.5390; van Setten et al., JCTC 11, 5665 (2015);
/// Zhu & Chan, JCTC 17, 727 (2021) (AC companion; not this path).
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Diagonal G0W0 quasiparticle energies of a closed-shell RHF reference.
struct G0W0Result {
    /// @brief Quasiparticle orbital energies ε_p^QP [Hartree], same order as
    ///        the RHF MO spectrum (not re-sorted).
    std::vector<double> qp_energy_ha;

    /// @brief Mean-field (RHF) orbital energies ε_p [Hartree].
    std::vector<double> mf_energy_ha;

    /// @brief HOMO→LUMO quasiparticle gap (ε_LUMO^QP − ε_HOMO^QP) [eV].
    double gap_ev = 0.0;

    /// @brief Vertical ionization potential −ε_HOMO^QP [eV].
    double ip_ev = 0.0;

    /// @brief Vertical electron affinity ε_LUMO^QP [eV] (positive ⇒ unbound
    ///        temporary anion still reported as the LUMO QP energy).
    double ea_ev = 0.0;

    /// @brief Number of occupied spatial MOs (closed-shell).
    int n_occ = 0;

    /// @brief true when every orbital's Newton QP solve converged.
    bool converged = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
