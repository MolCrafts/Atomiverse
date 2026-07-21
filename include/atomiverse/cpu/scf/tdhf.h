/// @file cpu/scf/tdhf.h
/// @brief Excited-state result types of the TDHF/RPA and TDA (CIS) solvers
///        (public, result-only header — mp2.h precedent).
///
/// User path: `RHF::tda(n_states, channel)` / `RHF::tdhf(n_states, channel)`
/// (see rhf.h). The eigenproblem machinery is private workspace
/// (`internal::TDHFPack` over the link-08 `internal::OrbitalHessian` A/B
/// kernel) and is not part of this header.
///
/// References: Dreuw & Head-Gordon, Chem. Rev. 105, 4009 (2005),
/// DOI 10.1021/cr0505627 (TDHF/RPA response formalism); Hirata &
/// Head-Gordon, Chem. Phys. Lett. 314, 291 (1999),
/// DOI 10.1016/S0009-2614(99)01149-5 (TDA); Szabo & Ostlund §7.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Spin symmetry of the excitation out of a closed-shell reference
///        (dimensionless selector): `singlet` (spin-conserving, dipole-
///        allowed) or `triplet` (spin-flip manifold, dipole-forbidden from
///        a singlet ground state).
enum class ExcitationChannel { singlet, triplet };

/// @brief Excitation energies and length-gauge transition properties of a
///        TDA or TDHF/RPA solve on a converged closed-shell reference.
struct ExcitedStatesResult {
    /// @brief Excitation energies ω_0n [eV], ascending.
    std::vector<double> energy_ev;

    /// @brief Length-gauge transition dipole moments μ_0n [a.u., e·Bohr],
    ///        shape [n_states, 3]. Singlet channel:
    ///        μ_0n = √2 Σ_ia (X+Y)_ia ⟨φ_i|r|φ_a⟩ (overall sign is the
    ///        arbitrary eigenvector phase); triplet channel: exactly zero
    ///        (spin-forbidden from the singlet ground state).
    xt::xtensor<double, 2> transition_dipole_au;

    /// @brief Length-gauge oscillator strengths (dimensionless):
    ///        f_0n = (2/3)·ω_0n[Ha]·|μ_0n|²[a.u.]. Zero for the triplet
    ///        channel.
    std::vector<double> oscillator_strength;

    /// @brief Which spin channel was solved.
    ExcitationChannel channel = ExcitationChannel::singlet;

    /// @brief true = Tamm-Dancoff approximation (B = 0; CIS on an HF
    ///        reference), false = full TDHF/RPA (dimensionless flag).
    bool tda = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
