/// @file cpu/scf/internal/mean_field_energy.h
/// @brief The four-branch mean-field total-energy functional evaluated at a
///        GIVEN pair of spin densities (restricted/unrestricted × HF/KS).
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// This is the energy expression `DensityStage` evaluates once per SCF
/// iteration, lifted out verbatim because Car–Parrinello needs the same
/// functional at a density that no SCF iteration produced (the propagated
/// orbitals of `electronic::Cpmd`). The two callers are the reason it exists:
/// a single-use block would have stayed inline in the stage.
///
/// **The two density pairs are not redundant.** Inside the SCF loop the
/// iteration's Fock matrices were built from the *previous* density while the
/// diagonalization has already produced a *new* one, and the two branches read
/// different ones:
///   - HF: \(E = \tfrac12\mathrm{Tr}[P_{\mathrm{fock}}(H_{\mathrm{core}}+F)]\)
///     — the half-trace identity is only valid on the density that produced
///     \(F\);
///   - KS: \(E = \mathrm{Tr}[H_{\mathrm{core}}P_{\mathrm{out}}]
///     + \tfrac12\mathrm{Tr}[J P_{\mathrm{out}}] + \dots\) — the shipped
///     convention, which pairs the output density with the previous
///     iteration's \(J\) / \(E_{xc}\) (the two coincide at convergence).
/// A fixed-density evaluation (CPMD) passes the SAME pair twice, which is what
/// makes the expression the ordinary textbook energy at that density.
///
/// References: Szabo & Ostlund, *Modern Quantum Chemistry*, eq. (3.184)
/// (RHF/UHF half-trace energy); Parr & Yang, *DFT of Atoms and Molecules*,
/// Ch. 7 (Kohn–Sham energy decomposition).
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief The decomposed mean-field energy at one density.
struct MeanFieldEnergy {
    double total = 0;        ///< Total energy \(E\) [Hartree].
    double one_electron = 0; ///< \(\mathrm{Tr}[H_{\mathrm{core}}P]\) [Hartree].
    /// @brief Coulomb energy \(\tfrac12\mathrm{Tr}[J P]\) [Hartree]. Exactly 0
    ///        on the HF branches, where the electron repulsion is never split
    ///        out of the \(\tfrac12\mathrm{Tr}[P(H+F)]\) half-trace.
    double coulomb = 0;
};

/// @brief Mean-field total energy at a given density (no SCF iteration).
///
/// Branches on `restricted` × `is_hf`; every branch adds the classical nuclear
/// repulsion and the continuum-solvent terms. The restricted convention is the
/// loop's: the α channel carries the FACTOR-2 closed-shell density and
/// \(P_\beta\) aliases it.
///
/// @param restricted  true → one spatial channel (RHF/RKS); false → α/β.
/// @param is_hf       true → Hartree–Fock; false → Kohn–Sham.
/// @param Hcore       Core Hamiltonian [Hartree], n × n (environment already
///                    folded in by the caller).
/// @param F_alpha     α Fock matrix [Hartree], built from `P_alpha_fock`.
/// @param F_beta      β Fock matrix [Hartree] (unrestricted HF only).
/// @param J           Coulomb matrix of the total density [Hartree] (KS only).
/// @param P_alpha_fock α density the Fock matrices were built from
///                    (dimensionless; factor-2 total when `restricted`).
/// @param P_beta_fock β density the Fock matrices were built from.
/// @param P_alpha_out α density this evaluation reports on (KS branches);
///                    equal to `P_alpha_fock` at a fixed density.
/// @param P_beta_out  β density this evaluation reports on (KS branches).
/// @param e_xc_grid   Grid-integrated \(E_{xc}\) [Hartree] (KS only).
/// @param e_exact_exchange Hybrid admixture \(\alpha_{\mathrm{HF}}E_K\)
///                    [Hartree] (KS global hybrids only).
/// @param v_nuc       Nuclear-repulsion (+ classical environment) [Hartree].
/// @param e_solvent   Continuum electrostatic free energy ΔG_el [Hartree].
/// @param e_pcm_fock_half \(\tfrac12\mathrm{Tr}[V_{\mathrm{pcm}}P]\) [Hartree],
///                    peeled out of the HF half-trace so ΔG_el replaces it.
/// @return Total / one-electron / Coulomb energies [Hartree].
MeanFieldEnergy mean_field_energy_at_density(
    bool restricted, bool is_hf, const RealMatrix& Hcore,
    const RealMatrix& F_alpha, const RealMatrix& F_beta, const RealMatrix& J,
    const RealMatrix& P_alpha_fock, const RealMatrix& P_beta_fock,
    const RealMatrix& P_alpha_out, const RealMatrix& P_beta_out,
    double e_xc_grid, double e_exact_exchange, double v_nuc, double e_solvent,
    double e_pcm_fock_half);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
