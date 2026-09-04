/// @file cpu/semiempirical/td_dftb2.h
/// @brief Excited-state result types of the linear-response TD-DFTB (Casida)
///        solver on an SCC-DFTB2 ground state (public, **result-only** header —
///        `cpu/scf/tdhf.h` precedent).
///
/// User path: `cpu::semiempirical::DFTB2::td_casida(frame, n_states, channel)`
/// (see `dftb2.h`). The Casida assembly, the Slater–Koster / γ tables and the
/// self-consistent-charge solve are private workspace of `DFTB2` and are not
/// part of this header — there is no `*Config` bag, no `*System`, no factory.
///
/// Reference: T. A. Niehaus, S. Suhai, F. Della Sala, P. Lugli, M. Elstner,
/// G. Seifert, Th. Frauenheim, "Tight-binding approach to time-dependent
/// density-functional response theory", *Phys. Rev. B* **63**, 085108 (2001),
/// DOI 10.1103/PhysRevB.63.085108.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Spin symmetry of the excitation out of a closed-shell SCC-DFTB2
///        reference (dimensionless selector).
///
/// - `singlet`: spin-conserving channel, coupling kernel
///   \(K^{S}_{ia,jb}=\sum_{AB}q^{ia}_A\gamma_{AB}q^{jb}_B\) — the same
///   \(\gamma_{AB}\) [eV] the second-order SCC energy uses.
/// - `triplet`: magnetization channel, on-site kernel
///   \(K^{T}_{ia,jb}=\sum_{A}q^{ia}_A W_A q^{jb}_A\). \(W_A\) [eV] is the
///   highest-occupied atomic orbital (HOAO) diagonal of
///   `DFTB2::spin_w_ha` (H \(W_{ss}\), C/N/O \(W_{pp}\)) converted Ha→eV;
///   `spin_w_ha` itself is
///   [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV)
///   and must be set or `DFTB2::td_casida` throws. Same collapse as
///   collinear ground-state `launch()`.
enum class TdDftbChannel { singlet, triplet };

/// @brief Vertical excitation energies and the assembled Casida quantities of
///        one TD-DFTB linear-response solve.
///
/// Vertical excitation energies, Casida matrix, and oscillator strengths.
/// Excited-state gradients are still out of scope. Every field is a plain
/// value type; nothing here is an input parameter bag.
struct TdDftbResult {
    /// @brief Excitation energies \(\omega_n\) [eV], ascending,
    ///        `n_states` entries.
    std::vector<double> energy_ev;

    /// @brief Dimensionless oscillator strengths \(f_n\) (length form),
    ///        `n_states` entries, same order as `energy_ev`.
    ///
    /// Singlet (Casida \(\Omega F=\omega^2 F\), \(F^\top F=I\)):
    /// \(f_I=\frac{2}{3\omega_I}\big|\sum_{ia}\sqrt{2}\,F^I_{ia}
    /// \sqrt{\varepsilon_a-\varepsilon_i}\,\boldsymbol{\mu}_{ia}\big|^2\)
    /// with pair dipoles \(\boldsymbol{\mu}_{ia}=\sum_A q^{ia}_A\mathbf{R}_A\)
    /// [e·Å]. Triplet channel is spin-forbidden in the dipole approximation
    /// → \(f_I=0\).
    std::vector<double> oscillator_strength;

    /// @brief Transition dipoles \(\boldsymbol{\mu}_I\) [e·Å] for each state,
    ///        row-major `[n_states, 3]` (x,y,z). Zero for triplets.
    std::vector<double> transition_dipole;

    /// @brief The `n_states` lowest single-particle (Kohn–Sham-like) gaps
    ///        \(\varepsilon_a-\varepsilon_i\) [eV], ascending.
    ///
    /// At `coupling_scale == 0` the coupling kernel vanishes and
    /// \(\Omega\) is diagonal with entries \((\varepsilon_a-\varepsilon_i)^2\),
    /// so `energy_ev` equals this vector to solver precision. That identity is
    /// the structural check on the whole assembly.
    std::vector<double> ks_gap_ev;

    /// @brief Occupied molecular-orbital index \(i\) of each \(ia\) pair
    ///        (dimensionless, 0-based, ascending-energy MO ordering),
    ///        `n_pair` entries.
    std::vector<int> pair_occ;

    /// @brief Virtual molecular-orbital index \(a\) of each \(ia\) pair
    ///        (dimensionless, 0-based), `n_pair` entries.
    std::vector<int> pair_virt;

    /// @brief Mulliken transition charges \(q^{ia}_A\) [e], row-major
    ///        `[n_pair, n_atoms]`:
    ///        \(q^{ia}_A=\tfrac12\sum_{\mu\in A}
    ///        (c_{\mu i}(Sc_a)_\mu+c_{\mu a}(Sc_i)_\mu)\).
    ///
    /// Orthogonality of the molecular orbitals makes
    /// \(\sum_A q^{ia}_A=\delta_{ia}=0\) for every occupied→virtual pair.
    std::vector<double> transition_charge;

    /// @brief The assembled Casida matrix \(\Omega\) [eV²], row-major
    ///        `[n_pair, n_pair]`:
    ///        \(\Omega_{ia,jb}=\delta_{ij}\delta_{ab}
    ///        (\varepsilon_a-\varepsilon_i)^2
    ///        +2\sqrt{\varepsilon_a-\varepsilon_i}\,K_{ia,jb}\,
    ///        \sqrt{\varepsilon_b-\varepsilon_j}\).
    ///
    /// Returned exactly as assembled (never re-symmetrized), so its symmetry
    /// is a real check rather than a tautology.
    std::vector<double> casida_ev2;

    /// @brief Number of occupied→virtual pairs in the active window
    ///        (dimensionless).
    int n_pair = 0;

    /// @brief Number of atoms the transition charges are resolved over
    ///        (dimensionless).
    int n_atoms = 0;

    /// @brief Which spin channel was solved (dimensionless selector).
    TdDftbChannel channel = TdDftbChannel::singlet;

    /// @brief Scale \(\lambda\) applied to the coupling kernel \(K\)
    ///        (dimensionless). 1 = the physical TD-DFTB response; 0 = the
    ///        non-interacting limit (bare single-particle gaps).
    double coupling_scale = 1.0;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
