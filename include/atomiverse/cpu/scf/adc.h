/// @file cpu/scf/adc.h
/// @brief ADC(2) / ADC(2)-x excitation energies (public API, result-only).
///
/// User path: `ADC<RHF>{mf}.adc2` / `adc2x` / `ip_adc2` / `ea_adc2` after
/// `mf.run` (not a member of `RHF`). Mirrors `pyscf.adc.ADC(mf)`:
/// EE is `method_type='ee'` (`adc(2)` or `adc(2)-x`); IP and EA are
/// `method_type='ip'` / `'ea'`.
///
/// EE path: intermediate-state representation of the PP
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

/// @brief ADC(2) family on a converged closed-shell mean-field reference.
///
/// Not a method of `RHF` or `MeanFieldMethod`. Call `mf.run` first.
/// Instantiated for `RHF`.
/// @tparam Method Concrete mean-field type.
template <typename Method>
class ADC {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit ADC(const Method& mf);

    /// @brief Strict ADC(2) singlet excitation energies.
    /// @param n_states Number of lowest roots (1 ≤ n_states ≤ n_occ·n_virt).
    /// @return Excitation energies [eV] and the MP2 correlation energy [Ha].
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is out of range or the
    ///         system has no virtual orbitals.
    ADC2Result adc2(int n_states = 1) const;

    /// @brief ADC(2)-x singlet excitation energies (first-order doubles block).
    /// @param n_states Number of lowest roots (1 ≤ n_states ≤ n_occ·n_virt).
    /// @return Excitation energies [eV] and the MP2 correlation energy [Ha].
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is out of range or the
    ///         system has no virtual orbitals.
    ADC2Result adc2x(int n_states = 1) const;

    /// @brief Strict ADC(2) ionization potentials.
    /// @param n_states Number of lowest IP roots (1 ≤ n_states ≤ n_occ).
    /// @return Ionization potentials [eV] and the MP2 correlation energy [Ha].
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is outside [1, n_occ] or
    ///         the system has no virtual orbitals.
    IPADC2Result ip_adc2(int n_states = 1) const;

    /// @brief Strict EA-ADC(2) electron-attachment energies.
    /// @param n_states Number of lowest EA roots
    ///                 (1 ≤ n_states ≤ n_virt + n_occ·n_virt²).
    /// @return Attachment energies [eV] and the MP2 correlation energy [Ha].
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is out of range or the
    ///         system has no virtual orbitals.
    EAADC2Result ea_adc2(int n_states = 1) const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
