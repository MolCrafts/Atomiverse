/// @file cpu/scf/cisd.h
/// @brief CIS(D) doubles correction to TDA/CIS excitation energies (public API).
///
/// User path: `CISD<RHF>{mf}.run(n_states)` after `mf.run`
/// (not a member of `RHF`). Head-Gordon, Rico, Oumi & Lee, Chem. Phys.
/// Lett. **219**, 21 (1994), DOI 10.1016/0009-2614(94)00070-0 —
/// spin-orbital second-order doubles correction on a CIS zeroth-order
/// root (size-consistent, O(N⁵) per state). PySCF has no CIS(D) class;
/// `pyscf.ci.cisd.CISD` is ground-state CI and is not this type.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief CIS(D) excitation energies and the parent CIS roots they correct.
struct CISDResult {
    /// @brief CIS(D) excitation energies ω_CIS(D) = ω_CIS + U + V [eV],
    ///        ascending in the parent CIS order (same root indexing as
    ///        `TDA<RHF>::run`).
    std::vector<double> energy_ev;

    /// @brief Parent CIS / TDA excitation energies ω_CIS [eV].
    std::vector<double> cis_energy_ev;

    /// @brief Doubles correction U + V per root [eV]
    ///        (ω_CIS(D) − ω_CIS; may raise or lower the root).
    std::vector<double> doubles_correction_ev;
};

/// @brief CIS(D) doubles correction on a converged closed-shell RHF.
///
/// Not a method of `RHF` or `MeanFieldMethod`. Call `mf.run` first. Parent
/// singles are the same TDA roots as `TDA<RHF>::run`. PySCF 2.14 has no
/// CIS(D) product (`ci/cisd.py` `CISD(mf)` solves ground-state CI).
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class CISD {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit CISD(const Method& mf);

    /// @brief CIS(D) correction of the lowest `n_states` singlet CIS roots.
    /// @param n_states Number of lowest CIS roots to correct (1 ≤ n_states ≤
    ///                 n_occ·n_virt; dimensionless count).
    /// @return CIS(D) and parent CIS excitation energies [eV] plus the
    ///         doubles correction U+V [eV] per root.
    /// @throws std::runtime_error when called before `mf.run`.
    /// @throws std::invalid_argument when n_states is outside
    ///         [1, n_occ·n_virt] or the system has no virtual orbitals.
    CISDResult run(int n_states = 1) const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
