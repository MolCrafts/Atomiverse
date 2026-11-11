/// @file cpu/scf/rpa.h
/// @brief Direct RPA (dRPA) correlation energy (public, result-only header —
///        mp2.h / g0w0.h / mp3.h precedent).
///
/// User path: `DirectRPA<RHF>{mf}.run()` after `mf.run` (not a member of
/// `RHF`).
/// Correlation energy via the Furche plasmon formula on the full direct-RPA
/// (Coulomb-only, α_hf = 0) Casida spectrum:
/// \f[
///   E_{\mathrm{c}}^{\mathrm{dRPA}}
///     = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
/// \f]
/// with singlet spin-adapted \(A,B\) (\(\alpha_{\mathrm{hf}}=0\)) and
/// \(\Omega_s\) the positive eigenvalues of the Hermitian reduction
/// \((A-B)^{1/2}(A+B)(A-B)^{1/2}\). Equivalent to the exact imaginary-axis
/// frequency integral of \(\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\) without
/// density fitting (matches PySCF `gw.rpa.RPA` / `DirectRPA` in the
/// complete-basis / non-DF limit; stock PySCF uses DF + Legendre quadrature
/// and differs by ~1e-5 Ha on H₂/STO-3G).
///
/// References: Furche, J. Chem. Phys. **114**, 5982 (2001),
/// DOI 10.1063/1.1353585; Ren et al., New J. Phys. **14**, 053020 (2012);
/// Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021).
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Direct RPA correlation energy of a closed-shell RHF reference.
struct DirectRPAResult {
    /// @brief dRPA correlation energy \(E_{\mathrm{c}}\) [Hartree].
    double e_corr = 0.0;

    /// @brief Total energy = scf.energy + e_corr [Hartree].
    double e_tot = 0.0;

    /// @brief Positive dRPA excitation energies \(\Omega_s\) [Hartree],
    ///        ascending (full spectrum, length \(n_{\mathrm{occ}}\cdot
    ///        n_{\mathrm{virt}}\)).
    std::vector<double> omega_ha;

    /// @brief \(\mathrm{Tr}\,A\) of the singlet dRPA \(A\) matrix [Hartree]
    ///        (diagnostic; enters the plasmon formula).
    double tr_A = 0.0;
};

/// @brief Direct RPA correlation energy of a converged closed-shell reference.
///
/// Not a method of `RHF` or `MeanFieldMethod`. Call `mf.run` first.
/// Instantiated for `RHF`. Mirrors PySCF `gw.rpa.RPA` in the non-DF limit.
/// @tparam Method Concrete mean-field type.
template <typename Method>
class DirectRPA {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit DirectRPA(const Method& mf);

    /// @brief Furche plasmon correlation energy on the Coulomb-only Casida
    ///        spectrum (α_hf = 0).
    /// @return Correlation and total energy [Hartree], plus Ω_s [Hartree].
    /// @throws std::runtime_error when called before `mf.run`, on a
    ///         non-restricted reference, or when dRPA is ill-conditioned.
    /// @throws std::invalid_argument when n_occ = 0 or n_virt = 0.
    DirectRPAResult run() const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
