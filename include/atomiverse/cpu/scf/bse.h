/// @file cpu/scf/bse.h
/// @brief Bethe–Salpeter equation (BSE) excitation energies (public,
///        result-only header — tdhf.h / g0w0.h precedent).
///
/// User path: `BSE<RHF>{mf}.run(n_states, tda)` after `mf.run` (not a member
/// of `RHF`). Spectral G0W0 quasiparticle poles replace the mean-field
/// orbital energies in the ph/hp Casida matrices; the particle–hole
/// attraction is the **static direct-RPA** screened Coulomb \(W\) (exact
/// 4-centre form, no density fitting). Matches the PySCF
/// `pyscf.gw.bse.BSE` working equations (Hillenbrand, Li & Zhu, JCP 162,
/// 174117 (2025); J. Comput. Chem. 38, 383 (2017)) evaluated with
/// frequency-exact G0W0 poles (`G0W0<RHF>::run` / `gw.gw_exact.GWExact`) and
/// non-DF ERIs.
///
/// Singlet channel (closed-shell spatial MOs):
/// \f{align*}{
///   A_{ia,jb} &= (\varepsilon_a^{\mathrm{QP}}-\varepsilon_i^{\mathrm{QP}})
///               \delta_{ij}\delta_{ab} + 2(ia|jb) - W_{ij,ab} \\
///   B_{ia,jb} &= 2(ia|jb) - W_{ib,aj}
/// \f}
/// with static RPA \(W = v + v\,\Pi\,v\),
/// \(\Pi = 4\,(D^{-1}-4v_{\mathrm{ph}})^{-1}\),
/// \(D^{-1}_{ia}=\varepsilon_i^{\mathrm{QP}}-\varepsilon_a^{\mathrm{QP}}\).
#pragma once

#include <vector>

#include "atomiverse/cpu/scf/g0w0.h"
#include "atomiverse/cpu/scf/tdhf.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Singlet BSE@G0W0 excitation spectrum of a closed-shell RHF reference.
struct BSEResult {
    /// @brief Excitation energies ω_0n [eV], ascending (lowest `n_states`).
    std::vector<double> energy_ev;

    /// @brief true = Tamm–Dancoff BSE (B = 0); false = full BSE/RPA.
    bool tda = false;

    /// @brief Spin channel (v1: singlet only).
    ExcitationChannel channel = ExcitationChannel::singlet;

    /// @brief G0W0 quasiparticle data used for the poles and static W.
    G0W0Result gw;
};

/// @brief Singlet BSE@G0W0 on a converged closed-shell mean-field reference.
///
/// Not a method of `RHF` or `MeanFieldMethod`. Call `mf.run` first.
/// Instantiated for `RHF`. Mirrors `pyscf.gw.bse.BSE`.
/// @tparam Method Concrete mean-field type.
template <typename Method>
class BSE {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit BSE(const Method& mf);

    /// @brief Lowest singlet BSE roots.
    /// @param n_states Number of roots in `[1, n_occ·n_virt]`.
    /// @param tda If true, Tamm–Dancoff (B = 0); else full BSE.
    /// @return Excitation energies [eV] plus the underlying G0W0 result.
    /// @throws std::runtime_error when called before `mf.run`, on a
    ///         non-restricted reference, or when G0W0 / BSE is ill-conditioned.
    /// @throws std::invalid_argument when `n_states` is out of range or the
    ///         system has no virtual orbitals.
    BSEResult run(int n_states = 1, bool tda = false) const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
