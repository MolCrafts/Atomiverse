/// @file cpu/scf/g0w0.h
/// @brief Spectral G0W0 on a closed-shell RHF reference (result + the
///        `G0W0` type; not a member of `RHF`).
///
/// User path: `G0W0<RHF>{mf}.run()` after `mf.run` (not a member of
/// `RHF`). Spectral (frequency-exact) G0W0 with direct-RPA (dRPA)
/// screening of W — `pyscf.gw.gw_exact.GWExact(mf).kernel`, which builds
/// `tdscf.dRPA` inside the kernel when no TD object was supplied. HF
/// reference (the `xc='hf'` density `GWExact` requires). Quasiparticle
/// equation solved orbit-by-orbit (non-linear Newton, not linearized).
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

/// @brief Spectral G0W0 quasiparticle energies of a converged closed-shell RHF.
///
/// Mirrors `pyscf.gw.gw_exact.GWExact`. `run` is `kernel`: dRPA screening
/// is built inside the call (`GWExact.kernel` constructs `tdscf.dRPA` when
/// `tdmf` is None). Not a method of `RHF` or `MeanFieldMethod`. Call
/// `mf.run` first.
/// @tparam Method Concrete mean-field type. Instantiated for `RHF`.
template <typename Method>
class G0W0 {
  public:
    /// @brief Bind a mean-field reference. The reference must outlive `*this`.
    /// @param mf Mean-field method that has already been run (not copied).
    explicit G0W0(const Method& mf);

    /// @brief Diagonal G0W0 quasiparticle spectrum.
    /// @return QP energies [Hartree], HOMO–LUMO gap / IP / EA [eV].
    /// @throws std::runtime_error when called before `mf.run`, on a
    ///         non-restricted reference, or when dRPA is ill-conditioned.
    /// @throws std::invalid_argument when n_occ = 0 or n_virt = 0.
    G0W0Result run() const;

  private:
    const Method* mf_ = nullptr;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
