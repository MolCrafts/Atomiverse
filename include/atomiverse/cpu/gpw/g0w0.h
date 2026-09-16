/// @file cpu/gpw/g0w0.h
/// @brief GPW spectral G0W0 quasiparticle result (public value type).
///
/// User path: `GpwDft::g0w0(atoms)` after a converged `run()` — the GPW
/// analogue of molecular `RHF::g0w0()` (`cpu/scf/g0w0.h`). Supports:
/// - **Γ RKS** (real MOs; RI-GPW or Poisson) and **multi-k RKS**
///   (complex Bloch + **finite-\(q\) inter-\(k\)** dRPA; RI-GPW product);
/// - **Γ / multi-k UKS** unrestricted spectral G0W0 (UGW): spin-conserving
///   dRPA over α→α ⊕ β→β, same-spin \(G_\sigma W\), MT \(\Sigma_x^\sigma\),
///   spin \(v_{\mathrm{xc}}^\sigma\) (PADE / PBE / PBEsol / RevPBE / **RPBE**
///   / BLYP / optB88-vdW via the same grid kernels as UKS SCF). Multi-k uses
///   finite-\(q\) inter-\(k\) \(W\) (same dielectric as multi-k RKS G0W0)
///   with spins explicit.
///
/// Screening is direct RPA (dRPA) on the same dielectric path as
/// `GpwDft::direct_rpa` / UMP2 RI tensors; the correlation self-energy is
/// the spectral (frequency-exact) sum over RPA poles. Exact-exchange
/// diagonals use truncated (MT) Coulomb so the product is G=0-safe without
/// a periodic-HF path.
///
/// Quasiparticle equation (Newton, orbit-by-orbit, per \(k\)):
/// \f[
///   \omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega)
///            + \Sigma_{x,p} - v_{\mathrm{xc},p}.
/// \f]
///
/// References: Hybertsen & Louie, Phys. Rev. B **34**, 5390 (1986);
/// Furche, J. Chem. Phys. **114**, 5982 (2001); molecular twin
/// `RHF::g0w0` / PySCF `gw.gw_exact.GWExact` form.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Diagonal spectral G0W0 quasiparticle energies of a GPW KS
///        reference (Γ/multi-k RKS, or Γ/multi-k UKS UGW).
struct G0W0Result {
    /// @brief Quasiparticle orbital energies \(\varepsilon_p^{\mathrm{QP}}\)
    ///        [Hartree], same order as the KS MO spectrum (not re-sorted).
    ///        Multi-k: spectrum at the \(k\) realising the min direct QP gap.
    ///        UKS: **α-channel** spectrum (primary report; dual-gates RKS
    ///        when \(n_\alpha=n_\beta\) and \(C^\alpha\approx C^\beta\)).
    std::vector<double> qp_energy_ha;

    /// @brief Mean-field (KS) orbital energies \(\varepsilon_p\) [Hartree].
    ///        Multi-k: same \(k\) as `qp_energy_ha`. UKS: α-channel.
    std::vector<double> mf_energy_ha;

    /// @brief Exact-exchange diagonals \(\Sigma_{x,p}\) [Hartree].
    ///        UKS: α-channel MT exchange.
    std::vector<double> sigma_x_ha;

    /// @brief Local XC diagonals \(v_{\mathrm{xc},p}\) [Hartree].
    ///        UKS: \(\langle p_\alpha|v_{\mathrm{xc}}^\alpha|p_\alpha\rangle\).
    std::vector<double> vxc_ha;

    /// @brief HOMO→LUMO quasiparticle gap [eV].
    ///        Multi-k: **minimum direct** gap over the sampled mesh.
    ///        UKS: α HOMO→LUMO QP gap.
    double gap_ev = 0.0;

    /// @brief KS HOMO→LUMO gap [eV] (diagnostic).
    ///        Multi-k: minimum direct KS gap over the mesh. UKS: α.
    double mf_gap_ev = 0.0;

    /// @brief Vertical ionization potential \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\)
    ///        [eV] on the reported spectrum (\(k\) of min direct QP gap / α).
    double ip_ev = 0.0;

    /// @brief Vertical electron affinity \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\)
    ///        [eV] on the reported spectrum.
    double ea_ev = 0.0;

    /// @brief Occupied count on the reported spectrum (RKS: spatial;
    ///        UKS: \(n_\alpha\)).
    int n_occ = 0;

    /// @brief Virtuals in the dRPA screening space (after cap).
    ///        UKS: \(n_{\mathrm{virt}}^\alpha + n_{\mathrm{virt}}^\beta\).
    int n_virt = 0;

    /// @brief true when `(ia|jb)` used the RI-GPW product path.
    bool ri = false;

    /// @brief Sampled k-points used (0 = real Γ path; multi-k count otherwise).
    int n_kpts = 0;

    /// @brief true when every orbital's Newton QP solve converged
    ///        (all \(k\) in the multi-k path).
    bool converged = false;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
