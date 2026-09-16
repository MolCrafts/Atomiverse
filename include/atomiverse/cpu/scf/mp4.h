/// @file cpu/scf/mp4.h
/// @brief Canonical closed-shell MP4(SDTQ) correlation energy (public API).
///
/// User path: `RHF::mp4(frame)` (RMP4 on a converged RHF reference).
/// Free helpers / pack-based entries stay private under `scf/internal/`.
///
/// The reported correlation energy is the sum of second-, third-, and
/// fourth-order Møller–Plesset corrections:
/// \f[
///   E_{\mathrm{corr}}^{\mathrm{MP4}} = E^{(2)} + E^{(3)} + E^{(4)},
/// \f]
/// with \(E^{(4)}=E_S^{(4)}+E_D^{(4)}+E_T^{(4)}+E_Q^{(4)}\) (singles, doubles,
/// triples, and disconnected-quadruples/renormalization channels).
///
/// Matches the spin-orbital residual form of Bartlett / Szabo–Ostlund
/// fourth-order PT. PySCF 2.x does not ship `mp.MP4`; the offline pin rebuilds
/// the same RSPT on PySCF RHF orbitals (determinant RSPT for the H₂ product
/// gate; equivalent algebraic SDTQ for general closed-shell systems).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Canonical closed-shell MP4(SDTQ) energy (RMP4).
struct MP4Result {
    /// @brief Total correlation \(E^{(2)}+E^{(3)}+E^{(4)}\) [Hartree].
    double e_corr = 0.0;
    /// @brief Total energy = scf.energy + e_corr [Hartree].
    double e_tot = 0.0;
    /// @brief Second-order piece \(E^{(2)}\) [Hartree] (same as plain RMP2).
    double e_corr_mp2 = 0.0;
    /// @brief Third-order piece \(E^{(3)}\) [Hartree].
    double e_corr_mp3 = 0.0;
    /// @brief Fourth-order piece \(E^{(4)}=E_S+E_D+E_T+E_Q\) [Hartree].
    double e_corr_mp4 = 0.0;
    /// @brief \(E_S^{(4)}\) singles channel [Hartree].
    double e_corr_s = 0.0;
    /// @brief \(E_D^{(4)}\) doubles channel (connected residual) [Hartree].
    double e_corr_d = 0.0;
    /// @brief \(E_T^{(4)}\) triples channel [Hartree].
    double e_corr_t = 0.0;
    /// @brief \(E_Q^{(4)}\) renormalization / disconnected-Q channel [Hartree].
    double e_corr_q = 0.0;
    /// @brief Number of lowest occupied spatial MOs frozen (0 = full).
    int frozen = 0;
};

// Public: RHF::mp4(frame). Pack entry is private (internal/mp4_pack.h).

} // namespace scf
} // namespace cpu
} // namespace ATV
