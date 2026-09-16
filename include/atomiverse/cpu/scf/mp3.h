/// @file cpu/scf/mp3.h
/// @brief Canonical closed-shell MP3 correlation energy (public API).
///
/// User path: `RHF::mp3(frame)` (RMP3 on a converged RHF reference).
/// Free helpers / pack-based entries stay private under `scf/internal/`.
///
/// The reported correlation energy is the sum of the second- and third-order
/// Møller–Plesset corrections:
/// \f[
///   E_{\mathrm{corr}}^{\mathrm{MP3}} = E^{(2)} + E^{(3)}.
/// \f]
/// Matches the spin-orbital / Szabo–Ostlund diagram sum used by Psi4
/// (`energy('mp3')`) and the Psi4NumPy reference (spatial reduction of the
/// same diagrams). PySCF 2.x does not ship `mp.MP3`; the offline pin rebuilds
/// the same formula on PySCF RHF orbitals + MO ERIs.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Canonical closed-shell MP3 energy (RMP3).
struct MP3Result {
    /// @brief Total correlation \(E^{(2)}+E^{(3)}\) [Hartree].
    double e_corr = 0.0;
    /// @brief Total energy = scf.energy + e_corr [Hartree].
    double e_tot = 0.0;
    /// @brief Second-order piece \(E^{(2)}\) [Hartree] (same as plain RMP2).
    double e_corr_mp2 = 0.0;
    /// @brief Third-order piece \(E^{(3)}\) [Hartree].
    double e_corr_mp3 = 0.0;
    /// @brief Number of lowest occupied spatial MOs frozen (0 = full).
    int frozen = 0;
};

// Public: RHF::mp3(frame). Pack entry is private (internal/mp3_pack.h).

} // namespace scf
} // namespace cpu
} // namespace ATV
