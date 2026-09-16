/// @file cpu/scf/mp2.h
/// @brief Canonical MP2 correlation energy (public API).
///
/// User path: `RHF::mp2(frame)` (closed-shell RMP2 / RI-MP2),
/// `UHF::mp2(frame)` (UMP2), or `ROHF::mp2(frame)` (ROMP2 = UMP2 on ROHF
/// orbitals with semi-canonical spin-Fock denominators; PySCF convention).
/// This header exposes only the `MP2Result` value type; the AO→MO transform
/// kernels are private workspace declared in `internal/ao2mo.h`.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Canonical or density-fitted MP2 energy (RMP2 / UMP2 / ROMP2 / RI-MP2 / SCS).
///
/// Spin-component channels `e_corr_os` / `e_corr_ss` are the **unscaled**
/// opposite-spin and same-spin pieces of the doubles sum (PySCF
/// `e_corr_os` / `e_corr_ss`).  The reported `e_corr` applies optional
/// Grimme scales from the method members `mp2_os_scale` / `mp2_ss_scale`:
///
/// \f[
///   E_{\mathrm{corr}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}
///                     + c_{\mathrm{SS}}\,E_{\mathrm{SS}}.
/// \f]
///
/// Defaults \(c_{\mathrm{OS}}=c_{\mathrm{SS}}=1\) recover plain MP2.
/// SCS-MP2 (Grimme, J. Chem. Phys. 118, 9095 (2003)): \(c_{\mathrm{OS}}=1.2\),
/// \(c_{\mathrm{SS}}=1/3\). SOS-MP2: \(c_{\mathrm{OS}}=1.3\), \(c_{\mathrm{SS}}=0\).
struct MP2Result {
    double e_corr; ///< Scaled MP2 correlation energy [Hartree].
    double e_tot;  ///< Total energy = scf.energy + e_corr [Hartree].
    /// @brief true when the correlation energy used density-fitted (ia|jb).
    bool ri = false;
    /// @brief Number of lowest occupied spatial MOs frozen in the sum
    ///        (0 = full valence+core correlation).
    int frozen = 0;
    /// @brief Unscaled opposite-spin correlation [Hartree].
    double e_corr_os = 0.0;
    /// @brief Unscaled same-spin correlation [Hartree].
    double e_corr_ss = 0.0;
};

// The AO→MO integral transforms are private post-HF workspace — declared in
// internal/ao2mo.h, not on this public path.
//
// Pack-based entry is private implementation — declared in
// internal/mp2_pack.h if needed by unit tests.
// Public: RHF::mp2(frame) / UHF::mp2(frame) / ROHF::mp2(frame).

} // namespace scf
} // namespace cpu
} // namespace ATV
