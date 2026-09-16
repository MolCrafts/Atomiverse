/// @file cpu/scf/ccsd.h
/// @brief Canonical closed-shell CCSD / CCSD(T) correlation energy (public API).
///
/// User path: `RHF::ccsd(frame)` and `RHF::ccsd_t(frame)` after (or driving)
/// a closed-shell RHF reference. Spin-orbital Gauss–Stanton intermediates
/// (J. Chem. Phys. **103**, 3561 (1995); Stanton et al., J. Chem. Phys.
/// **94**, 4334 (1991)); (T) from Raghavachari et al., Chem. Phys. Lett.
/// **157**, 479 (1989) / Watts, Gauss, Bartlett, J. Chem. Phys. **98**, 8718
/// (1993). Matches PySCF `cc.CCSD` / `ccsd_t()` on the same RHF orbitals.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CCSD energy on a converged RHF reference.
struct CCSDResult {
    /// @brief CCSD correlation energy E_CCSD − E_HF [Hartree] (≤ 0).
    double e_corr = 0.0;
    /// @brief Total energy E_HF + E_corr [Hartree].
    double e_tot = 0.0;
    /// @brief Jacobi micro-iterations used to converge the amplitudes.
    int iterations = 0;
    /// @brief True when max|Δt₁| and max|Δt₂| fell below the internal tol.
    bool converged = false;
};

/// @brief Closed-shell CCSD(T) energy on a converged RHF reference.
///
/// \f$E_{\mathrm{corr}} = E_{\mathrm{CCSD}} + E_{(T)}\f$, with
/// \f$E_{(T)}\f$ the non-iterative triples correction (Raghavachari 1989).
struct CCSDTResult {
    /// @brief Total correlation E_CCSD + E_(T) [Hartree] (≤ 0).
    double e_corr = 0.0;
    /// @brief Total energy E_HF + e_corr [Hartree].
    double e_tot = 0.0;
    /// @brief CCSD-only correlation E_CCSD − E_HF [Hartree].
    double e_corr_ccsd = 0.0;
    /// @brief Perturbative triples correction E_(T) [Hartree].
    double e_t = 0.0;
    /// @brief Jacobi micro-iterations used to converge the CCSD amplitudes.
    int iterations = 0;
    /// @brief True when the underlying CCSD amplitudes converged.
    bool converged = false;
};

// Pack-based entries are private — see internal/ccsd_pack.h.
// Public: MeanFieldMethod::ccsd(frame), MeanFieldMethod::ccsd_t(frame).

} // namespace scf
} // namespace cpu
} // namespace ATV
