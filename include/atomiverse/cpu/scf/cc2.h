/// @file cpu/scf/cc2.h
/// @brief Canonical closed-shell CC2 correlation energy (public API).
///
/// User path: `RHF::cc2(frame)` after (or driving) a closed-shell RHF
/// reference. Approximate coupled-cluster doubles of Christiansen, Koch &
/// Jørgensen, Chem. Phys. Lett. **243**, 409 (1995): full singles residual,
/// doubles residual truncated to at most linear in T₂ with the (T₁-dressed)
/// Fock operator. Matches PySCF `cc.RCCSD` with `cc2=True` on the same RHF
/// orbitals (stock PySCF has no top-level `cc.CC2` factory — the flag is the
/// product oracle).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CC2 energy on a converged RHF reference.
struct CC2Result {
    /// @brief CC2 correlation energy E_CC2 − E_HF [Hartree] (≤ 0).
    double e_corr = 0.0;
    /// @brief Total energy E_HF + E_corr [Hartree].
    double e_tot = 0.0;
    /// @brief Jacobi micro-iterations used to converge the amplitudes.
    int iterations = 0;
    /// @brief True when max|Δt₁| and max|Δt₂| fell below the internal tol.
    bool converged = false;
};

// Pack-based entry is private — see internal/ccsd_pack.h (`cc2_energy`).
// Public: MeanFieldMethod::cc2(frame).

} // namespace scf
} // namespace cpu
} // namespace ATV
