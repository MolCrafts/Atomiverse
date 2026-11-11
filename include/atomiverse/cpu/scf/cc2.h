/// @file cpu/scf/cc2.h
/// @brief Canonical closed-shell CC2 correlation energy (public API).
///
/// User path: `CC2<RHF>{rhf}.run(frame)` after (or driving) a closed-shell
/// RHF reference. Approximate coupled-cluster doubles of Christiansen, Koch
/// & Jørgensen, Chem. Phys. Lett. **243**, 409 (1995): full singles residual,
/// doubles residual truncated to at most linear in T₂. The product loop is
/// `cpu::scf::CC2` (`Driver<cc::Phase>`), PySCF `cc.RCCSD` with `cc2=True`
/// (stock PySCF has no top-level `cc.CC2` factory — the flag is the product
/// oracle). Hartree stays on `CC2Result`; `run(Frame&)` writes eV to
/// `kEnergy`.
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

// The product solve is `CC2::run` in post_hf.h. EOM-CC2 amplitudes stay on
// the spin-orbital pack (`ccsd_solve(..., cc2=true)`).

} // namespace scf
} // namespace cpu
} // namespace ATV
