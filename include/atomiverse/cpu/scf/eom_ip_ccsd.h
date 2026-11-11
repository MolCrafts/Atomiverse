/// @file cpu/scf/eom_ip_ccsd.h
/// @brief EOM-IP-CCSD ionization potentials (public API).
///
/// User path: `CCSD<RHF>{rhf}.eom_ip_ccsd(frame, n_states)` — drives closed-shell RHF,
/// spin-orbital Gauss–Stanton CCSD, then EOM-IP on the similarity-transformed
/// Hamiltonian (Stanton & Bartlett, J. Chem. Phys. **98**, 7029 (1993);
/// Tu, Wang & Li, J. Chem. Phys. **136**, 174102 (2012)). Matches PySCF
/// `cc.CCSD(...).ipccsd()` / `eom_gccsd.EOMIP` on the same RHF reference.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief EOM-IP-CCSD ionization potentials on a closed-shell
///        RHF → CCSD reference.
struct EOMIPCCSDResult {
    /// @brief Ionization potentials ω = E_CCSD(N) − E(N−1) [eV], ascending
    ///        (dense non-symmetric eigen-solve of the spin-orbital EOM-IP
    ///        Jacobian; spin-degenerate roots are de-duplicated).
    std::vector<double> energy_ev;

    /// @brief CCSD correlation energy E_CCSD − E_HF [Hartree].
    double e_corr_ha = 0.0;
    /// @brief CCSD total energy E_HF + E_corr [Hartree].
    double e_ccsd_ha = 0.0;
    /// @brief Jacobi micro-iterations used to converge the CCSD amplitudes.
    int ccsd_iterations = 0;
    /// @brief True when CCSD amplitudes converged and EOM roots were obtained.
    bool converged = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
