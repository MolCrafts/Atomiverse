/// @file cpu/scf/eom_ea_ccsd.h
/// @brief EOM-EA-CCSD electron-attachment energies (public API).
///
/// User path: `RHF::eom_ea_ccsd(frame, n_states)` — drives closed-shell RHF,
/// spin-orbital Gauss–Stanton CCSD, then EOM-EA on the similarity-transformed
/// Hamiltonian (Nooijen & Bartlett, J. Chem. Phys. **102**, 3629 (1995);
/// Stanton & Bartlett, J. Chem. Phys. **98**, 7029 (1993)). Matches PySCF
/// `cc.CCSD(...).eaccsd()` / `eom_gccsd.EOMEA` on the same RHF reference.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief EOM-EA-CCSD electron-attachment energies on a closed-shell
///        RHF → CCSD reference.
struct EOMEACCSDResult {
    /// @brief Attachment energies ω = E(N+1) − E_CCSD(N) [eV], ascending
    ///        (dense non-symmetric eigen-solve of the spin-orbital EOM-EA
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
