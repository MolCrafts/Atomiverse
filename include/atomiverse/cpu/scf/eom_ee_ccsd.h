/// @file cpu/scf/eom_ee_ccsd.h
/// @brief EOM-EE-CCSD singlet excitation energies (public API).
///
/// User path: `CCSD<RHF>{rhf}.eom_ee_ccsd(frame, n_states)` — drives
/// closed-shell RHF, spin-orbital Gauss–Stanton CCSD, then EOM-EE on the
/// similarity-transformed Hamiltonian (Wang, Tu & Wang, JCTC **10**, 5567
/// (2014); Stanton & Bartlett, J. Chem. Phys. **98**, 7029 (1993)). Matches
/// PySCF `cc.CCSD(mf).eomee_ccsd_singlet()`, which constructs
/// `eom_rccsd.EOMEESinglet`. Not a member of `RHF`.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Singlet EOM-EE-CCSD electronic excitation energies on a closed-shell
///        RHF → CCSD reference.
struct EOMEECCSDResult {
    /// @brief Singlet excitation energies ω [eV], ascending (dense non-symmetric
    ///        eigen-solve of the spin-orbital EOM-EE Jacobian, singlet-selected).
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
