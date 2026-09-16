/// @file cpu/scf/eom_cc2.h
/// @brief EOM-CC2 singlet excitation energies (public API).
///
/// User path: `RHF::eom_cc2(frame, n_states)` — drives closed-shell RHF,
/// spin-orbital Christiansen–Koch–Jørgensen CC2 ground state, then the
/// EOM-EE Jacobian of the similarity-transformed Hamiltonian on those
/// amplitudes (same right-hand σ as EOM-EE-CCSD / PySCF `eom_gccsd`,
/// evaluated at the CC2 T₁/T₂). Matches PySCF
/// `cc.RCCSD(mf); my.cc2=True; my.kernel(); my.eomee_ccsd_singlet()`.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Singlet EOM-CC2 electronic excitation energies on a closed-shell
///        RHF → CC2 reference.
struct EOMCC2Result {
    /// @brief Singlet excitation energies ω [eV], ascending (dense non-symmetric
    ///        eigen-solve of the spin-orbital EOM-EE Jacobian at CC2 amplitudes,
    ///        singlet-selected).
    std::vector<double> energy_ev;

    /// @brief CC2 correlation energy E_CC2 − E_HF [Hartree].
    double e_corr_ha = 0.0;
    /// @brief CC2 total energy E_HF + E_corr [Hartree].
    double e_cc2_ha = 0.0;
    /// @brief Jacobi micro-iterations used to converge the CC2 amplitudes.
    int cc2_iterations = 0;
    /// @brief True when CC2 amplitudes converged and EOM roots were obtained.
    bool converged = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
