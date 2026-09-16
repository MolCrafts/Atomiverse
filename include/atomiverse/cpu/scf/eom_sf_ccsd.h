/// @file cpu/scf/eom_sf_ccsd.h
/// @brief EOM-SF-CCSD spin-flip excitation energies (public API).
///
/// User path: `UHF::eom_sf_ccsd(frame, n_states)` — drives high-spin UHF,
/// unrestricted spin-orbital CCSD (UCCSD), then EOM spin-flip on the
/// similarity-transformed Hamiltonian (Krylov / Krylov & Sherrill SF-EOM-CC;
/// PySCF `cc.UCCSD(...).eomsf_ccsd()`). Targets ΔMs = −1 (or +1) from a
/// high-spin reference (e.g. O₂ triplet → Ms = 0 singlets/triplets).
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Spin-flip EOM-CCSD excitation energies on a UHF → UCCSD reference.
struct EOMSFCCSDResult {
    /// @brief SF excitation energies ω [eV], ascending (dense non-symmetric
    ///        eigen-solve of the spin-orbital EOM Jacobian, SF-selected).
    std::vector<double> energy_ev;

    /// @brief UCCSD correlation energy E_CCSD − E_HF [Hartree].
    double e_corr_ha = 0.0;
    /// @brief UCCSD total energy E_HF + E_corr [Hartree].
    double e_ccsd_ha = 0.0;
    /// @brief Jacobi micro-iterations used to converge the UCCSD amplitudes.
    int ccsd_iterations = 0;
    /// @brief True when UCCSD amplitudes converged and SF roots were obtained.
    bool converged = false;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
