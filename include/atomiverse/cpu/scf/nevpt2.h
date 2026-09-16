/// @file cpu/scf/nevpt2.h
/// @brief Strongly contracted NEVPT2 — public API via `RHF::nevpt2`.
///
/// Closed-shell SC-NEVPT2 on a CASSCF reference (Angeli, Cimiraglia, Malrieu,
/// J. Chem. Phys. **114**, 10252 (2001); **117**, 9138 (2002)). Matches PySCF
/// `mrpt.NEVPT(mc).kernel()` for the same closed-shell foothold.
///
/// On H₂/STO-3G with full-valence CASSCF(2,2) there is no inactive core and no
/// external virtual space (ncore = nvirt = 0). Every SC perturber class involves
/// at least one core or virtual index, so the PT2 correction is **identically
/// zero** and E_NEVPT2 = E_CASSCF = E_FCI in this basis.
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell strongly contracted NEVPT2 on a CASSCF reference.
struct NEVPT2Result {
    /// @brief Total NEVPT2 energy E_CASSCF + E_corr [Hartree].
    double e_nevpt2 = 0.0;
    /// @brief SC-NEVPT2 correlation correction vs CASSCF [Hartree]
    ///        (PySCF `mrpt.NEVPT.e_corr`; ≤ 0 when the external space is
    ///        non-empty and recovers dynamic correlation).
    double e_corr = 0.0;
    /// @brief CASSCF total energy [Hartree] used as the zeroth-order reference.
    double e_casscf = 0.0;
    /// @brief Number of active spatial orbitals.
    int ncas = 0;
    /// @brief Number of active electrons (closed-shell: even).
    int nelecas = 0;
    /// @brief Number of frozen doubly-occupied core spatial orbitals.
    int ncore = 0;
    /// @brief Number of external (virtual) spatial orbitals outside the CAS.
    int nvirt = 0;
    /// @brief True when the underlying CASSCF orbital optimization converged.
    bool converged = false;
};

// Pack-based entry is private — see internal/nevpt2_pack.h.
// Public: MeanFieldMethod::nevpt2(frame, ncas, nelecas) (closed-shell RHF).

} // namespace scf
} // namespace cpu
} // namespace ATV
