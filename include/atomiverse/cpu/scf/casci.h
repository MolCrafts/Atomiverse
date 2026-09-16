/// @file cpu/scf/casci.h
/// @brief Complete active-space CI (CASCI) — public API via `RHF::casci`.
///
/// Closed-shell CASCI on a converged RHF reference: frozen core is doubly
/// occupied; full CI is solved in the active spatial orbitals only (dense
/// determinant basis, same Slater–Condon kernel as `RHF::fci`). Matches
/// PySCF `mcscf.CASCI(mf, ncas, nelecas)` for the same RHF orbitals.
///
/// Roos, *The Complete Active Space Self-Consistent Field Method and its
/// Applications in Electronic Structure Calculations*, Adv. Chem. Phys.
/// **69**, 399 (1987).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CASCI ground-state energy on an RHF reference.
struct CASCIResult {
    /// @brief Total CASCI energy E_CASCI [Hartree] (nuclear + core + active).
    double e_casci = 0.0;
    /// @brief Correlation energy e_casci − E_SCF [Hartree] (≤ 0 when CAS
    ///        recovers correlation).
    double e_corr = 0.0;
    /// @brief Number of active spatial orbitals used in the CI.
    int ncas = 0;
    /// @brief Number of active electrons (closed-shell: even).
    int nelecas = 0;
    /// @brief Number of frozen doubly-occupied core spatial orbitals.
    int ncore = 0;
};

// Pack-based entry is private — see internal/casci_pack.h.
// Public: MeanFieldMethod::casci(frame, ncas, nelecas) (closed-shell RHF).

} // namespace scf
} // namespace cpu
} // namespace ATV
