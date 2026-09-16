/// @file cpu/scf/casscf.h
/// @brief Complete active-space SCF (CASSCF) — public API via `RHF::casscf`.
///
/// Closed-shell CASSCF on a converged RHF reference: frozen core is doubly
/// occupied; full CI is solved in the active spatial orbitals; orbitals are
/// optimized (Newton on the orbital gradient from spin-summed RDMs) until the
/// generalized Fock is symmetric to tolerance. Matches PySCF
/// `mcscf.CASSCF(mf, ncas, nelecas)` for the same closed-shell foothold.
///
/// On H₂/STO-3G with full-valence CAS(2,2) there are no non-redundant orbital
/// rotations (ncore=0, nvirt=0), so CASSCF energy equals CASCI / FCI.
///
/// Roos, *The Complete Active Space Self-Consistent Field Method and its
/// Applications in Electronic Structure Calculations*, Adv. Chem. Phys.
/// **69**, 399 (1987).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Closed-shell CASSCF ground-state energy on an RHF reference.
struct CASSCFResult {
    /// @brief Total CASSCF energy E_CASSCF [Hartree] (nuclear + core + active).
    double e_casscf = 0.0;
    /// @brief Correlation energy e_casscf − E_SCF [Hartree].
    double e_corr = 0.0;
    /// @brief Number of active spatial orbitals used in the CI.
    int ncas = 0;
    /// @brief Number of active electrons (closed-shell: even).
    int nelecas = 0;
    /// @brief Number of frozen doubly-occupied core spatial orbitals.
    int ncore = 0;
    /// @brief Macro-iterations of (CI + orbital step) until orbital gradient
    ///        convergence (dimensionless). Zero non-redundant rotations count
    ///        as one CI-only pass.
    int iterations = 0;
    /// @brief True when the orbital gradient fell below the method tolerance.
    bool converged = false;
};

// Pack-based entry is private — see internal/casscf_pack.h.
// Public: MeanFieldMethod::casscf(frame, ncas, nelecas) (closed-shell RHF).

} // namespace scf
} // namespace cpu
} // namespace ATV
