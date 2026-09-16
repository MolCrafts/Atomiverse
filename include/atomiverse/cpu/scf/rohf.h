/// @file cpu/scf/rohf.h
/// @brief Restricted open-shell Hartree–Fock (ROHF) concrete mean-field
///        method — Roothaan effective Fock, PySCF canonical (Guest–Saunders)
///        convention.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted open-shell Hartree–Fock — the spin-pure open-shell
///        mean-field method: ONE set of spatial orbitals (closed / open /
///        virtual three-space) with `n_alpha ≥ n_beta` split occupation, so
///        ⟨S²⟩ equals S(S+1) exactly (no UHF spin contamination).
///
/// Mechanics: the unrestricted Pople–Nesbet Fock build supplies F_α/F_β,
/// which the `RohfCoupleStage` couples into one Roothaan effective Fock
/// F_eff before a restricted-style single diagonalization (C_β = C_α);
/// densities are rebuilt by split occupation (P_α from the first n_α
/// columns, P_β from the first n_β). The energy is the coupling-free
/// Pople–Nesbet functional E = ½Tr[P_α(H+F_α)] + ½Tr[P_β(H+F_β)] + V_nn
/// [Hartree], evaluated on the TRUE spin Focks.
///
/// **Convention (binding):** F_eff uses the PySCF canonical-ROHF projector
/// assembly (`get_roothaan_fock`, Guest–Saunders coupling: diagonal blocks
/// cc = oo = vv = F_c ≡ (F_α+F_β)/2, off-diagonal co = F_β, ov = F_α,
/// cv = F_c). **PITFALL:** the Roothaan / Guest–Saunders /
/// McWeeny–Diercksen coupling-coefficient families give the SAME total
/// energy but DIFFERENT orbital energies — ε is a convention-bound
/// quantity. Orbital energies from this class are only comparable to
/// references using the same PySCF canonical convention.
///
/// Not yet supported for ROHF (throws, no fallback): `mp2()`, `fci()`,
/// `soscf`, and `level_shift`. Analytic nuclear forces and the analytic
/// nuclear Hessian (`forces`, `hessian`) are implemented.
///
/// References: Roothaan, Rev. Mod. Phys. 32, 179 (1960),
/// DOI 10.1103/RevModPhys.32.179; Guest & Saunders, Mol. Phys. 28, 819
/// (1974); Tsuchimochi & Scuseria, J. Chem. Phys. 133, 141102 (2010),
/// DOI 10.1063/1.3503173.
class ROHF : public MeanFieldMethod<ROHF> {
  public:
    int n_alpha = 0; ///< Alpha (spin-up) electrons (must be set before run).
    int n_beta = 0;  ///< Beta (spin-down) electrons; requires n_alpha ≥ n_beta.

    /// @brief Construct with the ROHF defaults (restricted-open HF; Hcore
    ///        initial guess; 0.7 density damping; DIIS capped at 6 vectors
    ///        and started at iteration 5; energy tolerance 1e-8 Ha). Set
    ///        `n_alpha` / `n_beta` before `run`.
    ROHF();
    /// @brief Construct with an explicit α / β electron count.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons (must satisfy n_alpha ≥ n_beta;
    ///                violated at run() with std::invalid_argument).
    explicit ROHF(int n_alpha, int n_beta);
    std::string name() const { return "rohf"; }

    /// @brief Nuclear Hessian d²E/dR² of the ROHF energy [eV/Å²].
    ///
    /// Central finite difference of the **analytic** ROHF nuclear forces
    /// (`forces()` / ROHF Lagrangian W = O_α F_α O_α + O_β F_β O_β). This
    /// is the correct second derivative of the ROHF energy under the
    /// constrained open-shell SCF; a fully analytic three-space ROHF CPHF
    /// is a follow-up (PySCF 2.14 also lacks `hessian.ROHF`). Step
    /// \(h = 10^{-4}\) Å; the assembled matrix is symmetrized.
    ///
    /// @param frame Geometry (Å) + Z in the type column; restored on return.
    /// @return `vib::HessianResult` with the dense symmetric `[3N, 3N]`
    ///         Hessian in eV/Å² and `n_atoms` set.
    /// @throws std::runtime_error when the reference or displaced SCF
    ///         does not converge, or with RI enabled.
    vib::HessianResult hessian(Frame& frame);

  protected:
    /// @brief Occupation is taken directly from the `n_alpha` / `n_beta`
    ///        members; the electron-count argument is unused for the
    ///        open-shell case.
    /// @throws std::invalid_argument when n_alpha < n_beta (ROHF defines the
    ///         open shell as excess α; no fallback).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
