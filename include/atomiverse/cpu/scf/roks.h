/// @file cpu/scf/roks.h
/// @brief Restricted open-shell Kohn–Sham (ROKS) concrete mean-field method —
///        Roothaan effective-Fock coupling on an unrestricted KS Fock pair
///        (PySCF canonical / Guest–Saunders convention).
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted open-shell Kohn–Sham — the spin-pure open-shell
///        density-functional method: ONE set of spatial orbitals (closed /
///        open / virtual three-space) with `n_alpha ≥ n_beta` split
///        occupation, so ⟨S²⟩ equals S(S+1) exactly (no UKS spin
///        contamination). The Fock pair is the unrestricted KS build
///        (J + Vxc_σ ± hybrid exchange) coupled into one Roothaan
///        effective Fock by `RohfCoupleStage` before a single
///        restricted-style diagonalization (C_β = C_α).
///
/// Energy is the coupling-free unrestricted KS functional evaluated on the
/// split-occupation density (E_one + E_J + E_x + E_xc + V_nn) [Hartree] —
/// same functional as UKS at the constrained RO density.
///
/// **Convention (binding):** F_eff uses the same PySCF
/// `get_roothaan_fock` projector assembly as ROHF (Guest–Saunders). Total
/// energy is portable; orbital energies are convention-bound.
///
/// Analytic nuclear forces are supported (`forces()`): ROHF-style
/// energy-weighted density W = O_a F_a O_a + O_b F_b O_b on the KS spin
/// Focks, plus the unrestricted KS J/K and XC grid gradient path (PySCF
/// `dft.ROKS.nuc_grad_method`). Not yet supported (throws, no fallback):
/// `mp2()`, `fci()`, `soscf`, `level_shift`, density fitting.
///
/// References: Roothaan, Rev. Mod. Phys. 32, 179 (1960); Guest & Saunders,
/// Mol. Phys. 28, 819 (1974); Tsuchimochi & Scuseria, J. Chem. Phys. 133,
/// 141102 (2010); PySCF `dft.ROKS`.
class ROKS : public MeanFieldMethod<ROKS> {
  public:
    const int n_alpha; ///< Alpha (spin-up) electrons.
    const int
        n_beta;      ///< Beta (spin-down) electrons; requires n_alpha ≥ n_beta.
    const XCType xc; ///< Exchange–correlation functional.

    /// @brief Construct ROKS (MINAO guess, MatchPySCF grid, 0.5 damping,
    ///        DIIS 6 vectors from iteration 1, energy tolerance 1e-8 Ha).
    /// @param basis Basis set.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons (must satisfy n_alpha ≥ n_beta;
    ///                violated at run() with std::invalid_argument).
    /// @param xc      Exchange–correlation functional.
    /// @param charge  Molecular charge [e].
    ROKS(BasisSet basis, int n_alpha, int n_beta, XCType xc = XCType::LSDA,
         int charge = 0);
    std::string name() const { return "roks"; }

  protected:
    /// @brief Occupation is taken from the `n_alpha` / `n_beta` members;
    ///        also latches the selected `xc` functional. The electron-count
    ///        argument is unused for the open-shell case.
    /// @throws std::invalid_argument when n_alpha < n_beta (ROKS defines the
    ///         open shell as excess α; no fallback).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
