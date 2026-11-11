/// @file cpu/scf/rohf.h
/// @brief Restricted open-shell Hartree–Fock (ROHF) concrete mean-field
///        method — Roothaan effective Fock, PySCF canonical (Guest–Saunders)
///        convention.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

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
/// Not yet supported for ROHF (throws, no fallback): `soscf` and
/// `level_shift`. Analytic nuclear forces (`forces`) are on this method.
/// The nuclear Hessian is `Hessian<ROHF>` (finite difference of those
/// forces), not a member — PySCF has no `hessian.ROHF`.
///
/// References: Roothaan, Rev. Mod. Phys. 32, 179 (1960),
/// DOI 10.1103/RevModPhys.32.179; Guest & Saunders, Mol. Phys. 28, 819
/// (1974); Tsuchimochi & Scuseria, J. Chem. Phys. 133, 141102 (2010),
/// DOI 10.1063/1.3503173.
class ROHF : public MeanFieldMethod<ROHF> {
  public:
    const int n_alpha; ///< Alpha (spin-up) electrons.
    const int
        n_beta; ///< Beta (spin-down) electrons; requires n_alpha ≥ n_beta.

    /// @brief Construct ROHF (Hcore guess, 0.7 damping, DIIS 6 vectors from
    ///        iteration 5, energy tolerance 1e-8 Ha).
    /// @param basis Basis set.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons (must satisfy n_alpha ≥ n_beta;
    ///                violated at run() with std::invalid_argument).
    /// @param charge  Molecular charge [e].
    ROHF(BasisSet basis, int n_alpha, int n_beta, int charge = 0);
    std::string name() const { return "rohf"; }

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
