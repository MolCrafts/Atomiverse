/// @file cpu/scf/uhf.h
/// @brief Unrestricted Hartree–Fock (UHF) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/delta_scf.h"
#include "atomiverse/cpu/scf/eom_sf_ccsd.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/scf_guess.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Unrestricted Hartree–Fock — the same mean-field, single-determinant
///        method as RHF but with independent spatial orbitals for the spin-up
///        (α) and spin-down (β) electrons, so open-shell / spin-polarized
///        systems (unequal α and β counts) are representable.
class UHF : public MeanFieldMethod<UHF> {
  public:
    int n_alpha = 0; ///< Alpha (spin-up) electrons (must be set before run).
    int n_beta = 0;  ///< Beta (spin-down) electrons (must be set before run).

    /// @brief Construct with the UHF defaults (unrestricted HF; Hcore initial
    ///        guess; 0.7 density damping; DIIS capped at 6 vectors and started
    ///        at iteration 5; energy tolerance 1e-8 Ha). Set `n_alpha` /
    ///        `n_beta` before `run`.
    UHF();
    /// @brief Construct with an explicit α / β electron count.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons.
    explicit UHF(int n_alpha, int n_beta);
    std::string name() const { return "uhf"; }

    /// @brief Analytic nuclear Hessian d²E/dR² via unrestricted CPHF
    ///        linear response — a method output beside `forces()`, feeding
    ///        the vib pipeline (`vib::MassWeighting` →
    ///        `vib::HessianDiagonalizer`).
    ///
    /// Runs a fresh UHF SCF on `frame` (writes the eV energy to
    /// `frame.meta[kEnergy]`), assembles the skeleton second-derivative
    /// integrals with per-spin densities, solves the coupled α/β CPHF
    /// equations (A+B)·U^x = −B^x for all 3N nuclear perturbations, and
    /// assembles E^{xy} (Pople / Yamaguchi unrestricted second derivatives).
    ///
    /// @param frame Geometry (Å) + Z in the type column; atoms not moved.
    /// @param guess Optional density-matrix guess for the reference SCF
    ///              (needed for multi-minima cases such as triplet O₂ —
    ///              ROHF warm-start).
    /// @return `vib::HessianResult` with the dense symmetric `[3N, 3N]`
    ///         Hessian in eV/Å² and `n_atoms` set; frequency /
    ///         thermochemistry fields stay empty for the vib components.
    /// @throws std::runtime_error when the SCF does not converge, on a
    ///         spherical-convention basis or ECP pack, with RI enabled, or
    ///         on CPHF non-convergence.
    /// @throws std::invalid_argument when either spin channel has no
    ///         virtual orbitals.
    vib::HessianResult hessian(Frame& frame,
                               const SCFInitialGuess* guess = nullptr);

    /// @brief Collinear spin-flip Tamm–Dancoff (SF-CIS) on the most recent
    ///        `run()`. High-spin UHF reference with Ms-decreasing singles
    ///        \(i_\alpha \to a_\beta\) (when \(n_\alpha \ge n_\beta\); else
    ///        \(i_\beta \to a_\alpha\)).
    ///
    /// Dense SF-CIS / SF-TDA Hamiltonian (Shao, Head-Gordon & Krylov,
    /// J. Chem. Phys. 118, 4807 (2003); collinear HF kernel — Coulomb vanishes
    /// for pure spin-flip, exchange only):
    ///
    ///   \(A_{ia,jb}=\delta_{ij}\delta_{ab}(\varepsilon_a^\beta-\varepsilon_i^\alpha)
    ///   -(ij|ab)\)
    ///
    /// with \(i,j\) occupied of the majority spin and \(a,b\) virtual of the
    /// minority spin. Dense `eigh` of **A** (`n_ov ≤ 512`). Oscillator
    /// strengths are zero (spin-forbidden from the high-spin reference in the
    /// collinear picture).
    ///
    /// @param n_states Number of lowest SF roots (1 ≤ n_states ≤ n_ov).
    /// @return Excitation energies [eV]; `tda = true`; `alda = false`.
    /// @throws std::runtime_error when called before `run()`, on a restricted
    ///         determinant, empty SF manifold, or n_ov > 512 (no fallback).
    /// @throws std::invalid_argument when n_states is outside [1, n_ov].
    ExcitedStatesResult sf_tda(int n_states) const;

    /// @brief Collinear spin-flip full linear response (SF-TDHF / SF-RPA) on
    ///        the most recent `run()`. Same Ms-decreasing singles space as
    ///        `sf_tda` (majority occupied → minority virtual).
    ///
    /// Collinear HF kernel (Shao, Head-Gordon & Krylov, J. Chem. Phys. 118,
    /// 4807 (2003); Coulomb vanishes for pure spin-flip):
    ///
    ///   \(A_{ia,jb}=\delta_{ij}\delta_{ab}(\varepsilon_a^{\mathrm{min}}
    ///   -\varepsilon_i^{\mathrm{maj}})-(ij|ab)\)
    ///
    ///   \(B_{ia,jb}=-(ib|aj)\)
    ///
    /// Solves the non-Hermitian RPA eigenproblem
    /// \(\begin{bmatrix}A&B\\-B&-A\end{bmatrix}\begin{bmatrix}X\\Y\end{bmatrix}
    /// =\omega\begin{bmatrix}X\\Y\end{bmatrix}\) via dense LAPACK `dgeev`
    /// (`n_ov ≤ 256` so the doubled space is ≤ 512). Returns the lowest
    /// **positive real** eigenvalues as excitation energies [eV]. Oscillator
    /// strengths are zero (spin-forbidden in the collinear picture). When
    /// \((A-B)\) is indefinite the Hermitian Casida reduction is undefined;
    /// the non-Hermitian form remains well-posed and is the product path.
    ///
    /// @param n_states Number of lowest positive SF-RPA roots
    ///                 (1 ≤ n_states ≤ n_ov).
    /// @return Excitation energies [eV]; `tda = false`; `alda = false`.
    /// @throws std::runtime_error when called before `run()`, on a restricted
    ///         determinant, empty SF manifold, n_ov > 256, `dgeev` failure, or
    ///         fewer than `n_states` positive real roots (no fallback).
    /// @throws std::invalid_argument when n_states is outside [1, n_ov].
    ExcitedStatesResult sf_tddft(int n_states) const;

    /// @brief EOM-SF-CCSD spin-flip excitation energies on a UHF → UCCSD
    ///        reference (drives SCF + UCCSD + SF-EOM).
    ///
    /// High-spin UHF (set `n_alpha` / `n_beta`), unrestricted spin-orbital
    /// CCSD, then equation-of-motion spin-flip singles+doubles. Matches
    /// PySCF `cc.UCCSD(mf).eomsf_ccsd(nroots=n_states)` on the same
    /// geometry / basis (cart). Product surface: ω in eV.
    ///
    /// @param frame    Geometry [Å] + Z; energy meta written from UHF.
    /// @param n_states Number of lowest SF roots (dimensionless, ≥ 1).
    /// @param guess    Optional density guess for the reference UHF (ROHF
    ///                 warm-start for multi-minima cases such as triplet O₂).
    /// @return SF excitation energies [eV] plus UCCSD energetics [Ha].
    /// @throws std::invalid_argument when SCF does not converge, n_states
    ///         is invalid, or the SF manifold is empty (no fallback).
    EOMSFCCSDResult eom_sf_ccsd(Frame& frame, int n_states = 1,
                                const SCFInitialGuess* guess = nullptr);

    /// @brief ΔSCF vertical excitation with MOM/IMOM occupation locking.
    ///
    /// Runs a ground-state UHF on `frame`, promotes one α electron from MO
    /// index `from` to virtual `to` via
    /// `SlaterDeterminant::with_occupation`, seeds the non-Aufbau density,
    /// and re-SCFs with `mom = style` (default `IMOM` = PySCF
    /// `scf.addons.mom_occ_`). Frame energy meta is left at the **excited**
    /// state; ground/excited Hartree energies and ΔE [eV] are on the return.
    ///
    /// @param frame Geometry [Å]; mutated only for energy meta on the
    ///              excited-state run.
    /// @param from  Occupied α MO index to depopulate (0-based; f=1).
    /// @param to    Virtual α MO index to populate (0-based; f=0).
    /// @param style `MomStyle::IMOM` (default) or `MomStyle::MOM`.
    /// @param guess Optional density guess for the ground-state UHF only.
    /// @return `DeltaScfResult` with e_ground / e_excited [Ha], delta_e_ev.
    /// @throws std::invalid_argument on bad indices or style==Off.
    /// @throws std::runtime_error when either SCF does not converge.
    DeltaScfResult delta_scf(Frame& frame, int from, int to,
                             MomStyle style = MomStyle::IMOM,
                             const SCFInitialGuess* guess = nullptr);

  protected:
    /// @brief Occupation is taken directly from the `n_alpha` / `n_beta`
    ///        members; the electron-count argument is unused for the open-shell
    ///        case.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
