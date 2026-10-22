/// @file cpu/scf/uhf.h
/// @brief Unrestricted Hartree–Fock (UHF) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/delta_scf.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/scf_guess.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Unrestricted Hartree–Fock — the same mean-field, single-determinant
///        method as RHF but with independent spatial orbitals for the spin-up
///        (α) and spin-down (β) electrons, so open-shell / spin-polarized
///        systems (unequal α and β counts) are representable.
class UHF : public MeanFieldMethod<UHF> {
  public:
    const int n_alpha; ///< Alpha (spin-up) electrons.
    const int n_beta;  ///< Beta (spin-down) electrons.

    /// @brief Basis, α/β counts and molecular charge (Hcore guess, 0.7
    ///        density damping, DIIS 6 vectors from iteration 5).
    ///
    /// CUDA twin: `cuda::scf::UHF{basis, n_alpha, n_beta, charge}`.
    /// `n_alpha + n_beta` must equal ΣZ − charge; the integral build checks it.
    /// Post-HF, solvent, and the analytic Hessian stay off this class.
    /// `delta_scf` is the MOM/IMOM loop on the CPU type only.
    ///
    /// @param basis_in Basis set (e.g. `BasisSet::STO3G`).
    /// @param n_alpha_in Spin-up electrons.
    /// @param n_beta_in Spin-down electrons.
    /// @param charge_in Molecular charge [e].
    /// @param tol_in Energy tolerance [Ha].
    /// @param max_iter_in Maximum SCF iterations [dimensionless].
    UHF(BasisSet basis_in, int n_alpha_in, int n_beta_in, int charge_in = 0,
        double tol_in = 1e-8, int max_iter_in = 100);

    std::string name() const { return "uhf"; }

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
