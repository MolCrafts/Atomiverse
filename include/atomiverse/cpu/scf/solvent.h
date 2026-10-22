/// @file cpu/scf/solvent.h
/// @brief Continuum solvent attached beside a mean-field method.
///
/// Transplanted from `pyscf.solvent._attach_solvent`:
/// `SCFWithSolvent(mf, solvent)` copies the SCF object and stores the solvent
/// model on `WithSolvent`. The reaction field enters `get_veff` / `energy_elec`
/// of the wrapper, not fields of the base `RHF` class (`scf.hf.RHF` has no
/// `eps`). `undo_solvent` is unnecessary here: the unwrapped method is `mf`
/// before it is moved into `WithSolvent`.
///
/// `pyscf.df.df_jk.density_fit(mf)` is the same kind of wrapper for the JK
/// build. Density fitting stays an SCF Fock selector (`MeanFieldMethod::ri`),
/// not a post-HF type.
#pragma once

#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Mean-field method plus a continuum solvent.
///
/// Owns the method. `run` / `forces` bind the solvent onto the private SCF
/// reaction-field slot, then call the method. Gas phase is an unwrapped
/// method, not `eps = 0` on `MeanFieldMethod`.
/// @tparam Method Concrete mean-field type (`RHF`, `UHF`, `RKS`, `UKS`, …).
template <typename Method>
class WithSolvent {
  public:
    /// @brief The mean-field method. Configure basis, charge, and tolerance
    ///        here. Do not call `mf.run` if the solvent must be active —
    ///        call `WithSolvent::run`.
    Method mf;

    /// @brief C-PCM, COSMO (via `x`) or IEF-PCM.
    const SolventModel model;
    /// @brief Dielectric constant ε (dimensionless); 0 is vacuum.
    const double eps;
    /// @brief Multiplier on Bondi vdW radii (dimensionless).
    double radii_scale = 1.2;
    /// @brief Conductor-like scaling x in f(ε)=(ε−1)/(ε+x): 0 = C-PCM,
    ///        0.5 = COSMO. Ignored for IEF-PCM.
    double x = 0.0;

    /// @brief Take ownership of a method and attach a continuum solvent
    ///        (PySCF `SCFWithSolvent`).
    /// @param method Mean-field method (moved).
    /// @param model Solvent model.
    /// @param eps Dielectric constant ε (dimensionless); `eps > 0` turns the
    ///        reaction field on.
    WithSolvent(Method method, SolventModel model, double eps)
        : mf(std::move(method)), model(model), eps(eps) {
        bind_();
    }

    /// @brief SCF with the reaction field in the Fock build.
    /// @param frame Geometry [Å]. Energy [eV] written to `kEnergy`.
    /// @param guess Optional density guess.
    void run(Frame& frame, const SCFInitialGuess* guess = nullptr) {
        bind_();
        mf.run(frame, guess);
    }

    /// @brief Const-geometry SCF. Does not write Frame meta.
    /// @param frame Geometry [Å], left untouched.
    /// @param guess Optional density guess.
    void run(const Frame& frame, const SCFInitialGuess* guess = nullptr) {
        bind_();
        mf.run(frame, guess);
    }

    /// @brief Analytic free-energy forces [eV/Å], including ΔG_el.
    /// @param frame Geometry. Forces and energy written on the frame.
    /// @param guess Optional density guess.
    /// @return Forces and total free energy.
    GradientResult forces(Frame& frame,
                          const SCFInitialGuess* guess = nullptr) {
        bind_();
        return mf.forces(frame, guess);
    }

    /// @brief Electrostatic solvation free energy of the most recent run.
    /// @return ΔG_el [Hartree]. 0 when `eps == 0`.
    double e_solvent() const { return mf.e_solvent_; }

  private:
    void bind_() {
        mf.solvent_eps_ = eps;
        mf.solvent_radii_scale_ = radii_scale;
        mf.solvent_x_ = x;
        mf.solvent_model_ = model;
    }
};

} // namespace scf
} // namespace cpu
} // namespace ATV
