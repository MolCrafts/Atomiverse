/// @file cpu/dynamics/fix_efield.h
/// @brief CPU uniform external electric field — LAMMPS-equivalent `fix efield`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/curvature_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Apply a uniform electric field to charged atoms (any `units` style).
///
/// Mirrors LAMMPS `fix efield Ex Ey Ez` for the **constant** field path
/// (`lammps/src/fix_efield.cpp`, charge interactions only):
///
///   \[
///     \mathbf{F}_i \mathrel{+}= q_i \mathbf{E}\,\texttt{qe2f},\qquad
///     U \mathrel{+}= -q_i\,\texttt{qe2f}\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}
///   \]
///
/// with \(\mathbf{E}\) in the run's LAMMPS field unit (V/Å for `real` and
/// `metal`, V/m for `si`, statvolt/cm for `cgs`, V/cm for `electron`, V/µm
/// for `micro`, V/nm for `nano`, reduced for `lj`), \(q\) in the style's
/// charge unit, and `qe2f` the run's `ctx.runtime.units.qe2f` (LAMMPS
/// `force->qe2f`: 23.060549 for real, 1 for metal), so \(\mathbf F\) is in
/// the style's force unit and \(U\) in its energy unit. Unwrapped
/// coordinates use the image flags on Frame,
/// \(\mathbf{r}^{\mathrm{u}} = \mathbf{r} + H\mathbf{n}\) with the lattice
/// vectors in the columns of \(H\) (LAMMPS `Domain::unmap`), so orthorhombic
/// and triclinic cells are both supported.
///
/// Under PBC the energy is the LAMMPS `fsum[0]` bookkeeping, not a bounded
/// potential: it is continuous only while the image flags track every
/// boundary crossing, and it drifts linearly as ions migrate along the field
/// (that drift is the work the field does). It enters `kEnergy` like a
/// `fix_modify energy yes` fix. **No virial** is tallied — LAMMPS adds
/// \(F\otimes\mathbf r^{\mathrm u}\) only under `fix_modify virial yes`,
/// which is off by default, so pressure / stress exclude the field.
///
/// `default_bindings()` is `md::Phase::post_force` (LAMMPS POST_FORCE mask)
/// for `bind_defaults`; the categorical `set_evaluator` places it in the force
/// phase, which gives the same total force. For single-point Evaluate,
/// register explicitly:
/// `eval.on(evaluate::Phase::evaluate).add(FixEfield{Ex,Ey,Ez})`.
///
/// An `ExternalDrive`: the field is applied work, not part of the interaction
/// potential, so the configurational temperature excludes its force.
///
/// CUDA twin: `cuda::md::FixEfield` (same law, same energy convention).
///
/// ── Not in scope (see spec `external-efield-md`) ──────────────────────
/// variable (`v_`) fields, region keyword, dipole torque, energy keyword
/// for minimization-only variable energy.
class FixEfield : public cpu::Bias, public cpu::ExternalDrive {
  public:
    /// @param Ex Electric field x-component [style field unit; V/Å for
    ///           real and metal].
    /// @param Ey Electric field y-component [style field unit].
    /// @param Ez Electric field z-component [style field unit].
    FixEfield(double Ex, double Ey, double Ez) noexcept
        : Ex(Ex), Ey(Ey), Ez(Ez) {}

    /// Field components [style field unit] — public members (no Config bag).
    double Ex = 0.0;
    double Ey = 0.0;
    double Ez = 0.0;

    std::string name() const override { return "fix_efield"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After physical forces; before final velocity half-kick.
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }

    const ExternalDrive* as_external_drive() const override { return this; }
    /// @brief F −= q E qe2f on owned rows (the force `launch` adds).
    void subtract_drive_force(const LaunchContext& ctx, int n, double* fx,
                              double* fy, double* fz) const override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @return Last field energy contribution written this launch [style
    ///         energy unit].
    double last_energy() const noexcept { return last_u_; }

  private:
    double last_u_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
