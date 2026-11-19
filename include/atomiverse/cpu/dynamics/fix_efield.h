/// @file cpu/dynamics/fix_efield.h
/// @brief CPU uniform external electric field — LAMMPS-equivalent `fix efield`.
#pragma once
#include "atomiverse/core/unit_system.h"

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Apply a uniform electric field to charged atoms (run units).
///
/// Mirrors LAMMPS `fix efield Ex Ey Ez` for the **constant** field path
/// (`lammps/src/fix_efield.cpp`, charge interactions only):
///
///   \[
///     \mathbf{F}_i \mathrel{+}= q_i \mathbf{E},\qquad
///     U \mathrel{+}= -q_i\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}
///   \]
///
/// Field, charge, force and energy follow the run's unit profile.
/// The unit module prepares field amplitudes in force/charge units.
/// Unwrapped coordinates use the image flags on Frame,
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
/// CUDA twin: `cuda::md::FixEfield` (same law, same energy convention).
///
/// ── Not in scope (see spec `external-efield-md`) ──────────────────────
/// variable (`v_`) fields, region keyword, dipole torque, energy keyword
/// for minimization-only variable energy.
class FixEfield : public cpu::Bias {
  public:
    /// @param Ex Electric field x-component [run electric field].
    /// @param Ey Electric field y-component [run electric field].
    /// @param Ez Electric field z-component [run electric field].
    FixEfield(double Ex, double Ey, double Ez) noexcept
        : Ex(Ex), Ey(Ey), Ez(Ez) {}

    /// Field components [run electric field] — public members (no Config bag).
    double Ex = 0.0;
    double Ey = 0.0;
    double Ez = 0.0;

    std::string name() const override { return "fix_efield"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After physical forces; before final velocity half-kick.
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void build(const BuildContext& ctx) override {
        field_units_ = ctx.runtime.units.electric_field;
    }
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }
    void launch(const cpu::LaunchContext& ctx) override;

    /// @return Last field energy contribution written this launch [run energy].
    double last_energy() const noexcept { return last_u_; }

  private:
    units::Conversion field_units_;
    double last_u_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
