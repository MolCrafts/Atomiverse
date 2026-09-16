/// @file cpu/dynamics/fix_addforce.h
/// @brief CPU constant external force — LAMMPS-equivalent `fix addforce`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Add a constant force to atoms of one type (metal units).
///
/// Mirrors LAMMPS `fix addforce fx fy fz` for the **constant** force path
/// (`lammps/src/fix_addforce.cpp`, group = one atom type):
///
///   \[
///     \mathbf{F}_i \mathrel{+}= \mathbf{F}^{\mathrm{add}},\qquad
///     U \mathrel{+}= -\mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}_i^{\mathrm{u}}
///   \]
///
/// with \(\mathbf{F}^{\mathrm{add}}\) in eV/Å, \(\mathbf{F}\) in eV/Å,
/// \(U\) in eV. Unwrapped coordinates use orthorhombic image flags
/// (\(\mathbf{r}^{\mathrm{u}} = \mathbf{r} + \mathrm{image}\cdot\mathbf{L}\));
/// triclinic unwrap is not supported in this foothold (throws if non-ortho).
///
/// Bound by default at `md::Phase::post_force` (LAMMPS POST_FORCE mask) so
/// pair / bonded / kspace forces accumulate first. For single-point
/// Evaluate, register explicitly:
/// `eval.on(evaluate::Phase::evaluate).add(FixAddForce{Fx,Fy,Fz,type})`.
///
/// ── Not in scope (see spec `constant-force-pull-ion`) ────────────────
/// variable (`v_`) forces, region keyword, `energy v_…`, RESPA level,
/// LAMMPS `fix smd` spring/CV path, virial tally for the added force.
class FixAddForce : public cpu::Bias {
  public:
    /// @param Fx Force x-component [eV/Å] (LAMMPS metal units).
    /// @param Fy Force y-component [eV/Å].
    /// @param Fz Force z-component [eV/Å].
    /// @param type_id Atom type to accelerate (matches `kAtom_type`).
    FixAddForce(double Fx, double Fy, double Fz, int type_id) noexcept
        : Fx(Fx), Fy(Fy), Fz(Fz), type_id(type_id) {}

    /// Force components [eV/Å] — public members (no Config bag).
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;
    /// Selected atom type id (matches `frame.span(kAtom_type)`).
    int type_id = 0;

    std::string name() const override { return "fix_addforce"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After physical forces; before final velocity half-kick.
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @return Last bias energy contribution written this launch [eV].
    double last_energy() const noexcept { return last_u_; }

  private:
    double last_u_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
