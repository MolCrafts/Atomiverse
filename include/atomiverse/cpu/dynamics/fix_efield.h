/// @file cpu/dynamics/fix_efield.h
/// @brief CPU uniform external electric field — LAMMPS-equivalent `fix efield`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Apply a uniform electric field to charged atoms (metal units).
///
/// Mirrors LAMMPS `fix efield Ex Ey Ez` for the **constant** field path
/// (`lammps/src/fix_efield.cpp`, charge interactions only):
///
///   \[
///     \mathbf{F}_i \mathrel{+}= q_i \mathbf{E},\qquad
///     U \mathrel{+}= -q_i\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}
///   \]
///
/// with \(\mathbf{E}\) in V/Å, \(q\) in elementary charge, \(\mathbf{F}\) in
/// eV/Å, \(U\) in eV. Metal-unit `qe2f = 1` (no extra scale). Unwrapped
/// coordinates use the orthorhombic image flags on Frame
/// (\(\mathbf{r}^{\mathrm{u}} = \mathbf{r} + \mathrm{image}\cdot\mathbf{L}\));
/// triclinic unwrap is not supported in this foothold (throws if non-ortho).
///
/// Bound by default at `md::Phase::post_force` (LAMMPS POST_FORCE mask) so
/// pair / bonded / kspace forces accumulate first. For single-point
/// Evaluate, register explicitly:
/// `eval.on(evaluate::Phase::evaluate).add(FixEfield{Ex,Ey,Ez})`.
///
/// ── Not in scope (see spec `external-efield-md`) ──────────────────────
/// variable (`v_`) fields, region keyword, dipole torque, energy keyword
/// for minimization-only variable energy.
class FixEfield : public cpu::Bias {
  public:
    /// @param Ex Electric field x-component [V/Å] (LAMMPS metal units).
    /// @param Ey Electric field y-component [V/Å].
    /// @param Ez Electric field z-component [V/Å].
    FixEfield(double Ex, double Ey, double Ez) noexcept
        : Ex(Ex), Ey(Ey), Ez(Ez) {}

    /// Field components [V/Å] — public members (no Config bag).
    double Ex = 0.0;
    double Ey = 0.0;
    double Ez = 0.0;

    std::string name() const override { return "fix_efield"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After physical forces; before final velocity half-kick.
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @return Last field energy contribution written this launch [eV].
    double last_energy() const noexcept { return last_u_; }

  private:
    double last_u_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
