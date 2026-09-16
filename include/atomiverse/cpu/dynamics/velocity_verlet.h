/// @file cpu/dynamics/velocity_verlet.h
/// @brief CPU VelocityVerlet integrator.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Velocity Verlet symplectic integrator (Swope et al., 1982).
///
/// initial_integrate: v(t+dt/2) = v(t) + (dt/2)*f(t)/m, x(t+dt) = x(t) + dt*v(t+dt/2)
/// final_integrate:   v(t+dt)   = v(t+dt/2) + (dt/2)*f(t+dt)/m
///
/// ## LAMMPS correspondence — `fix nve` (audit: ALIGNED)
///
/// Mirrors LAMMPS `fix nve` (`lammps/src/fix_nve.cpp`), the parameter-free
/// plain velocity-Verlet integrator. Per-parameter audit verdict against
/// the verbatim LAMMPS source:
///
/// | Aspect | LAMMPS `fix nve` | `VelocityVerlet` | Verdict |
/// |---|---|---|---|
/// | Parameters | none — uses the global `timestep` | `dt` ctor arg | ALIGNED — `dt` mirrors the global LAMMPS `timestep`; LAMMPS reads it per-fix via `update->dt`. |
/// | Half-kick | `init()`: `dtf = 0.5*dt*ftm2v`; `initial_integrate()`: `v += dtfm*f`, `x += dtv*v`; `final_integrate()`: `v += dtfm*f` (`fix_nve.cpp` L59, L89-94, L132-135) | `dtf = 0.5*dt_*units::force_to_accel`; `dtfm = dtf/mass`; same kick split | ALIGNED — identical formula and integrate-half split. |
/// | Unit conversion | `force->ftm2v` folded into `dtf` | `units::force_to_accel` constant folded into `dtf` | ALIGNED — `units::force_to_accel` is the project's `ftm2v` equivalent for the eV/Å/fs/amu unit system. |
///
/// No behavioural change required — the component already matches
/// `fix nve` exactly. `fix nve/sphere`/`nve/asphere` rotational variants
/// are out of scope (point-mass NVE only).
class VelocityVerlet : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    /// @param dt Timestep [fs]. Mirrors the global LAMMPS `timestep`
    ///           consumed by `fix nve` via `update->dt`.
    explicit VelocityVerlet(double dt);

    std::string name() const override { return "velocity_verlet"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @brief Published nuclear timestep [fs] (the constructor `dt`).
    double timestep() const override { return dt_; }
    /* role via Integrator base */
    /// @brief Write-set: half-kick (v) + drift (x).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    double dt_;
};

} // namespace md
} // namespace cpu
} // namespace ATV
