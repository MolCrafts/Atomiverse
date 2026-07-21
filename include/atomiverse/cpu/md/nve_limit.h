/// @file cpu/md/nve_limit.h
/// @brief CPU velocity-Verlet integrator with a per-step displacement
///        (distance) cap. LAMMPS-equivalent `fix nve/limit`
///        (`lammps/src/fix_nve_limit.cpp`).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Velocity Verlet with a per-step displacement (distance) cap.
///
/// ## LAMMPS correspondence — `fix nve/limit xmax` (audit: ALIGNED)
///
/// Mirrors LAMMPS `fix nve/limit xmax` (`lammps/src/fix_nve_limit.cpp`):
/// standard velocity-Verlet, but after every `v += dtfm*f` kick a
/// per-atom check caps the per-step displacement so `|v|·dt <= xmax` --
/// i.e. `xmax` is a **maximum distance moved per step** [Å], NOT a
/// velocity. If exceeded, `v` is rescaled (direction-preserving) so the
/// step displacement equals exactly `xmax`. The cap runs in BOTH halves
/// of the integrator (after the initial half-kick AND after the final
/// half-kick), guaranteeing that no atom moves more than `xmax` in any
/// single step.
///
/// Per-parameter audit verdict against the verbatim LAMMPS source:
///
/// | Aspect | LAMMPS `fix nve/limit` | `NVELimit` | Verdict |
/// |---|---|---|---|
/// | `xmax` meaning | max distance moved per step [distance unit]; `init()`: `vlimitsq = (xlimit/dtv)^2` (`fix_nve_limit.cpp` L66) | `vmax = xmax_/dt_`, `vmax_sq = vmax*vmax` | ALIGNED — `xmax` is a distance; the cap is `|v|·dt <= xmax`, equivalently `|v| <= xmax/dt`. |
/// | Cap norm | Euclidean: `vsq = v·v`; `if (vsq > vlimitsq)` (L105-112, L168-175) | `vsq = vx²+vy²+vz²`; `if (vsq > vmax_sq)` | ALIGNED — vector L2 norm, not per-component. |
/// | Rescale | `scale = sqrt(vlimitsq/vsq); v *= scale` | `s = vmax/sqrt(vsq); v *= s` | ALIGNED — `vmax/sqrt(vsq) == sqrt(vmax²/vsq) == sqrt(vlimitsq/vsq)`. |
/// | Both halves | cap applied in `initial_integrate()` AND `final_integrate()` (L83-143, L147-198) | cap applied in both `initial_integrate` and `final_integrate` branches | ALIGNED — verified in `nve_limit.cpp` and by `test_nve_limit_lammps_align`. |
/// | `xmax` default | required (`narg != 4` → error, L35) | required ctor arg, no default | ALIGNED — Atomiverse keeps it required. |
/// | Argument order | `fix nve/limit xmax` (timestep is global) | ctor `(dt, xmax)` | ALIGNED (semantic) — `dt` mirrors the global `timestep`; `xmax` is the lone `fix`-line argument. |
///
/// No behavioural change required — the component already matches
/// `fix nve/limit` exactly.
///
/// Use case: integrate from far-from-equilibrium initial conditions
/// (e.g. compressed amorphous polymer melts) without atoms flying apart
/// from transient force spikes.  Once the system relaxes, the cap is
/// rarely hit and the integrator behaves as plain velocity-Verlet.
///
/// Bound to `initial_integrate @ 0` and `final_integrate @ 0` --
/// replaces (do not combine with) `VelocityVerlet`.
class NVELimit : public cpu::Integrator {
  public:
    /// @param dt   Timestep [fs]. Mirrors the global LAMMPS `timestep`.
    /// @param xmax Maximum distance an atom may move in one step [Å]
    ///             (LAMMPS `fix nve/limit xmax`). The cap enforces
    ///             `|v|·dt <= xmax` using the Euclidean speed, i.e.
    ///             `|v| <= xmax/dt`.
    NVELimit(double dt, double xmax);

    std::string name() const override { return "nve_limit"; }
    /* role via Integrator base */
    /// @brief Write-set: velocity Verlet with a per-step displacement cap.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    double dt_;
    double xmax_;
};

} // namespace md
} // namespace cpu
} // namespace ATV
