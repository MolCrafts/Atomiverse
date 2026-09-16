/// @file cpu/observe/virial_stress.h
/// @brief CPU kinetic-stress contribution to `frame.stress`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Computes the kinetic-stress contribution `Σ m·v_α·v_β` and
/// writes it to `frame.stress_kinetic` (Voigt order: xx, yy, zz, xy,
/// xz, yz).
///
/// Pair virial is accumulated separately by the pair potentials (LJ
/// writes `r_ij · F_ij` into `frame.stress`). The split lets each
/// piece carry its own time slice -- the virial is fixed once forces
/// are computed at `post_force`, but the kinetic stress changes
/// every time velocities are updated. Integrators like
/// `NoseHooverNPT` that need the pressure *after* a velocity-rescale
/// step recompute the kinetic part fresh from current velocities
/// (the trace equals `2·KE`); they read only `frame.stress` (virial)
/// from this component's output. The reduce across MPI ranks happens
/// separately at `end_of_step @ -50` (see `Reduce`), so any
/// component scheduled after that point reads the global values.
///
/// ── LAMMPS `compute pressure` correspondence (`lammps-api-align-04`) ──
///
/// Reference: LAMMPS `src/compute_pressure.cpp::compute_scalar()`
/// (branch `stable`). The canonical scalar pressure is
///
///     P = (2·KE + Σ virial) / (d·V)        [d = 3 in 3-D]
///
/// — `compute_pressure.cpp:257`:
/// `scalar = (dof·boltz·T + virial[0]+virial[1]+virial[2])/3·(1/V)·nktv2p`,
/// where `dof·boltz·T = Σ m·v²` is exactly `2·KE`. Per-parameter audit:
///
///   | Aspect       | LAMMPS `compute pressure`           | Atomiverse                              | Verdict |
///   |--------------|-------------------------------------|-----------------------------------------|---------|
///   | Kinetic term | `Σ m·vα·vβ`, trace = `2·KE`          | this component → `frame.stress_kinetic` | aligned |
///   | Voigt order  | xx, yy, zz, xy, xz, yz              | same                                    | aligned |
///   | Virial sign  | `Σ r·F`; repulsion ⇒ positive trace | `PairLJCut` accumulates `Σ dx·(dx·fpair)` with `fpair > 0` for repulsion ⇒ positive trace | aligned |
///   | Pressure sign| `P = +trace/(dV)`, positive ⇔ compressive | `NoseHooverNPT` and `Barostat` both compute `+trace/(3V)` | aligned |
///
/// Virial-sign verdict: a repulsive (over-compressed) state yields a
/// positive `frame.stress` trace, so `P = +trace(σ)/(3V)` is positive
/// (compressive) — matching `compute pressure` and both barostats. No
/// sign correction needed; `frame.stress` is already LAMMPS-consistent.
/// **Audit confirmed aligned — no code change to this component.**
class Virial : public cpu::Observer {
  public:
    std::string name() const override { return "virial"; }
    bool provides_system_stress() const override { return true; }
    /// @brief Writes `frame.stress` / `frame.stress_kinetic` — the driver
    /// infrastructure scan detects the canonical Stress provider via
    /// `role()==Observer && capabilities()&Stress`, not via `name()`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, -50)};
    }
    void launch(const cpu::LaunchContext& ctx) override;
};

} // namespace md
} // namespace cpu
} // namespace ATV
