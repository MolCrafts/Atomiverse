/// @file cpu/dynamics/barostat.h
/// @brief CPU Berendsen isotropic barostat.
#pragma once

#include "atomiverse/cpu/component/timestep_protocol.h"
#include <cmath>
#include <stdexcept>

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/barostat.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Berendsen isotropic pressure coupling (Berendsen et al. 1984).
///
/// LAMMPS-equivalent `fix press/berendsen iso P P Pdamp` (optionally
/// `... modulus K`). At each call it computes the LAMMPS-standard
/// scalar pressure `P = +trace(σ)/(3·V)` (positive ⇔ compressive)
/// from `frame.stress` + `frame.stress_kinetic` (both in unnormalised
/// virial units; this component applies the `V` division), then
/// rescales the box edges and every owned atom's position by the
/// dilation factor
///
///     μ = (1 − dt/(Pdamp·β) · (P_target − P))^(1/3)
///
/// — exactly `fix_press_berendsen.cpp::end_of_step()`'s `dilation`.
///
/// Bound at `end_of_step @ -45`, after `Reduce @ -50`. In multi-rank
/// runs `frame.stress` is the globally-reduced virial after Reduce,
/// so every rank computes the SAME `μ` and applies the SAME box update --
/// the box stays consistent across ranks without any extra collective.
///
/// ── LAMMPS correspondence (`src/fix_press_berendsen.cpp`) ────────────
///
/// Reference: LAMMPS `src/fix_press_berendsen.cpp`, fetched verbatim
/// (`github.com/lammps/lammps`, branch `stable`). Per-parameter audit:
///
///   | Aspect            | LAMMPS `fix press/berendsen`               | `Barostat` (this class)                        | Verdict   |
///   |-------------------|--------------------------------------------|------------------------------------------------|-----------|
///   | Pressure sign     | `P = +trace(σ)/(3V)` (`compute pressure`)  | `P = +trace(σ)/(3V)` — LAMMPS-standard         | aligned   |
///   | `Pdamp`           | barostat coupling time constant [time]     | `tau_P` [style time]                           | aligned   |
///   | `bulkmodulus` (β) | `modulus` keyword, default `10.0` (P units)| `bulk_modulus`, default `10.0` [style pressure]| aligned   |
///   | Dilation formula  | `μ = (1 − dt/Pdamp·(P_t − P)/β)^(1/3)`     | identical                                      | aligned   |
///   | Box+atom remap    | `remap()` rescales boxlo/hi + coords       | `box.scale_isotropic(μ)` + affine atom rescale | aligned   |
///   | Parameter order   | `fix press/berendsen iso P P Pdamp`        | ctor `(P_target, tau_P, dt, bulk_modulus)`     | aligned   |
///
/// `bulk_modulus` defaults to LAMMPS's `bulkmodulus = 10.0`
/// (`fix_press_berendsen.cpp:52`), in the run's pressure unit like the
/// target (real: atm, metal: bar). The earlier default of 1.0 dated from
/// pressures in eV/Å³, where it meant ~1.6 Mbar; read in bar or atm it made
/// the dilation ~10⁶ times more aggressive than intended.
///
/// Sign-convention note: this `P = +trace/(3V)` matches `NoseHooverNPT`
/// and LAMMPS `compute pressure`. The link `lammps-api-align-04` flipped
/// the previous `P = −trace/(3V)` here; because the dilation argument was
/// re-derived together with the sign (`(P_target − P)` instead of
/// `(P − P_target)`), `μ` is numerically unchanged for the same physical
/// state — a representation change, not a behaviour change.
class Barostat : public cpu::Barostat, public cpu::TimestepConsumer {
  public:
    /// @brief Consume the enclosing integrator's timestep [style time].
    cpu::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return false; }
    void set_timestep(double timestep) override {
        if (!std::isfinite(timestep) || timestep <= 0.0)
            throw std::invalid_argument(
                "Barostat: timestep must be finite and positive");
        dt_ = timestep;
    }

    /// @brief LAMMPS `fix press/berendsen`-equivalent isotropic barostat.
    /// @param P_target     Target pressure [style pressure], LAMMPS-standard sign
    ///                     (`+trace(σ)/(3V)`, positive ⇔ compressive).
    ///                     Like LAMMPS, the run's pressure unit (metal: bar, real: atm).
    /// @param tau_P        Barostat coupling time constant [style time]
    ///                     (LAMMPS `Pdamp`).
    /// @param dt           MD step size [style time].
    /// @param bulk_modulus Tunable modulus `β` [style pressure] (LAMMPS
    ///                     `modulus`). Larger ⇒ smaller per-step
    ///                     rescale ⇒ slower relaxation. Default `10.0`,
    ///                     the LAMMPS default.
    Barostat(double P_target, double tau_P, double dt,
             double bulk_modulus = 10.0);

    std::string name() const override { return "barostat"; }
    /* role via Barostat base */
    /// @brief Reads `frame.stress` and `frame.stress_kinetic` (written by
    /// `KineticEnergy` at end_of_step @ -100) to derive the internal
    /// pressure for the rescaling factor `μ`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, -45)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Toggle pressure coupling at runtime (default: enabled).
    /// When disabled `launch()` is a no-op — useful for multi-stage
    /// protocols that mix NVT and NPT segments under one driver.
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
    bool enabled() const noexcept { return enabled_; }

    /// @brief Retune the pressure setpoint between `Driver::run()` segments.
    void set_P_target(double P_target) noexcept { P_target_ = P_target; }
    double P_target() const noexcept { return P_target_; }

    /// @brief Retune the modulus `β` (LAMMPS `bulkmodulus`) at runtime.
    void set_bulk_modulus(double bulk_modulus) noexcept {
        bulk_modulus_ = bulk_modulus;
    }
    double bulk_modulus() const noexcept { return bulk_modulus_; }

  private:
    double P_target_, tau_P_, dt_, bulk_modulus_;
    bool enabled_ = true;
};

} // namespace md
} // namespace cpu
} // namespace ATV
