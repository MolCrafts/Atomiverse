/// @file cpu/md/barostat.h
/// @brief CPU Berendsen isotropic barostat.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/barostat.h"
#include "atomiverse/facade/stage.h"

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
///   | `Pdamp`           | barostat coupling time constant [time]     | `tau_P` [fs]                                   | aligned   |
///   | `bulkmodulus` (β) | `modulus` keyword, default `10.0` (P units)| `bulk_modulus` keyword; default `1.0` — see [†]| divergence (default) |
///   | Dilation formula  | `μ = (1 − dt/Pdamp·(P_t − P)/β)^(1/3)`     | identical                                      | aligned   |
///   | Box+atom remap    | `remap()` rescales boxlo/hi + coords       | `box.scale_isotropic(μ)` + affine atom rescale | aligned   |
///   | Parameter order   | `fix press/berendsen iso P P Pdamp`        | ctor `(P_target, tau_P, dt, bulk_modulus)`     | aligned   |
///
/// [†] **Source-justified divergence — `bulk_modulus` default.** LAMMPS
/// `fix_press_berendsen.cpp:52` initialises `bulkmodulus = 10.0`. The
/// `bulkmodulus` keyword and the `1/β` division in the dilation formula
/// are reproduced verbatim, but the Atomiverse *default* is `1.0`, not
/// `10.0`. Rationale: the pre-`lammps-api-align-04` `Barostat` folded
/// `β` to `1` (no modulus term at all). The link's mandate is that the
/// pressure-sign flip be a *representation change only* — `μ`
/// numerically unchanged for the same physical state. A `β = 10`
/// default would silently change `μ` for every existing 3-argument
/// caller (10× slower relaxation), which is a behaviour change the link
/// explicitly forbids. Defaulting `β = 1` keeps the flip purely
/// representational; callers that want LAMMPS-identical relaxation pass
/// `bulk_modulus = 10.0` explicitly (`fix press/berendsen ... modulus 10`).
///
/// Sign-convention note: this `P = +trace/(3V)` matches `NoseHooverNPT`
/// and LAMMPS `compute pressure`. The link `lammps-api-align-04` flipped
/// the previous `P = −trace/(3V)` here; because the dilation argument was
/// re-derived together with the sign (`(P_target − P)` instead of
/// `(P − P_target)`), `μ` is numerically unchanged for the same physical
/// state — a representation change, not a behaviour change.
class Barostat : public cpu::Barostat {
  public:
    /// @brief LAMMPS `fix press/berendsen`-equivalent isotropic barostat.
    /// @param P_target     Target pressure [eV/Å³], LAMMPS-standard sign
    ///                     (`+trace(σ)/(3V)`, positive ⇔ compressive).
    ///                     1 bar ≈ 6.2415e-7 eV/Å³.
    /// @param tau_P        Barostat coupling time constant [fs]
    ///                     (LAMMPS `Pdamp`).
    /// @param dt           MD step size [fs].
    /// @param bulk_modulus Tunable modulus `β` [eV/Å³] (LAMMPS
    ///                     `bulkmodulus`). Larger ⇒ smaller per-step
    ///                     rescale ⇒ slower relaxation. Default `1.0`
    ///                     — a documented divergence from LAMMPS's
    ///                     `10.0` so the `lammps-api-align-04` sign
    ///                     flip stays representation-only (see the
    ///                     class-level `[†]` note); pass `10.0` for
    ///                     LAMMPS-identical relaxation.
    Barostat(double P_target, double tau_P, double dt,
             double bulk_modulus = 1.0);

    std::string name() const override { return "barostat"; }
    /* role via Barostat base */
    /// @brief Reads `frame.stress` to derive internal pressure for
    /// the rescaling factor `μ`. Triggers auto-injection of
    /// `Virial` in `MD::install_defaults()` when no canonical
    /// stress provider is otherwise registered.
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
