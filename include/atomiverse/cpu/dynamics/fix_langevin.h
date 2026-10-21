/// @file cpu/dynamics/fix_langevin.h
/// @brief CPU Langevin thermostat -- LAMMPS-equivalent `fix langevin`.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/thermostat.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Schneider-Stoll Langevin thermostat applied as an additive force.
///
/// Mirrors LAMMPS `fix langevin Tstart Tstop damp seed`
/// (`lammps/src/fix_langevin.cpp`). At every force evaluation, before the
/// velocity update, two forces are added per atom:
///
/// 1. Friction (Stokes drag):  F_drag = - m / (damp · ftm2v) · v
/// 2. Random kick:              F_rand =   sqrt(2 m kB T / (damp · dt · mvv2e)) / ftm2v · ξ
///
/// where `ξ ~ N(0, 1)` per Cartesian component per atom per force evaluation,
/// `ftm2v = units::force_to_accel` and `mvv2e = units::mass_v2_to_energy` (the metal-unit conversion
/// constants). The combined force satisfies the fluctuation-dissipation
/// theorem so velocity-Verlet integration of `f_total = f_phys + F_drag
/// + F_rand` produces the canonical NVT ensemble at the live setpoint.
///
/// ── LAMMPS correspondence ───────────────────────────────────────────
///
/// Reference: LAMMPS `src/fix_langevin.cpp`
/// (github.com/lammps/lammps, branch `stable`, fetched 2026-05-20).
/// Audit verdict per parameter (spec lammps-api-align-03-thermostats):
///
///   | Aspect            | LAMMPS `fix langevin`              | This class                          | Verdict |
///   |-------------------|-----------------------------------|-------------------------------------|---------|
///   | Argument order    | `Tstart Tstop damp seed`          | ctor `(T_start, T_stop, damp, dt, seed)` | aligned (fixed) |
///   | Temperature spec  | `Tstart`/`Tstop` linear ramp      | `T_start`/`T_stop` + `set_run_span` ramp | aligned (fixed) |
///   | `damp` meaning    | friction *time constant* [time]; γ = 1/damp; `gfactor1 = -m/t_period/ftm2v` (`fix_langevin.cpp:306`) | `damp` [fs], `gfactor1 = -m/(damp·ftm2v)` | aligned |
///   | Ramp formula      | `compute_target()`: `delta = (ntimestep-beginstep)/(endstep-beginstep)`, `t_target = t_start + delta·(t_stop-t_start)`, random force ∝ `sqrt(t_target)` (`fix_langevin.cpp:794-802,661`) | identical; `delta` clamped to `[0,1]`; random force ∝ `sqrt(T_live)` | aligned |
///   | `seed`            | seeds Marsaglia `RanMars`; positive int (`fix_langevin.cpp:78,84`) | `std::uint32_t` splitmix stream | divergence (RNG by design — see below) |
///   | Force formula     | drag + FDT-consistent random kick | Schneider-Stoll drag + random, FDT-consistent | aligned |
///   | `scale` keyword   | optional per-type friction scaling | not supported | deferred (spec out-of-scope) |
///   | `tally` keyword   | reservoir energy tally             | not supported | deferred (spec out-of-scope) |
///   | `zero` keyword    | COM-drift removal                  | not supported | deferred — overlaps `ZeroMomentum` |
///   | `angmom`/`omega`  | rotational-DOF thermostatting      | not supported | deferred (spec out-of-scope) |
///
/// **RNG divergence (deliberate).** LAMMPS draws uniform samples from
/// the Marsaglia-Tsang `RanMars` generator; Atomiverse draws Gaussian
/// samples from a per-`(seed, force-evaluation sample, atom-tag)` splitmix
/// stream. Build resets the sample sequence; every force launch consumes
/// one sample, including launches with no owned atoms. This keeps ranks in
/// lockstep and gives the initial-force bootstrap a different random vector
/// from the first integrated step, even when both have `step == 0`.
/// The trajectory step controls only the temperature ramp. The two generators
/// are statistically equivalent in the many-particle limit -- ⟨T⟩ and
/// the velocity distribution match -- but individual trajectories
/// differ. Only ensemble averages are contracted to match LAMMPS.
///
/// ── Temperature ramp ────────────────────────────────────────────────
///
/// `T_stop` (ctor arg 2, default sentinel `< 0` ⇒ folds to `T_start`)
/// enables a linear thermostat-setpoint ramp. With the run span
/// `[nfirst, nlast]` supplied via `set_run_span()` (mirrors LAMMPS
/// capturing `update->beginstep` / `endstep` at run setup), the live
/// setpoint each step is
///
///     delta    = clamp((step − nfirst) / (nlast − nfirst), 0, 1)
///     T_live   = T_start + delta·(T_stop − T_start)
///
/// exactly as `fix_langevin.cpp::compute_target()`. With `T_stop ==
/// T_start` (the default) the ramp is the constant `T_start`, i.e. no
/// behavioural change for existing callers.
///
/// Bound at `post_force @ +500` so the thermostat forces are added
/// after every physical-force evaluator has accumulated, but before
/// any user-defined post-force corrections that might depend on the
/// total. Pair this component with `VelocityVerlet` (or `NVELimit`
/// for far-from-equilibrium starts) -- the integrator picks up the
/// modified forces in the next velocity half-kick.
class FixLangevin : public cpu::Thermostat {
  public:
    /// @param T_start Ramp start temperature [K] (LAMMPS `Tstart`).
    /// @param T_stop  Ramp end temperature [K] (LAMMPS `Tstop`). A
    ///                negative value (the default) means "no ramp" —
    ///                it folds to `T_start`, a constant-T thermostat.
    /// @param damp    Damping time constant [fs] (LAMMPS `damp`).
    ///                Effective friction γ = 1 / damp.
    /// @param dt      Timestep [fs].
    /// @param seed    RNG seed; reproducible across rank counts.
    FixLangevin(double T_start, double T_stop, double damp, double dt,
                std::uint32_t seed);

    /// @brief Set the temperature-ramp run span `[nfirst, nlast]` (in MD
    ///        step indices), mirroring LAMMPS `update->beginstep` /
    ///        `update->endstep` captured at run setup.
    ///
    /// Must be called before the driver's `run()` if a ramp
    /// (`T_stop != T_start`) is configured. With the default span
    /// (`nlast <= nfirst`) the ramp `delta` is pinned to 0, so the
    /// setpoint stays at `T_start` — a no-op, consistent with the
    /// `T_stop == T_start` default.
    ///
    /// @param nfirst First MD step index of the run.
    /// @param nlast  Last MD step index of the run.
    /// @return `*this`, so the call can chain before registration.
    FixLangevin& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    /// @return Live thermostat setpoint [K] — for diagnostics / observers.
    ///         Equals `T_start` until the first `launch()` of a ramped run.
    double T_target() const noexcept { return T_live_; }

    std::string name() const override { return "fix_langevin"; }
    /* role via Thermostat base */
    /// @brief Writes drag + random forces; the Thermostat kind stays
    ///        energy/force-false so only this concrete overrides.
    bool provides_force() const override { return true; }
    /// @brief Langevin's role is Thermostat, but it functionally writes
    /// drag + random forces into fx/fy/fz — so it advertises
    /// `provides_force()`. This lets a pure Langevin run (no LJ / bonded
    /// evaluator) satisfy `cpu::md::MD::validate_components()`'s "something
    /// writes forces" contract.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +500)};
    }
    /// @brief Restart the random sample sequence for a fresh trajectory.
    /// @param ctx Build context.
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    double T_start_; ///< Ramp start temperature [K] (LAMMPS `t_start`).
    double T_stop_;  ///< Ramp end temperature [K] (LAMMPS `t_stop`).
    double damp_;
    double dt_;
    std::uint32_t seed_;
    /// One random sample per force evaluation, independent of step cadence.
    /// Reset on build; retained across continued trajectory segments.
    std::uint64_t sample_counter_ = 0;

    // Temperature-ramp run span (LAMMPS `update->beginstep` / `endstep`).
    std::int64_t nfirst_ = 0; ///< First MD step of the run.
    std::int64_t nlast_ = 0;  ///< Last MD step of the run.
    /// @brief Live thermostat setpoint [K], interpolated each step from
    /// the ramp; equals `T_start_` when no ramp is active.
    double T_live_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
