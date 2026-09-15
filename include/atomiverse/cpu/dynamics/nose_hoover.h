/// @file cpu/dynamics/nose_hoover.h
/// @brief CPU Nose-Hoover thermostat -- LAMMPS-equivalent `fix nvt`.
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

/// @brief Single-chain Nose-Hoover thermostat (Martyna et al., 1992).
///
/// Mirrors LAMMPS `fix nvt temp Tstart Tstop Tdamp` (the NVT branch of
/// `FixNH`, `lammps/src/fix_nh.cpp`). Updates the thermostat variable
/// `xi_` from the kinetic-energy error and rescales velocities to drive
/// the system toward the live target temperature.
///
/// Auto-binds to `initial_integrate` and `final_integrate` at order
/// `+100` (after the velocity-Verlet half-kicks).
///
/// ── LAMMPS correspondence ───────────────────────────────────────────
///
/// Reference: LAMMPS `src/fix_nh.cpp` NVT branch (`tstat_flag`)
/// (github.com/lammps/lammps, branch `stable`, fetched 2026-05-20).
/// Audit verdict per parameter (spec lammps-api-align-03-thermostats):
///
///   | Aspect            | LAMMPS `fix nvt temp`                               | This class                          | Verdict |
///   |-------------------|-----------------------------------------------------|-------------------------------------|---------|
///   | Argument order    | `temp Tstart Tstop Tdamp` (`fix_nh.cpp:132-135`)    | ctor `(T_start, tau_T, dt)` + `set_T_stop` | aligned (rename) |
///   | `Tdamp` meaning   | thermostat coupling *time constant* [time]; `t_freq = 1/t_period` (`fix_nh.cpp:511`) | `tau_T` [fs]; `Q_ = Nf·kB·T·tau_T²`  | aligned (renamed `tau`→`tau_T`) |
///   | Temperature spec  | `Tstart`/`Tstop` linear ramp (`compute_temp_target()`, `fix_nh.cpp:2204`) | `T_start`/`T_stop` + `set_run_span` ramp | aligned (fixed) |
///   | Ramp formula      | `delta = (ntimestep-beginstep)/(endstep-beginstep)`, `t_target = t_start + delta·(t_stop-t_start)` (`fix_nh.cpp:2206-2209`) | identical; `delta` clamped to `[0,1]` | aligned |
///   | Chain length      | `mtchain` default 3 (`fix_nh.cpp:78`); full M-chain `nhc_temp_integrate()` | single chain (`xi_` scalar)         | **documented divergence** — see below |
///   | `drag`            | optional empirical drag, default 0 (`fix_nh.cpp:75`); `tdrag_factor = 1 - dt·t_freq·drag/nc_tchain` (`fix_nh.cpp:692`) | not supported                       | **documented divergence** — see below |
///   | DOF count         | `tdof = 3N − extra_dof` (COM removed)               | `n_dof_global_ = 3N − 3`, cached at `build()` | aligned |
///
/// **Chain-length / `drag` divergence (deliberate, source-justified).**
/// LAMMPS `FixNH` carries an M-link Nosé-Hoover *chain* (`mtchain`,
/// default 3) plus an empirical `drag` factor, both exercised by
/// `nhc_temp_integrate()`. `cpu::md::NoseHoover` is deliberately the
/// classical *single-chain* Nosé-Hoover thermostat (one scalar `xi_`):
/// it is the simplest correct canonical-ensemble thermostat and is kept
/// as a lightweight option. The full LAMMPS-parity M-chain thermostat
/// (`mtchain` default 3, `drag` default 0, per-link `tdrag_factor`
/// matching `fix_nh.cpp::nhc_temp_integrate`) already exists in the
/// sibling component `cpu::md::NoseHooverNPT` (its `Config::mtchain` /
/// `Config::drag` fields). Re-implementing the M-chain here would
/// duplicate that code with no new capability. Verdict: **accepted
/// divergence** — callers needing LAMMPS `tchain`/`drag` parity use
/// `NoseHooverNPT` (`fix npt`/`fix nvt`-with-chains); `NoseHoover`
/// remains the single-chain form. The single-chain thermostat still
/// samples the correct canonical ensemble (Martyna et al. 1992 §II);
/// the M-chain only improves ergodicity for stiff few-DOF systems.
///
/// ── Temperature ramp ────────────────────────────────────────────────
///
/// `T_stop` (set via `set_T_stop()`, default `= T_start`) enables a
/// linear thermostat-setpoint ramp. With the run span `[nfirst, nlast]`
/// supplied via `set_run_span()` (mirrors LAMMPS capturing
/// `update->beginstep` / `endstep`), the live setpoint each step is
///
///     delta    = clamp((step − nfirst) / (nlast − nfirst), 0, 1)
///     T_live   = T_start + delta·(T_stop − T_start)
///
/// exactly as `fix_nh.cpp::compute_temp_target()`. With `T_stop ==
/// T_start` (the default) the ramp is the constant `T_start` — no
/// behavioural change for existing callers. `set_T_target` is retained
/// for the segmented-run idiom (retune the setpoint between `run()`
/// calls); the ramp is the continuous alternative.
class NoseHoover : public cpu::Thermostat {
  public:
    /// @param T_start Ramp start / target temperature [K]
    ///                (LAMMPS `Tstart`).
    /// @param tau_T   Thermostat coupling time constant [fs]
    ///                (LAMMPS `Tdamp`).
    /// @param dt      Timestep [fs].
    NoseHoover(double T_start, double tau_T, double dt);

    std::string name() const override { return "nose_hoover"; }
    /* role via Thermostat base */
    /// @brief Write-set: rescales velocities by the thermostat factor.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, +100),
                ATV::make_binding(ATV::md::Phase::final_integrate, +100)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Retune the thermostat setpoint between `Driver::run()` calls.
    ///
    /// Updates `T_start_` (the ramp anchor) and rescales the chain mass
    /// `Q_` to the new setpoint using the global DOF count cached at
    /// `build()`. Chain coordinate `xi_` is preserved by default — a
    /// continuous ramp keeps the integrator state, where a hard reset
    /// (e.g. between independent equilibration runs) would zero it.
    /// `T_stop_` is also moved to the new value, so a `set_T_target`
    /// call cancels any previously configured continuous ramp; configure
    /// the ramp again afterwards with `set_T_stop` if needed.
    ///
    /// Invariant: must be called between `Driver::run(...)` invocations on
    /// a built driver, never during a launch. The setter does not touch
    /// `n_dof_global_`, so the post-build atom count is reused; rebuild
    /// the driver if the system size changes.
    ///
    /// @param T_target New target temperature [K].
    /// @param reset_chain If true, also resets the integrator variable
    ///        `xi_` to 0 (use only when starting a fresh equilibration).
    void set_T_target(double T_target, bool reset_chain = false);

    /// @brief Set the ramp end temperature [K] (LAMMPS `Tstop`).
    ///
    /// Enables a continuous linear ramp from `T_start` to `T_stop` over
    /// the run span set by `set_run_span()`. With `T_stop == T_start`
    /// (the default) there is no ramp. Mirrors the second temperature
    /// argument of LAMMPS `fix nvt temp Tstart Tstop Tdamp`.
    ///
    /// @param T_stop Ramp end temperature [K].
    /// @return `*this`, so the call can chain before registration.
    NoseHoover& set_T_stop(double T_stop) {
        T_stop_ = T_stop;
        return *this;
    }

    /// @brief Set the temperature-ramp run span `[nfirst, nlast]` (in MD
    ///        step indices), mirroring LAMMPS `update->beginstep` /
    ///        `update->endstep` captured at run setup.
    ///
    /// Must be called before the driver's `run()` if a ramp
    /// (`T_stop != T_start`) is configured. With the default span
    /// (`nlast <= nfirst`) the ramp `delta` is pinned to 0, so the
    /// setpoint stays at `T_start` — a no-op.
    ///
    /// @param nfirst First MD step index of the run.
    /// @param nlast  Last MD step index of the run.
    /// @return `*this`, so the call can chain before registration.
    NoseHoover& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    /// @return Live thermostat setpoint [K] — for diagnostics / observers.
    double T_target() const noexcept { return T_live_; }

  private:
    /// @brief Interpolate the live thermostat setpoint for `step`,
    ///        mirroring LAMMPS `fix_nh.cpp::compute_temp_target()`.
    ///        Returns `T_start_` when no ramp is active.
    double compute_temp_target(std::int64_t step) const;

    double T_start_; ///< Ramp start temperature [K] (LAMMPS `t_start`).
    double T_stop_;  ///< Ramp end temperature [K] (LAMMPS `t_stop`).
    double tau_T_;   ///< Coupling time constant [fs] (LAMMPS `Tdamp`).
    double dt_;
    double xi_ = 0;        ///< Thermostat variable (single chain).
    double Q_ = 0;         ///< Thermostat mass.
    int n_dof_global_ = 0; ///< Cached at build() — used by set_T_target.

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
