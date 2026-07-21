/// @file cpu/md/nose_hoover_npt.h
/// @brief CPU MTK Nose-Hoover NPT integrator — isotropic *or* anisotropic
///        per-axis barostat, with an optional continuous temperature ramp.
///        LAMMPS-equivalent `fix npt iso` / `fix npt x .. y .. z ..`.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Martyna-Tobias-Klein isotropic NPT integrator with M-chain NHC.
///
/// Mirrors LAMMPS `fix npt temp T_target T_target tau_T iso P_target
/// P_target tau_P mtk`. Each of the two NHC objects (one for the atom
/// KE, one for the barostat KE) is a **Nosé-Hoover chain of length M**
/// (default `M = 3`, matching LAMMPS `mtchain`/`mpchain`). The chain
/// provides natural damping of the innermost thermostat-momentum
/// oscillation that a single-chain integrator cannot absorb, which is
/// what keeps an over-compressed start-state (large `P_target − P_int`)
/// from producing the 8000 K-then-freeze ringing the debug_amorphous
/// harness exhibited under length-1 chains.
///
/// Algorithm (Tuckerman 2006, *J. Phys. A* 39, 5629; Shinoda 2004,
/// *Phys. Rev. B* 69, 134103):
///
/// Trotter splitting per timestep `dt`. The outer-NHC pair runs
/// baro-then-atoms on the front half and atoms-then-baro on the back
/// half — required for symmetric Trotter and matches LAMMPS
/// `fix_nh.cpp::initial_integrate` / `final_integrate` ordering:
///
///     [P1] NHC-baro step       (dt/2)   updates η_b, η̇_b; scales ε̇
///     [P2] NHC-atom step       (dt/2)   updates η_t, η̇_t; scales v
///     [P3] barostat-momentum   (dt/2)   ε̇ += G_ε · dt/2
///     [P4] barostat-velocity   (dt/2)   v *= exp(-(1+3/N) · ε̇ · dt/2)
///     [P5] velocity half-kick  (dt/2)   v += (dt/2) · F/m
///     [P6] position+box update (dt)     x = x · e^{ε̇dt} + v · dt · (e^{ε̇dt}-1)/(ε̇dt);
///                                       box *= e^{ε̇dt}
///                                  ↓ (force evaluation; pair virial)
///     [P7] velocity half-kick  (dt/2)   v += (dt/2) · F/m
///     [P8] barostat-velocity   (dt/2)
///     [P9] barostat-momentum   (dt/2)   ε̇ += G_ε(P_new, KE_new) · dt/2
///     [P10] NHC-atom step      (dt/2)
///     [P11] NHC-baro step      (dt/2)
///
/// where the driving force is
///
///     G_ε = [3·V·(P_internal - P_target) + (6/N_dof)·KE] / W
///
/// with `P_internal = (σ_xx + σ_yy + σ_zz) / (3V)` — the LAMMPS-standard
/// pressure-tensor convention (positive trace ⇔ compressive). `P_target`
/// uses the same sign convention as LAMMPS `fix npt iso`, so passing
/// `P_target = 0` matches LAMMPS at the same setpoint. Since
/// `lammps-api-align-04` this is the SAME sign convention used by
/// `cpu::md::Barostat` (Berendsen) and `cpu::md::Virial` /
/// `compute pressure`: a single positive-trace-is-compressive
/// convention holds across every Atomiverse pressure component.
///
/// Phase distribution:
///
///   * `initial_integrate @ 0`  -- steps P1 through P6
///   * `final_integrate   @ 0`  -- step  P7
///   * `end_of_step      @ -40` -- steps P8 through P11
///                                 (after `Reduce @ -50` so
///                                  pressure & KE are global)
///
/// ── LAMMPS correspondence ───────────────────────────────────────────
///
/// Reference: LAMMPS `src/fix_nh.cpp` (github.com/lammps/lammps,
/// release 22 Jul 2025). Atomiverse implements the same MTK scheme;
/// each helper maps to a `FixNH` method:
///
///   | Atomiverse (this class)        | LAMMPS `FixNH` (`src/fix_nh.cpp`) |
///   |--------------------------------|-----------------------------------|
///   | `build()` (Q/W mass init)      | `FixNH::setup()`                  |
///   | `launch()` initial_integrate   | `FixNH::initial_integrate()`      |
///   | `launch()` final + end_of_step | `FixNH::final_integrate()`        |
///   | `nhc_atoms_half()`             | `FixNH::nhc_temp_integrate()`     |
///   | `nhc_baro_half()`              | `FixNH::nhc_press_integrate()`    |
///   | `update_eps_dot()`             | `FixNH::nh_omega_dot()`           |
///   | `scale_v_by_baro()`            | `FixNH::nh_v_press()`             |
///   | `kick_v_by_force()`            | `FixNH::nve_v()`                  |
///   | `propagate_x_and_box()`        | `FixNH::nve_x()` + `FixNH::remap()`|
///
/// One deliberate divergence: LAMMPS splits the central box operator
/// as `remap(dt/2) → nve_x(dt) → remap(dt/2)`; Atomiverse fuses it
/// into one `propagate_x_and_box()` using the exact analytic drift
/// factor `(e^{ε̇dt}-1)/(ε̇dt)` — equal to LAMMPS's two-remap result
/// to O((ε̇dt)²).
///
/// ── Per-parameter audit (`lammps-api-align-04`) ──────────────────────
///
/// Every `Config` field maps to a `fix npt` keyword. Audit verdict —
/// each row checked against `src/fix_nh.cpp` (branch `stable`):
///
///   | `Config` field   | LAMMPS `fix npt` keyword            | Verdict |
///   |------------------|-------------------------------------|---------|
///   | `T_target`       | `temp Tstart ...` (ramp start)      | aligned |
///   | `T_stop`         | `temp ... Tstop ...` (ramp end)     | aligned |
///   | `tau_T`          | `temp ... ... Tdamp`                | aligned |
///   | `P_target`       | `iso Pstart Pstop` (here Pstart==Pstop) | aligned |
///   | `tau_P`          | `iso ... ... Pdamp`                 | aligned |
///   | `dt`             | `timestep` (global, not a fix arg)  | aligned |
///   | `drag`           | `drag Df`                           | aligned |
///   | `mtchain`        | `mtchain N` (thermostat NHC length) | aligned |
///   | `mpchain`        | `mpchain N` (barostat NHC length)   | aligned |
///   | `couple_axis`    | `x`/`y`/`z` per-axis `p_flag[]`     | aligned |
///
/// Pressure-sign verdict: `NoseHooverNPT` already uses the LAMMPS
/// `compute pressure` convention `P = +trace(σ)/(3V)`, positive ⇔
/// compressive — no change. This is the reference sign for the whole
/// `lammps-api-align-04` link. **Audit only — no code change here;**
/// the MTK algorithm itself was aligned in `msu-pe-align-01-aniso-npt`.
///
/// ── Anisotropic per-axis barostat (msu-pe-align-01) ──────────────────
///
/// `Config::couple_axis` (`std::array<bool,3>`, default `{true,true,true}`)
/// selects which of x, y, z are barostatted. The barostat momentum is a
/// per-axis vector `ε̇_α` (member `eps_dot_`), mirroring LAMMPS's per-axis
/// `omega_dot[]` and `p_flag[]`. For each active axis α the driving force
/// is the MTK per-axis form
///
///     G_{ε,α} = [ V·(σ_αα/V − P_target) + (2/N_dof)·KE ] / W
///
/// (the diagonal LAMMPS `fix_nh.cpp::nh_omega_dot` `f_omega`, scaled by
/// our `W = 3·omega_mass`). An axis with `couple_axis[α]==false` keeps
/// `ε̇_α == 0` and its box edge is never rescaled — a rigid (NVT-along-α)
/// edge, matching LAMMPS `pdim`/`p_flag`. With all three flags `true`
/// and one shared `P_target`, the per-axis update reduces algebraically
/// to the isotropic `G_ε = 3·G_{ε,iso}` so the all-axes path is
/// bit-compatible with the pre-change isotropic integrator.
///
/// ── Continuous temperature ramp (msu-pe-align-01) ────────────────────
///
/// `Config::T_stop` (default `= T_target`) enables a linear thermostat
/// setpoint ramp. With the run span `[nfirst, nlast]` supplied via
/// `set_run_span()` (mirrors LAMMPS capturing `update->beginstep` /
/// `endstep` at run setup), the live setpoint each step is
///
///     delta    = (step − nfirst) / (nlast − nfirst)   ∈ [0,1]
///     T_target = T_start + delta·(T_stop − T_start)
///
/// exactly as `fix_nh.cpp::compute_temp_target()`. With `T_stop ==
/// T_target` (the default) the ramp is the constant `T_start`, i.e. no
/// behavioural change for existing callers.
///
/// **Replaces (do not add alongside)** `VelocityVerlet`, `NoseHoover`,
/// and `Barostat`. Compatible with `KineticEnergy` -- this component
/// rewrites `frame.kinetic` after the second NHC update so any thermo
/// output bound at `end_of_step @ > -40` reads the post-NPT value.
class NoseHooverNPT : public cpu::Integrator {
  public:
    /// @brief All NPT parameters, grouped so each field is named at the
    /// call site and the order of `tau_T` vs `tau_P` can't be swapped
    /// silently. Built as an aggregate (no defaulted fields) so the
    /// compiler flags any omission via `-Wmissing-field-initializers`.
    struct Config {
        double T_target;   ///< Target temperature [K].
        double tau_T;      ///< Thermostat coupling time [fs] (LAMMPS `Tdamp`).
        double P_target;   ///< Target pressure [eV/Å³], LAMMPS-standard
                           ///  sign (`+trace(σ)/(3V)`, positive ⇔
                           ///  compressive) — same convention as LAMMPS
                           ///  `fix npt iso`. 1 bar = 6.2415e-7 eV/Å³.
        double tau_P;      ///< Barostat coupling time [fs] (LAMMPS `Pdamp`).
        double dt;         ///< Timestep [fs].
        double drag = 0.0; ///< LAMMPS-style empirical drag on the
                           ///  innermost thermostat and barostat NHC
                           ///  momenta (applied at each chain update as
                           ///  `eta_dot[0] *= 1 - drag*half_dt/tau`).
                           ///  Zero disables. `fix npt ... drag 2`.
        int mtchain = 3;   ///< Atom-NHC chain length. LAMMPS default 3.
        int mpchain = 3;   ///< Baro-NHC chain length. LAMMPS default 3.

        /// @brief Per-axis barostat enable flags `{x, y, z}`.
        ///
        /// `true` ⇒ that box edge is barostatted toward `P_target`;
        /// `false` ⇒ that edge is held rigid (its `ε̇_α` stays 0 and it
        /// is never rescaled). Default `{true,true,true}` = the isotropic
        /// LAMMPS `fix npt iso` behaviour. Mirrors LAMMPS `p_flag[0..2]`.
        std::array<bool, 3> couple_axis = {true, true, true};

        /// @brief Final thermostat setpoint of the temperature ramp [K].
        ///
        /// `T_target` is treated as `T_start`. If `T_stop != T_target`,
        /// the live setpoint sweeps linearly from `T_target` to `T_stop`
        /// across the run span set by `set_run_span()`. Default
        /// `T_stop == T_target` ⇒ a constant-temperature run (no ramp),
        /// so existing aggregate-initialised callers are unaffected.
        /// Mirrors LAMMPS `fix npt temp T_start T_stop tau`.
        double T_stop = -1.0; ///< -1 sentinel: ctor folds to `T_target`.
    };

    /// @param cfg Aggregate NPT configuration. See `Config` for units.
    explicit NoseHooverNPT(Config cfg);

    /// @brief Set the temperature-ramp run span `[nfirst, nlast]` (in MD
    ///        step indices), mirroring LAMMPS `update->beginstep` /
    ///        `update->endstep` captured at run setup.
    ///
    /// Must be called before the driver's `run()` if a ramp
    /// (`T_stop != T_target`) is configured. With the default span
    /// (`nlast <= nfirst`) the ramp `delta` is pinned to 0, so the
    /// setpoint stays at `T_target` — a no-op, consistent with the
    /// `T_stop == T_target` default.
    ///
    /// @param nfirst First MD step index of the run.
    /// @param nlast  Last MD step index of the run.
    /// @return `*this`, so the call can chain before `set_integrator`.
    NoseHooverNPT& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }

    bool requires_system_stress() const override { return true; }
    std::string name() const override { return "nose_hoover_npt"; }
    /// @brief Role tag — Integrator. Although this component subsumes
    /// VelocityVerlet + NoseHoover + Barostat in one Trotter splitting,
    /// only one structural role is exposed; the integrator role is the
    /// load-bearing one for `cpu::md::MD::validate_components()`.
    /* role via Integrator base */
    /// @brief Write-set: MTK: propagates x and v, and scales the cell.
    /// @brief Reads the kinetic-stress component of `frame.stress` to
    /// compute the internal pressure feedback. Triggers auto-injection
    /// of `Virial` in `MD::install_defaults()` when no canonical
    /// stress provider is otherwise registered.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, -40)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    // Configuration
    double T_target_; ///< Ramp start temperature [K] (LAMMPS `t_start`).
    double T_stop_;   ///< Ramp end temperature [K] (LAMMPS `t_stop`).
    double tau_T_;
    double P_target_;
    double tau_P_;
    double dt_;
    double drag_;
    int mtchain_;
    int mpchain_;
    /// @brief Per-axis barostat enable flags `{x,y,z}` — LAMMPS `p_flag`.
    std::array<bool, 3> couple_axis_;
    /// @brief Count of `true` entries in `couple_axis_` — LAMMPS `pdim`.
    int pdim_ = 3;

    // Temperature-ramp run span (LAMMPS `update->beginstep` / `endstep`).
    std::int64_t nfirst_ = 0; ///< First MD step of the run.
    std::int64_t nlast_ = 0;  ///< Last MD step of the run.
    /// @brief Live thermostat setpoint [K], interpolated each step from
    /// the ramp; equals `T_target_` when no ramp is active.
    double T_live_ = 0.0;

    // Derived in build(): (1 - drag * half_dt / tau). `1.0` = no drag.
    double tdrag_factor_ = 1.0;
    double pdrag_factor_ = 1.0;

    // Derived in build()
    int n_dof_ = 0;           ///< 3N - 3 (COM removed)
    double kT_ = 0.0;         ///< kB · T_target [eV]
    std::vector<double> Q_t_; ///< Atom-NHC masses per chain link [eV·fs²]
    std::vector<double> Q_b_; ///< Baro-NHC masses per chain link [eV·fs²]
    double W_ = 0.0;          ///< Barostat mass [eV·fs²]

    // Extended-Lagrangian state (per-step)
    std::vector<double> eta_t_dot_; ///< Atom-NHC momenta [1/fs]
    std::vector<double> eta_t_;     ///< Atom-NHC positions
    std::vector<double> eta_b_dot_; ///< Baro-NHC momenta [1/fs]
    std::vector<double> eta_b_;     ///< Baro-NHC positions
    /// @brief Per-axis box log-length momenta `ε̇_{x,y,z}` [1/fs] —
    /// LAMMPS `omega_dot[0..2]`. An axis with `couple_axis_[α]==false`
    /// stays exactly 0. For the all-axes-coupled case all three entries
    /// are equal and the integrator reduces to the isotropic scheme.
    std::array<double, 3> eps_dot_ = {0.0, 0.0, 0.0};

    // Cached global kinetic energy carried between phases
    double KE_global_ = 0.0; ///< 0.5 · Σ m·v² (global, eV)

    // Helpers --------------------------------------------------------
    double compute_global_KE(const cpu::LaunchContext& ctx) const;
    /// @brief Per-axis kinetic energy `{KE_x, KE_y, KE_z}` [eV] from
    ///        the current velocities (MPI-reduced). Each `KE_α =
    ///        ½·Σ m·v_α²`; the trace `KE_x+KE_y+KE_z` equals the global
    ///        kinetic energy. Used by the anisotropic barostat so each
    ///        axis's driving pressure carries its own kinetic diagonal,
    ///        which is what keeps the per-axis Trotter split symplectic.
    std::array<double, 3> compute_axis_KE(const cpu::LaunchContext& ctx) const;
    /// @brief Interpolate the live thermostat setpoint for `step`,
    ///        mirroring LAMMPS `fix_nh.cpp::compute_temp_target()`.
    ///        Updates `T_live_` and `kT_`.
    void compute_temp_target(int step);
    void nhc_atoms_half(const cpu::LaunchContext& ctx, double half_dt);
    void nhc_baro_half(double half_dt);
    void update_eps_dot(const cpu::LaunchContext& ctx, double half_dt);
    void scale_v_by_baro(const cpu::LaunchContext& ctx, double half_dt);
    void kick_v_by_force(const cpu::LaunchContext& ctx, double half_dt);
    void propagate_x_and_box(const cpu::LaunchContext& ctx);
    /// @brief The conserved MTK extended Hamiltonian — written to
    /// `frame.conserved` each step so `Thermo` can log it and a flat
    /// trace can confirm the integrator is symplectic.
    double extended_hamiltonian(const cpu::LaunchContext& ctx) const;
};

} // namespace md
} // namespace cpu
} // namespace ATV
