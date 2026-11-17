/// @file cpu/dynamics/fix_csvr.h
/// @brief CPU CSVR (Bussi) stochastic-velocity-rescaling thermostat —
///        LAMMPS `fix temp/csvr`, optionally on a type group and with the
///        `temp/com` bias.
///
/// Port reference: LAMMPS `src/EXTRA-FIX/fix_temp_csvr.cpp` and
/// `src/compute_temp_com.cpp`, e8c4c278a8.
/// Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
/// the U.S. Government retains certain rights. GPL-2.0-only.
#pragma once

#include "atomiverse/cpu/component/timestep_protocol.h"
#include <cmath>
#include <stdexcept>

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/csvr.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/thermostat.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief CSVR thermostat — canonical sampling through velocity rescaling.
///
/// Bussi, Donadio, Parrinello, "Canonical sampling through velocity
/// rescaling", J. Chem. Phys. **126**, 014101 (2007), DOI 10.1063/1.2408420.
/// CP2K's `&THERMOSTAT TYPE CSVR`; LAMMPS `fix temp/csvr`.
///
/// Once per MD step (end_of_step) the kinetic energy `K` of the thermostat
/// group is stochastically driven toward `K̄ = ½ N_f k_B T` and the group's
/// velocities are rescaled by one factor `α` (Bussi Eq. A7, `core/csvr.h`).
/// The two per-step randoms come from the counter-keyed stream of
/// `core/csvr.h`, seeded from `(seed, step)` only — identical on every MPI
/// rank and to `cuda::md::FixCSVR`.
///
/// **Group.** `types` selects the atoms whose force-field type
/// (`kAtom_type`) is listed; empty = every atom. `N_f = 3 N_g − 3 −
/// constraints owned by group atoms` (LAMMPS `compute temp` /
/// `compute temp/com` `dof_compute` with `extra_dof = 3` and
/// `FixShake::dof(igroup)`); for the whole system this is the driver's
/// `n_dof_global`.
///
/// **Bias.** `TemperatureBias::Com` measures `K` from velocities relative
/// to the group's centre-of-mass velocity and rescales only that peculiar
/// part, `v ← v_cm + α (v − v_cm)`, so the group's momentum is untouched
/// (LAMMPS `fix_modify <fix> temp <compute temp/com>`: `remove_bias`, scale,
/// `restore_bias`). Several instances (e.g. one per species, each with its
/// own seed) may be registered on one driver.
///
/// **Bookkeeping.** Each step adds `−ΔK` of its rescale to the Frame's
/// persistent `kThermostatEnergy` (Bussi Eq. 15) and publishes
/// `kConserved = kEnergy + K_total + kThermostatEnergy`, `K_total` the whole
/// system's kinetic energy after this rescale; with several instances the
/// last one writes the final value. `kKinetic` stays the
/// `KineticEnergy` (end_of_step −100) value, as for every end-of-step
/// thermostat.
///
/// Supports the `T_start`→`T_stop` linear ramp of `cpu::md::NoseHoover`
/// via `set_T_stop` + `set_run_span`.
class FixCSVR : public cpu::Thermostat, public cpu::TimestepConsumer {
  public:
    cpu::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return false; }
    void set_timestep(double dt) override {
        if (!std::isfinite(dt) || dt <= 0.0)
            throw std::invalid_argument(
                "FixCSVR: timestep must be finite and positive");
        dt_ = dt;
    }

    /// @brief Whole-system thermostat, no bias.
    /// @param T_start Target / ramp-start temperature [K].
    /// @param tau     Coupling time constant [fs] (CP2K `TIMECON`, LAMMPS
    ///                `Tdamp`).
    /// @param dt      Timestep [fs].
    /// @param seed    RNG seed; reproducible across rank counts.
    FixCSVR(double T_start, double tau, double dt, std::uint32_t seed);

    /// @brief Group thermostat.
    /// @param types Force-field types of the group (`kAtom_type`); empty =
    ///              every atom.
    /// @param bias  Velocity bias removed before measuring/rescaling.
    /// @throws std::invalid_argument for a negative type, a non-finite or
    ///         non-positive T_start / tau / dt.
    FixCSVR(double T_start, double tau, double dt, std::uint32_t seed,
            std::vector<int> types, ATV::md::TemperatureBias bias);

    std::string name() const override { return "fix_csvr"; }
    /// @brief Applied once per step, after the velocity-Verlet step.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +100)};
    }
    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Set the ramp end temperature [K]; enables the linear ramp
    ///        over the span set by `set_run_span`. Default `= T_start`.
    /// @return `*this`, for chaining before registration.
    FixCSVR& set_T_stop(double T_stop) {
        T_stop_ = T_stop;
        return *this;
    }
    /// @brief Set the ramp run span `[nfirst, nlast]` in MD step indices.
    /// @return `*this`, for chaining before registration.
    FixCSVR& set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        nfirst_ = nfirst;
        nlast_ = nlast;
        return *this;
    }
    /// @return Live thermostat setpoint [K] — for diagnostics / observers.
    double T_target() const noexcept { return T_live_; }
    /// @brief This instance's accumulated bath energy `−Σ ΔK` [eV].
    double thermostat_energy() const noexcept { return thermostat_energy_; }
    /// @return Group degrees of freedom `N_f` (after `build`).
    std::int64_t n_dof() const noexcept { return n_dof_; }
    /// @return Group force-field types (empty = every atom).
    const std::vector<int>& types() const noexcept { return types_; }
    /// @return Velocity bias.
    ATV::md::TemperatureBias bias() const noexcept { return bias_; }

  private:
    bool in_group_(int type) const {
        return types_.empty()
               || (type >= 0 && static_cast<std::size_t>(type) < member_.size()
                   && member_[static_cast<std::size_t>(type)] != 0);
    }

    double T_start_;     ///< Ramp start / target temperature [K].
    double T_stop_;      ///< Ramp end temperature [K].
    double tau_;         ///< Coupling time constant [fs].
    double dt_;          ///< Timestep [fs].
    std::uint32_t seed_; ///< RNG seed (rank-count invariant).
    std::vector<int> types_;
    std::vector<char> member_; ///< member_[type] != 0 ⇔ type in group.
    ATV::md::TemperatureBias bias_ = ATV::md::TemperatureBias::None;
    std::int64_t n_dof_ = 0;       ///< Group N_f, fixed at build().
    double thermostat_energy_ = 0; ///< This instance's −Σ ΔK [eV].

    std::int64_t nfirst_ = 0; ///< First MD step of the run (ramp span).
    std::int64_t nlast_ = 0;  ///< Last MD step of the run (ramp span).
    double T_live_ = 0.0;     ///< Live setpoint [K], interpolated each step.
};

} // namespace md
} // namespace cpu
} // namespace ATV
