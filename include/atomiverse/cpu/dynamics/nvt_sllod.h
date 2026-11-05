/// @file cpu/dynamics/nvt_sllod.h
/// @brief NVT SLLOD integrator: LAMMPS `fix nvt/sllod` + `fix deform` ERATE.
///
/// Replaces `VelocityVerlet`. The thermostat is inside the step, so do not
/// add `NoseHoover` or `Langevin` beside it. `remap v` lives here; the
/// existing `Deform` class stays `remap x`.
#pragma once

#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Reversible NVT SLLOD with one ERATE direction and `remap v`.
///
/// `temperature_K` is the thermostat target, `tau_fs` is LAMMPS `Tdamp`
/// (100 fs in the mechanical protocol), `dt_fs` is the timestep, and
/// `rate_per_fs` is the engineering rate. `direction` is the Voigt index
/// 0..5 (`x y z yz xz xy`). `tchain` is the Nosé–Hoover chain length
/// (LAMMPS `tchain`; `fix nvt/sllod` defaults it to 1, `fix_nvt_sllod.cpp`).
///
/// The step follows LAMMPS `FixNH` + `FixNVTSllod` + `FixDeform`:
/// `initial_integrate` thermostats with the peculiar KE carried from the
/// previous `final_integrate` (fresh only on the first step after `build()`,
/// as `FixNH::setup`), kicks and moves; `final_integrate` kicks, measures
/// the peculiar KE and thermostats; `end_of_step` deforms the cell and
/// decides a box flip when a tilt passes half its box length (`flip yes`).
/// A decided flip forces the next step to reneighbor; its `pre_exchange`
/// installs the flipped tilts, flips the image flags and remaps. Owned atoms
/// are wrapped (with the `remap v` velocity jump) only at `pre_exchange`,
/// i.e. on reneighbor steps (LAMMPS `Domain::pbc` before `Comm::exchange`),
/// so the Verlet displacement check sees continuous coordinates.
///
/// The strain clock is an internal counter, snapshotted from the live cell
/// on the first launch after `build()`. It is not `LaunchContext::step`.
class NvtSllod : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    /// @param temperature_K Target temperature [K].
    /// @param tau_fs Thermostat damping time (LAMMPS `Tdamp`) [fs].
    /// @param dt_fs Timestep [fs].
    /// @param rate_per_fs Engineering strain rate [1/fs].
    /// @param direction Voigt index 0..5: x y z yz xz xy.
    /// @param tchain Nosé–Hoover chain length (>= 1; LAMMPS `tchain`).
    NvtSllod(double temperature_K, double tau_fs, double dt_fs,
             double rate_per_fs, int direction, int tchain = 1);

    bool supports_input_changes(InputChange) const override { return true; }
    std::string name() const override { return "nvt_sllod"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    double timestep() const override { return dt_; }
    bool requires_system_stress() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::pre_exchange, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Current Nosé–Hoover chain (LAMMPS `eta`, `eta_dot`).
    /// @return Chain positions [1] and velocities [1/fs], innermost first.
    ATV::md::SllodChainState thermostat_chain() const {
        return {eta_, eta_dot_};
    }

  private:
    void capture_(const cpu::LaunchContext& ctx);
    void thermostat_(const cpu::LaunchContext& ctx, bool measure);
    double peculiar_ke_(const cpu::LaunchContext& ctx) const;
    void nve_v_(const cpu::LaunchContext& ctx);
    void nve_x_(const cpu::LaunchContext& ctx);
    void pre_exchange_(const cpu::LaunchContext& ctx);
    void end_of_step_(const cpu::LaunchContext& ctx);

    double T_;
    double tau_;
    double dt_;
    double rate_;
    int dir_;
    int tchain_;
    int n_dof_ = 0;
    bool armed_ = false;
    bool kicked_ = false;
    long long step_ = 0;
    ATV::md::SllodFlow flow_{};
    std::vector<double> eta_;      ///< Chain positions (LAMMPS `eta`).
    std::vector<double> eta_dot_;  ///< Chain velocities [1/fs].
    std::vector<double> eta_mass_; ///< Chain masses [eV fs^2].
    /// Peculiar KE [eV] carried between half-steps (LAMMPS `t_current`).
    double ke_current_ = 0;
    bool ke_valid_ = false;
};

} // namespace md
} // namespace cpu
} // namespace ATV
