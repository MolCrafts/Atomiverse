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
/// 0..5 (`x y z yz xz xy`).
///
/// The strain clock is an internal counter, snapshotted from the live cell
/// on the first launch after `build()`. It is not `LaunchContext::step`.
class NvtSllod : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    NvtSllod(double temperature_K, double tau_fs, double dt_fs,
             double rate_per_fs, int direction);

    bool supports_input_changes(InputChange) const override { return true; }
    std::string name() const override { return "nvt_sllod"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    double timestep() const override { return dt_; }
    bool requires_system_stress() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0),
                ATV::make_binding(ATV::md::Phase::end_of_step, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    void capture_(const cpu::LaunchContext& ctx);
    void thermostat_(const cpu::LaunchContext& ctx);
    void nve_v_(const cpu::LaunchContext& ctx);
    void nve_x_(const cpu::LaunchContext& ctx);
    void end_of_step_(const cpu::LaunchContext& ctx);

    double T_;
    double tau_;
    double dt_;
    double rate_;
    int dir_;
    int n_dof_ = 0;
    bool armed_ = false;
    bool kicked_ = false;
    long long step_ = 0;
    ATV::md::SllodFlow flow_{};
    ATV::md::SllodChain chain_{};
};

} // namespace md
} // namespace cpu
} // namespace ATV
