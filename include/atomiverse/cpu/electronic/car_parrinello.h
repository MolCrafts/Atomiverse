/// @file cpu/electronic/car_parrinello.h
#include <cmath>
/// @brief CarParrinello — force-slot evaluator: fictitious-mass CP on a GPW method.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Optional Nosé–Hoover control of fictitious electronic kinetic energy.
struct ElectronicThermostat {
    double ekin_target_ha = 1e-5; ///< Target \(K_0\) [Hartree].
    double tau_e_fs = 50.0;       ///< Coupling time \(\tau_e\) [fs].
};

/// @brief Car–Parrinello MD evaluator: owns a pure `GPWEvaluator` + \(\mu\).
///
/// All CP parameters are constructor arguments. There is no `method.cpmd =
/// true` public switch. Register with `sim.set_evaluator(std::move(cp))`.
class CarParrinello : public cpu::GPW, public cpu::TimestepConsumer {
  public:
    /// @brief Construct with fictitious mass; electronic NVE (no e-thermostat).
    /// @param method Pure GPW single-point/BOMD evaluator (moved in).
    /// @param mu_au Fictitious mass \(\mu\) [a.u. electron mass]; must be > 0.
    CarParrinello(gpw::GPWEvaluator method, double mu_au)
        : method_(std::move(method)) {
        if (!(mu_au > 0.0) || !std::isfinite(mu_au))
            throw std::invalid_argument(
                "CarParrinello: mu_au must be finite and > 0");
        method_.cpmd = true;
        method_.cpmd_mu = mu_au;
        method_.cpmd_thermostat = false;
        method_.compute_forces = true;
    }

    /// @brief Construct with fictitious mass + electronic Nosé–Hoover.
    CarParrinello(gpw::GPWEvaluator method, double mu_au,
                  ElectronicThermostat th)
        : CarParrinello(std::move(method), mu_au) {
        if (!(th.ekin_target_ha > 0.0) || !std::isfinite(th.ekin_target_ha))
            throw std::invalid_argument(
                "CarParrinello: ekin_target_ha must be finite and > 0");
        if (!(th.tau_e_fs > 0.0) || !std::isfinite(th.tau_e_fs))
            throw std::invalid_argument(
                "CarParrinello: tau_e_fs must be finite and > 0");
        method_.cpmd_thermostat = true;
        method_.cpmd_ekin_target_ha = th.ekin_target_ha;
        method_.cpmd_tau_e = th.tau_e_fs;
    }

    std::string name() const override { return "car_parrinello"; }

    cpu::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return true; }
    void set_timestep(double dt_fs) override { method_.set_timestep(dt_fs); }

    void build(const cpu::BuildContext& ctx) override { method_.build(ctx); }
    void launch(const cpu::LaunchContext& ctx) override { method_.launch(ctx); }

    gpw::GPWEvaluator& method() { return method_; }
    const gpw::GPWEvaluator& method() const { return method_; }

    double last_energy_hartree() const { return method_.last_energy_hartree(); }
    double last_electronic_ke_ev() const {
        return method_.last_electronic_ke_ev();
    }
    double last_electronic_thermostat_ev() const {
        return method_.last_electronic_thermostat_ev();
    }
    const gpw::GpwResult& last_result() const { return method_.last_result(); }

  private:
    gpw::GPWEvaluator method_;
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
