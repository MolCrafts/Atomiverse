/// @file cuda/md/nose_hoover.cuh
/// @brief NoseHoover -- single-chain Nosé-Hoover thermostat for NVT.
///
/// Modifies velocities at initial_integrate and final_integrate (after VelocityVerlet).
/// Reference: Nosé (1984), Hoover (1985).
/// CUDA-free header.
#pragma once

#include <memory>
#include <string>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/facade/component_role.h"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Single-chain Nosé-Hoover thermostat.
///
/// Composes with VelocityVerlet (runs after it at same phases via order +100).
/// Maintains thermostat variable ξ on device. Rescales velocities to target temperature.
class NoseHoover : public Component {
  public:
    /// @brief Construct thermostat.
    /// @param T_target Target temperature [K].
    /// @param tau      Coupling time constant [fs].
    /// @param dt       Time step [fs] (must match integrator).
    NoseHoover(double T_target, double tau, double dt);
    ~NoseHoover() override;
    NoseHoover(NoseHoover&&) noexcept;
    NoseHoover& operator=(NoseHoover&&) noexcept;

    std::string name() const override { return "nose_hoover"; }
    Role role() const override { return 0; }

    /// @brief Structural role — the temperature controller.
    ///
    /// Mirrors `cpu::md::NoseHoover`.
    /// @return ComponentRole::Thermostat.
    /* role via Thermostat base */
    /// @brief Write-set: rescales velocities by the thermostat factor.

    std::vector<DefaultBinding> default_bindings() const override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    double T_target_, tau_, dt_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
