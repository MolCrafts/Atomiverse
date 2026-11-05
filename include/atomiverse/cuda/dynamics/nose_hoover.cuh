/// @file cuda/dynamics/nose_hoover.cuh
/// @brief NoseHoover -- single-chain Nosé-Hoover thermostat for NVT.
///
/// Modifies velocities at initial_integrate and final_integrate (after VelocityVerlet).
/// Reference: Nosé (1984), Hoover (1985).
/// CUDA-free header.
#pragma once

#include "atomiverse/cuda/component/thermostat.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <memory>
#include <string>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Single-chain Nosé-Hoover thermostat.
///
/// Composes with VelocityVerlet (runs after it at same phases via order +100).
/// Maintains thermostat variable ξ on device. Rescales velocities to target
/// temperature. Same law as `cpu::md::NoseHoover` (one scalar ξ,
/// `Q = N_f k_B T τ²`, half-step update `ξ += dt/2 (2KE − N_f k_B T)/Q`,
/// velocities × `exp(−ξ dt/2)`), and like it a `TimestepConsumer`: the
/// integrator's dt replaces the constructor dt before `build()`. A system
/// with no thermal degrees of freedom is left untouched.
class NoseHoover : public cuda::Thermostat, public cuda::TimestepConsumer {
  public:
    /// @brief Construct thermostat.
    /// @param T_target Target temperature [K].
    /// @param tau      Coupling time constant [fs].
    /// @param dt       Time step [fs]; replaced by the integrator's dt.
    /// @throws std::invalid_argument unless T_target, tau and dt are finite
    ///         and > 0.
    NoseHoover(double T_target, double tau, double dt);

    /// @brief Consume the enclosing integrator's timestep [fs].
    cuda::TimestepConsumer* as_timestep_consumer() override { return this; }
    /// @brief The constructor dt serves a run without an integrator clock.
    bool timestep_required() const override { return false; }
    /// @brief Receive the integrator timestep [fs].
    /// @throws std::invalid_argument when `dt_fs` is not finite and > 0.
    void set_timestep(double dt_fs) override;
    ~NoseHoover() override;
    NoseHoover(NoseHoover&&) noexcept;
    NoseHoover& operator=(NoseHoover&&) noexcept;

    std::string name() const override { return "nose_hoover"; }
    bool graph_capturable() const override { return true; }
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
