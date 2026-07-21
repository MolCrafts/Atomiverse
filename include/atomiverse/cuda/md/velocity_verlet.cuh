/// @file cuda/md/velocity_verlet.cuh
/// @brief VelocityVerlet -- velocity-Verlet time integrator component.
///
/// Implements the velocity-Verlet algorithm for Newtonian dynamics:
///
/// Initial integrate (first half-step):
/// @f[ v(t + \tfrac{1}{2}\Delta t) = v(t) + \tfrac{1}{2}\Delta t \cdot a(t) @f]
/// @f[ x(t + \Delta t) = x(t) + \Delta t \cdot v(t + \tfrac{1}{2}\Delta t) @f]
///
/// Final integrate (second half-step):
/// @f[ v(t + \Delta t) = v(t + \tfrac{1}{2}\Delta t) + \tfrac{1}{2}\Delta t \cdot a(t + \Delta t) @f]
///
/// Auto-binds to initial_integrate and final_integrate phases.
/// CUDA-free header.
///
/// Reference: Verlet (1967). Computer "Experiments" on Classical Fluids.
/// Physical Review, 159(1), 98-103.
///
/// @see md::Phase::initial_integrate, md::Phase::final_integrate
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Velocity-Verlet integrator. Auto-binds to initial_integrate and final_integrate.
///
/// Dispatches to the appropriate kernel based on `ctx.stage`:
/// - `Phase::initial_integrate`: updates velocities (half-step) and positions.
/// - `Phase::final_integrate`: updates velocities (half-step).
class VelocityVerlet : public Component {
  public:
    /// @brief Construct with a time step.
    /// @param dt Time step [fs].
    explicit VelocityVerlet(double dt) : dt_(dt) {}

    /// @brief Component name.
    /// @return "velocity_verlet".
    std::string name() const override { return "velocity_verlet"; }
    Role role() const override { return 0; }

    /// @brief Structural role — the time-step integrator.
    ///
    /// Mirrors `cpu::md::VelocityVerlet`. `cuda::md::MD::validate_components()`
    /// requires ≥ 1 `Integrator`-role component at build time.
    /// @return ComponentRole::Integrator.
    /* role via Integrator base */
    /// @brief Write-set: half-kick (v) + drift (x).

    /// @brief Default bindings: initial_integrate (order 0) and final_integrate (order 0).
    /// @return Two DefaultBinding entries.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    /// @brief Per-step kernel launch (dispatches by phase).
    /// @param ctx Launch context with phase and step information.
    void launch(const LaunchContext& ctx) override;

  private:
    double dt_; ///< Time step [fs].
};

} // namespace md
} // namespace cuda
} // namespace ATV
