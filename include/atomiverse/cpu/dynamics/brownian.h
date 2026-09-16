/// @file cpu/dynamics/brownian.h
/// @brief Overdamped Brownian dynamics Integrator (CPU).
///
///   r ← r + (D / kT) F Δt + √(2 D Δt) ξ,  ξ ~ N(0,1)
///
/// Free particle (F=0): Einstein ⟨r²⟩ = 6 D t in 3D.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/timestep_protocol.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Overdamped Brownian dynamics integrator.
class Brownian : public cpu::Integrator, public cpu::TimestepProvider {
  public:
    /// @param dt Timestep [fs].
    /// @param D Diffusion constant [Å²/fs].
    /// @param temperature Temperature [K] (for force drift term).
    /// @param seed RNG seed.
    Brownian(double dt, double D, double temperature = 300.0,
             std::uint64_t seed = 1);

    std::string name() const override { return "brownian"; }
    const cpu::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @brief Published nuclear timestep [fs] (the constructor `dt`).
    double timestep() const override { return dt_; }

    std::vector<DefaultBinding> default_bindings() const override {
        // Single-phase move at initial_integrate; final_integrate is no-op.
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    double D() const { return D_; }
    double dt() const { return dt_; }

  private:
    double dt_;
    double D_;
    double temperature_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};

} // namespace md
} // namespace cpu
} // namespace ATV
