/// @file cuda/dynamics/brownian.cuh
/// @brief Overdamped Brownian dynamics integrator (CUDA).
///
///   r ← r + (D / kT) F Δt + √(2 D Δt) ξ,   ξ ~ N(0, 1)
///
/// Sibling of `cpu::md::Brownian`. The move runs at `initial_integrate`;
/// `final_integrate` is a no-op (overdamped, no half-kick). The device
/// kernel is `brownian_integrate` (`src/cuda/dynamics/brownian.cu`).
/// A CPU-only build has no kernel and `launch` throws that name.
#pragma once

#include "atomiverse/cuda/component/integrator.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Overdamped Brownian dynamics integrator.
///
/// Positions advance by the Einstein drift plus a tag-keyed Gaussian.
/// Velocities are left untouched. Orthorhombic cells with `length(0)`
/// below 1e5 Å are wrapped; a free sandbox is not.
class Brownian : public cuda::Integrator, public cuda::TimestepProvider {
  public:
    /// @brief Publishes the step to `cuda::md::MD` (timestep protocol).
    const cuda::TimestepProvider* as_timestep_provider() const override {
        return this;
    }
    /// @return The constructor `dt` [fs].
    double timestep() const override { return dt_; }
    /// @brief Construct the integrator.
    /// @param dt Timestep [fs]. Must be > 0.
    /// @param D Diffusion constant [Å²/fs]. Must be > 0.
    /// @param temperature Temperature [K] for the drift term. Must be > 0.
    /// @param seed RNG seed. Mixed with the step and the atom tag.
    Brownian(double dt, double D, double temperature = 300.0,
             std::uint64_t seed = 1)
        : dt_(dt), D_(D), temperature_(temperature), seed_(seed) {
        if (!(dt > 0.0))
            throw std::invalid_argument("cuda::md::Brownian: dt must be > 0");
        if (!(D > 0.0))
            throw std::invalid_argument("cuda::md::Brownian: D must be > 0");
        if (!(temperature > 0.0))
            throw std::invalid_argument(
                "cuda::md::Brownian: temperature must be > 0");
    }

    /// @brief Component name.
    /// @return "brownian".
    std::string name() const override { return "brownian"; }
    /// @brief Integrator role. Zero means required and free to write positions.
    /// @return 0.
    Role role() const override { return 0; }

    /// @brief Published timestep [fs].
    /// @return The constructor `dt`.
    double dt() const { return dt_; }
    /// @brief Diffusion constant [Å²/fs].
    /// @return The constructor `D`.
    double D() const { return D_; }

    /// @brief Bind the move at `initial_integrate` and a no-op at
    ///        `final_integrate`, both at order 0.
    /// @return Two `DefaultBinding` entries, order 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, 0)};
    }

    /// @brief Advance positions on the device, or throw if the kernel
    ///        is not in this build.
    /// @param ctx Launch context. Only `initial_integrate` does work.
    void launch(const LaunchContext& ctx) override;

  private:
    double dt_;
    double D_;
    double temperature_;
    std::uint64_t seed_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
