/// @file cuda/dynamics/barostat.cuh
/// @brief Barostat -- Berendsen-style isotropic pressure coupling.
///
/// Rescales box and positions: μ = (1 - β*dt/τ_p * (P_target - P))^(1/3).
/// Reads stress tensor from DeviceFrame::stress (requires Virial to run first).
/// CUDA-free header.
#pragma once

#include "atomiverse/cuda/component/barostat.cuh"

#include <memory>
#include <string>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Berendsen isotropic barostat.
///
/// Binds to final_integrate at order +200 (after VelocityVerlet + NoseHoover).
/// Requires Virial at post_force to populate DeviceFrame::stress.
class Barostat : public cuda::Barostat {
  public:
    /// @brief Construct barostat.
    /// @param P_target Target pressure [eV/Angstrom^3].
    /// @param tau_p    Pressure coupling time [fs].
    /// @param dt       Time step [fs].
    Barostat(double P_target, double tau_p, double dt);
    ~Barostat() override;
    Barostat(Barostat&&) noexcept;
    Barostat& operator=(Barostat&&) noexcept;

    std::string name() const override { return "barostat"; }
    Role role() const override { return 0; }
    bool requires_system_stress() const override { return true; }

    /// @brief Structural role — the pressure controller.
    ///
    /// Mirrors `cpu::md::Barostat`.
    /// @return ComponentRole::Barostat.
    /* role via Barostat base */
    /// @brief Reads — consumes `Stress` to drive the Berendsen volume
    ///        rescale. Triggers `cuda::md::MD::install_infra_` to
    ///        auto-inject `cuda::Virial` via capability matching
    ///        (the `role()==Barostat` proxy is no longer required).

    std::vector<DefaultBinding> default_bindings() const override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    double P_target_, tau_p_, dt_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
