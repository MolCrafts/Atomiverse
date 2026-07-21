/// @file cpu/component/zero.h
/// @brief Phase-agnostic accumulator-zeroing components: ZeroForces,
///        ZeroEnergy, ZeroStress, ZeroKinetic.
///
/// Shared backend-root infrastructure — every category driver binds these
/// to its own phases (`cpu::md::MD` at `pre_force`, `cpu::minimizer::Minimize`
/// and `cpu::Evaluate` at their force/prepare phases). They carry no
/// `default_bindings()` precisely because the binding phase is the driver's
/// decision, not the component's.
#pragma once

#include <string>

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Zero all owned-row forces. Phase-agnostic — driver binds to its
/// own phase. Ghost rows are zeroed by `HaloSend` when it (re)populates them.
class ZeroForces : public cpu::Component {
  public:
    std::string name() const override { return "zero_forces"; }
    void launch(const cpu::LaunchContext& ctx) override;
};

/// @brief Zero potential energy (total + per-term decomposition).
/// Phase-agnostic — driver binds to its own phase.
class ZeroEnergy : public cpu::Component {
  public:
    std::string name() const override { return "zero_energy"; }
    void launch(const cpu::LaunchContext& ctx) override;
};

/// @brief Zero stress tensor. Phase-agnostic — driver binds to its own phase.
///
/// Stress accumulation by force kernels (LJ pair virial, future bonded
/// virials) requires a clean baseline at the start of each step.
class ZeroStress : public cpu::Component {
  public:
    std::string name() const override { return "zero_stress"; }
    void launch(const cpu::LaunchContext& ctx) override;
};

/// @brief Zero kinetic energy. Phase-agnostic — driver binds to its own phase.
class ZeroKinetic : public cpu::Component {
  public:
    std::string name() const override { return "zero_kinetic"; }
    void launch(const cpu::LaunchContext& ctx) override;
};

} // namespace cpu
} // namespace ATV
