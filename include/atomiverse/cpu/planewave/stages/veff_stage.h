/// @file cpu/planewave/stages/veff_stage.h
/// @brief VeffStage — `pw::Phase::veff`: Poisson + XC + Veff from ρ.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Effective-potential build stage for the plane-wave Driver loop.
///
/// Invokes a path-local step (restricted: Poisson → XC → Veff) supplied by
/// the method at wire time. Self-guards on `pw::Phase::veff`.
class VeffStage : public cpu::Component {
  public:
    /// @brief Zero-arg step body for one Veff assembly.
    using StepFn = std::function<void()>;

    /// @param step Path-local potential assembly.
    explicit VeffStage(StepFn step) : step_(std::move(step)) {}

    /// @brief Component name.
    /// @return `"pw_veff_stage"`.
    std::string name() const override { return "pw_veff_stage"; }

    /// @brief Default binding: `pw::Phase::veff` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::pw::Phase::veff, 0)};
    }

    /// @brief Build Veff from the current density.
    /// @param ctx Launch context; self-guards on `pw::Phase::veff`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::pw::Phase::veff))
            return;
        step_();
    }

  private:
    StepFn step_;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
