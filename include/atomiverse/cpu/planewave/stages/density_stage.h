/// @file cpu/planewave/stages/density_stage.h
/// @brief DensityStage — `pw::Phase::density`: pre-mix density bookkeeping.
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

/// @brief Density stage for the plane-wave Driver loop.
///
/// On the restricted fixed-occupation path the real-space accumulation is
/// already complete after `DiagonalizeStage`; this stage is the hook for
/// pre-mix bookkeeping (energy terms from the new bands, etc.) before
/// `MixStage` runs Pulay. Path-local step is supplied at wire time.
class DensityStage : public cpu::Component {
  public:
    /// @brief Zero-arg step body for one density bookkeeping pass.
    using StepFn = std::function<void()>;

    /// @param step Path-local density bookkeeping (may be a no-op).
    explicit DensityStage(StepFn step) : step_(std::move(step)) {}

    /// @brief Component name.
    /// @return `"pw_density_stage"`.
    std::string name() const override { return "pw_density_stage"; }

    /// @brief Default binding: `pw::Phase::density` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::pw::Phase::density, 0)};
    }

    /// @brief Run density bookkeeping.
    /// @param ctx Launch context; self-guards on `pw::Phase::density`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::pw::Phase::density))
            return;
        step_();
    }

  private:
    StepFn step_;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
