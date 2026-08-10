/// @file cpu/planewave/stages/mix_stage.h
/// @brief MixStage — `pw::Phase::mix`: Pulay density mix.
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

/// @brief Density-mixing stage for the plane-wave Driver loop.
///
/// Restricted fixed-occupation path: `DensityBuilder::finish` (FFT ρ(G) +
/// Pulay). Unlike Gaussian SCF / GPW (H-space DIIS before diagonalize), PW
/// mixes **after** the density rebuild.
class MixStage : public cpu::Component {
  public:
    /// @brief Zero-arg step body for one density mix.
    using StepFn = std::function<void()>;

    /// @param step Path-local mix (typically `DensityBuilder::finish`).
    explicit MixStage(StepFn step) : step_(std::move(step)) {}

    /// @brief Component name.
    /// @return `"pw_mix_stage"`.
    std::string name() const override { return "pw_mix_stage"; }

    /// @brief Default binding: `pw::Phase::mix` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::pw::Phase::mix, 0)};
    }

    /// @brief Mix the density.
    /// @param ctx Launch context; self-guards on `pw::Phase::mix`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::pw::Phase::mix))
            return;
        step_();
    }

  private:
    StepFn step_;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
