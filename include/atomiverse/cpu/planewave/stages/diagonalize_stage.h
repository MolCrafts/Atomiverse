/// @file cpu/planewave/stages/diagonalize_stage.h
/// @brief DiagonalizeStage — `pw::Phase::diagonalize`: Davidson + accumulate.
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

/// @brief Band diagonalization stage for the plane-wave Driver loop.
///
/// Path-local step runs per-k Davidson and accumulates the real-space
/// density (begin + accumulate + MPI allreduce of the partial density).
class DiagonalizeStage : public cpu::Component {
  public:
    /// @brief Zero-arg step body for one diagonalize / accumulate pass.
    using StepFn = std::function<void()>;

    /// @param step Path-local Davidson + density accumulate.
    explicit DiagonalizeStage(StepFn step) : step_(std::move(step)) {}

    /// @brief Component name.
    /// @return `"pw_diagonalize_stage"`.
    std::string name() const override { return "pw_diagonalize_stage"; }

    /// @brief Default binding: `pw::Phase::diagonalize` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::pw::Phase::diagonalize, 0)};
    }

    /// @brief Diagonalize and accumulate density.
    /// @param ctx Launch context; self-guards on `pw::Phase::diagonalize`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::pw::Phase::diagonalize))
            return;
        step_();
    }

  private:
    StepFn step_;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
