/// @file cpu/semiempirical/stages/nddo_stages.h
/// @brief NDDO SCF stage and convergence criterion for `nddo::Phase`.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/criterion.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief One NDDO SCF stage. The step body is supplied by `NddoMethod`.
class NddoStep : public cpu::Component {
  public:
    /// @brief Zero-argument body for this phase.
    using StepFn = std::function<void()>;

    /// @param phase Stage this object is bound to.
    /// @param component_name Component `name()`.
    /// @param step Body invoked when the launch is at `phase`.
    NddoStep(ATV::nddo::Phase phase, std::string component_name, StepFn step)
        : phase_(phase), name_(std::move(component_name)),
          step_(std::move(step)) {}

    /// @brief Component name.
    /// @return The name given at construction.
    std::string name() const override { return name_; }

    /// @brief Default binding: `phase` at order 0.
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(phase_, 0)};
    }

    /// @brief Run the step when `ctx` is at this phase.
    /// @param ctx Launch context.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(phase_))
            return;
        step_();
    }

  private:
    ATV::nddo::Phase phase_;
    std::string name_;
    StepFn step_;
};

/// @brief Judges the flag the check stage wrote. Does not recompute it.
class NddoConverged : public cpu::Criterion {
  public:
    /// @brief This criterion must be satisfied for the loop to converge.
    /// @return True.
    bool contributes_to_convergence() const override { return true; }

    /// @param converged Flag set by the check stage.
    explicit NddoConverged(const bool& converged) : converged_(converged) {}

    /// @brief Component identifier / stop reason.
    /// @return `"nddo_converged"`.
    std::string name() const override { return "nddo_converged"; }

    /// @brief Bind to `nddo::Phase::check`.
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::nddo::Phase::check, 0)};
    }

    /// @brief No-op measure. The check stage already decided the flag.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override {
        if (!ctx.at(ATV::nddo::Phase::check))
            return;
    }

    /// @brief Judge the check-stage flag.
    /// @param state Unused.
    /// @return `Converged` when the flag is set.
    ATV::Verdict verdict(const cpu::LoopState& state) override {
        (void)state;
        return converged_ ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

  private:
    const bool& converged_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
