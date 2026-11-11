/// @file cpu/semiempirical/internal/scc_stages.h
/// @brief SCC stage callbacks and charge-residual Criterion shared by method loops.
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/criterion.h"

namespace ATV::cpu::semiempirical {
namespace internal {

/// @brief One SCC stage. The body is the caller's physics for that phase.
class SccStep : public cpu::Component {
  public:
    SccStep(ATV::scc::Phase phase, std::string component_name,
            std::function<void()> step)
        : phase_(phase), name_(std::move(component_name)),
          step_(std::move(step)) {}

    std::string name() const override { return name_; }

    void set_step(std::function<void()> step) { step_ = std::move(step); }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(phase_))
            return;
        if (step_)
            step_();
    }

  private:
    ATV::scc::Phase phase_;
    std::string name_;
    std::function<void()> step_;
};

/// @brief Charge-residual requirement. Does not recompute the residual.
class SccResidual : public cpu::Criterion {
  public:
    SccResidual(const double& residual, const double& tol)
        : residual_(residual), tol_(tol) {}

    bool contributes_to_convergence() const override { return true; }
    std::string name() const override { return "scc_residual"; }
    void launch(const cpu::LaunchContext&) override {}

    ATV::Verdict verdict(const cpu::LoopState&) override {
        return accepts() ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

    bool accepts() const {
        if (!std::isfinite(residual_) || residual_ < 0.0)
            throw std::runtime_error(
                "SCC: invalid or unpublished charge residual");
        return residual_ < tol_;
    }

  private:
    const double& residual_;
    const double& tol_;
};

} // namespace internal
} // namespace ATV::cpu::semiempirical
