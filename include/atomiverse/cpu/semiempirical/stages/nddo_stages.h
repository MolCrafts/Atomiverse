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
    bool supports_input_changes(InputChange) const override { return true; }
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

/// @brief Judges the density and energy residuals `scf_check` published.
///
/// Both residuals must be inside their tolerances (`<=`). The method accepts
/// a warm-start density only after its complete requested evaluation succeeds.
/// This class does not recompute the residuals.
class NddoConverged : public cpu::Criterion {
  public:
    bool supports_input_changes(InputChange) const override { return true; }
    /// @brief This criterion must be satisfied for the loop to converge.
    /// @return True.
    bool contributes_to_convergence() const override { return true; }

    /// @param density_residual Max |ΔP| of this iteration.
    /// @param energy_residual |ΔE| of this iteration [eV].
    /// @param density_tol Density tolerance. Inclusive.
    /// @param energy_tol Energy tolerance [eV]. Inclusive.
    NddoConverged(const double& density_residual, const double& energy_residual,
                  double density_tol, double energy_tol)
        : density_residual_(density_residual),
          energy_residual_(energy_residual), density_tol_(density_tol),
          energy_tol_(energy_tol) {}

    /// @brief Update the numeric controls without rebinding the residuals.
    /// @param density_tol Density tolerance (dimensionless), validated by the solver.
    /// @param energy_tol Energy tolerance [eV], validated by the solver.
    void set_tolerances(double density_tol, double energy_tol) {
        density_tol_ = density_tol;
        energy_tol_ = energy_tol;
    }

    /// @brief Component identifier / stop reason.
    /// @return `"nddo_converged"`.
    std::string name() const override { return "nddo_converged"; }

    /// @brief Bind to `nddo::Phase::check`.
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::nddo::Phase::check, 0)};
    }

    /// @brief No-op measure. `scf_check` already published the residuals.
    /// @param ctx Launch context.
    void launch(const LaunchContext& ctx) override {
        if (!ctx.at(ATV::nddo::Phase::check))
            return;
    }

    /// @brief Judge the published residuals.
    /// @param state Unused.
    /// @return `Converged` when both residuals are inside their tolerances.
    ATV::Verdict verdict(const cpu::LoopState& state) override {
        (void)state;
        const bool met = density_residual_ <= density_tol_
                         && energy_residual_ <= energy_tol_;
        return met ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

  private:
    const double& density_residual_;
    const double& energy_residual_;
    double density_tol_;
    double energy_tol_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
