/// @file cpu/planewave/stages/pw_converged.h
/// @brief PwConverged — Criterion at `pw::Phase::check` for plane-wave SCF.
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/criterion.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Plane-wave convergence Criterion — shipping ‖Δρ‖₂ residual gate.
///
/// Does not re-measure: the path-local mix / residual step already wrote
/// `converged`. This Criterion only **judges** that flag so the Driver can
/// stop. Algorithms still never write `IterationStatus` themselves.
class PwConverged : public cpu::Criterion {
  public:
    /// @param converged Flag set when the density residual is below tol.
    explicit PwConverged(const bool& converged) : converged_(converged) {}

    /// @brief Component identifier / stop reason.
    /// @return `"pw_converged"`.
    std::string name() const override { return "pw_converged"; }

    /// @brief Bind to `pw::Phase::check` (measurement is a no-op).
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::pw::Phase::check, 0)};
    }

    /// @brief No-op measure (gate already decided in the residual step).
    /// @param ctx Launch context; self-guards on `pw::Phase::check`.
    void launch(const LaunchContext& ctx) override {
        if (!ctx.at(ATV::pw::Phase::check))
            return;
    }

    /// @brief Judge the residual gate.
    /// @param state Unused.
    /// @return `Converged` when the residual flag is set.
    ATV::Verdict verdict(const cpu::LoopState& state) override {
        (void)state;
        if (std::getenv("ATV_PW_NOSTOP") != nullptr)
            return ATV::Verdict::Continue;
        return converged_ ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

    /// @brief Reset (nothing owned beyond the external flag).
    void reset() {}

  private:
    const bool& converged_;
};

} // namespace pw
} // namespace cpu
} // namespace ATV
