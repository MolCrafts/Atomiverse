/// @file cpu/gpw/stages/gpw_converged.h
/// @brief GpwConverged — Criterion at `gpw::Phase::check` for GPW SCF.
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/criterion.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief GPW convergence Criterion — shipping `|ΔE| ∧ max|err|` gate.
///
/// Does not re-measure: KsBuildStage already applied the gate to
/// `skip_update`. This Criterion only **judges** that flag so the Driver
/// can stop. Algorithms still never write `IterationStatus` themselves.
class GpwConverged : public cpu::Criterion {
  public:
    /// @param skip_update Flag set by KsBuildStage when the residual gate holds.
    explicit GpwConverged(const bool& skip_update)
        : skip_update_(skip_update) {}

    /// @brief Component identifier / stop reason.
    /// @return `"gpw_converged"`.
    std::string name() const override { return "gpw_converged"; }

    /// @brief Bind to `gpw::Phase::check` (measurement is a no-op).
    /// @return Single binding.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::check, 0)};
    }

    /// @brief No-op measure (gate already decided in KsBuildStage).
    /// @param ctx Launch context; self-guards on `gpw::Phase::check`.
    void launch(const LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::check))
            return;
    }

    /// @brief Judge the KsBuild residual gate.
    /// @param state Unused.
    /// @return `Converged` when `skip_update` is set (and NOSTOP unset).
    ATV::Verdict verdict(const cpu::LoopState& state) override {
        (void)state;
        if (std::getenv("ATV_GPW_NOSTOP") != nullptr)
            return ATV::Verdict::Continue;
        return skip_update_ ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

    /// @brief Reset (nothing owned beyond the external flag).
    void reset() {}

  private:
    const bool& skip_update_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
