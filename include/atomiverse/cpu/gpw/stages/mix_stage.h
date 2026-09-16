/// @file cpu/gpw/stages/mix_stage.h
/// @brief MixStage — `gpw::Phase::mix`: Pulay DIIS / path-local mix.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Mix stage for the GPW Driver loop (path-local Op).
///
/// When `skip_update` is set, the Op is not called (break-before-mix: raw H
/// from KsBuild is kept for store). Otherwise the path Op runs Pulay DIIS or
/// linear-mix bookkeeping.
class MixStage : public cpu::Component {
  public:
    /// @brief Path-local mix body (DIIS step or pass-through).
    using Op = std::function<void()>;

    /// @param op           Path-local mix.
    /// @param skip_update  When true, leave mixed H unchanged (converged break).
    MixStage(Op op, const bool& skip_update)
        : op_(std::move(op)), skip_update_(skip_update) {}

    /// @brief Component name.
    /// @return `"gpw_mix_stage"`.
    std::string name() const override { return "gpw_mix_stage"; }

    /// @brief Default binding: `gpw::Phase::mix` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::mix, 0)};
    }

    /// @brief DIIS / mix step unless skip_update.
    /// @param ctx Launch context; self-guards on `gpw::Phase::mix`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::mix))
            return;
        if (skip_update_)
            return;
        op_();
    }

    /// @brief Clear path-owned DIIS history (Op may capture reset hooks).
    void reset() {
        if (reset_)
            reset_();
    }

    /// @brief Optional reset hook for DIIS history at the start of a run.
    /// @param r Reset callable.
    void set_reset(std::function<void()> r) { reset_ = std::move(r); }

  private:
    Op op_;
    const bool& skip_update_;
    std::function<void()> reset_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
