/// @file cpu/gpw/stages/diagonalize_stage.h
/// @brief DiagonalizeStage — `gpw::Phase::diagonalize`: KS eigenproblem.
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

/// @brief Eigenproblem stage for the GPW Driver loop (path-local Op).
///
/// Path Op diagonalizes the mixed KS matrix (Γ real, multi-k complex,
/// dual-spin UKS). Honors `skip_update` so a just-converged iteration leaves
/// orbitals untouched.
class DiagonalizeStage : public cpu::Component {
  public:
    /// @brief Path-local diagonalizer body.
    using Op = std::function<void()>;

    /// @param op           Path-local diagonalize.
    /// @param skip_update  When true, leave C unchanged.
    DiagonalizeStage(Op op, const bool& skip_update)
        : op_(std::move(op)), skip_update_(skip_update) {}

    /// @brief Component name.
    /// @return `"gpw_diagonalize_stage"`.
    std::string name() const override { return "gpw_diagonalize_stage"; }

    /// @brief Default binding: `gpw::Phase::diagonalize` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::diagonalize, 0)};
    }

    /// @brief Diagonalize unless skip_update.
    /// @param ctx Launch context; self-guards on `gpw::Phase::diagonalize`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::diagonalize))
            return;
        if (skip_update_)
            return;
        op_();
    }

  private:
    Op op_;
    const bool& skip_update_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
