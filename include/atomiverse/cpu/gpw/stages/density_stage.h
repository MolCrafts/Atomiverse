/// @file cpu/gpw/stages/density_stage.h
/// @brief DensityStage — `gpw::Phase::density`: P from occupied orbitals.
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

/// @brief Density-build stage for the GPW Driver loop (path-local Op).
///
/// Path Op rebuilds or mixes density from orbitals (Γ RKS occ=2, Γ UKS dual
/// occ=1, multi-k integer / metallic smear). Honors `skip_update`.
class DensityStage : public cpu::Component {
  public:
    /// @brief Path-local density body.
    using Op = std::function<void()>;

    /// @param op           Path-local density rebuild / mix.
    /// @param skip_update  When true, leave P unchanged (converged break).
    DensityStage(Op op, const bool& skip_update)
        : op_(std::move(op)), skip_update_(skip_update) {}

    /// @brief Component name.
    /// @return `"gpw_density_stage"`.
    std::string name() const override { return "gpw_density_stage"; }

    /// @brief Default binding: `gpw::Phase::density` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::density, 0)};
    }

    /// @brief Rebuild or mix density unless skip_update.
    /// @param ctx Launch context; self-guards on `gpw::Phase::density`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::density))
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
