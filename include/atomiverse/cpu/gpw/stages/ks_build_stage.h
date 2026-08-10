/// @file cpu/gpw/stages/ks_build_stage.h
/// @brief KsBuildStage — `gpw::Phase::ks_build`: H[P] + energy + residual gate.
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

/// @brief KS Hamiltonian build stage for the GPW Driver loop.
///
/// Invokes a **path-local** Op (Γ RKS, Γ UKS, multi-k RKS/UKS, …) supplied
/// by the method at wire time. The Op owns residual/DIIS-error evaluation and
/// sets `skip_update` when the shipping gate is met so Mix/Diag/Density leave
/// the raw H for `store_converged_state` (break-before-mix semantics).
class KsBuildStage : public cpu::Component {
  public:
    /// @brief Path-local body: build H, energy, residual; arm skip_update.
    using Op = std::function<void()>;

    /// @param op Path-local KS build + residual gate.
    explicit KsBuildStage(Op op) : op_(std::move(op)) {}

    /// @brief Component name.
    /// @return `"gpw_ks_build_stage"`.
    std::string name() const override { return "gpw_ks_build_stage"; }

    /// @brief Default binding: `gpw::Phase::ks_build` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::ks_build, 0)};
    }

    /// @brief Run the path-local KS build Op.
    /// @param ctx Launch context; self-guards on `gpw::Phase::ks_build`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::ks_build))
            return;
        op_();
    }

  private:
    Op op_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
