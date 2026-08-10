/// @file cpu/gpw/stages/diagonalize_stage.h
/// @brief DiagonalizeStage — `gpw::Phase::diagonalize`: KS eigenproblem.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Generalized eigenproblem stage for the GPW Driver loop.
///
/// Calls a path-local diagonalizer (level-shift aware for Γ RKS) so the free
/// helper `diagonalize_fock_shifted` stays the single implementation.
class DiagonalizeStage : public cpu::Component {
  public:
    /// @brief (eig, C) from the mixed KS matrix.
    using DiagFn = std::function<std::pair<linalg::RealVector, linalg::RealMatrix>(
        const linalg::RealMatrix& H_ext, const linalg::RealMatrix& C_prev)>;

    /// @param diag         Path-local diagonalizer.
    /// @param H_ext        Mixed KS matrix (read).
    /// @param C            MO coefficients (write); also previous C for shift.
    /// @param skip_update  When true, leave C unchanged.
    DiagonalizeStage(DiagFn diag, const linalg::RealMatrix& H_ext,
                     linalg::RealMatrix& C, const bool& skip_update)
        : diag_(std::move(diag)), H_ext_(H_ext), C_(C),
          skip_update_(skip_update) {}

    /// @brief Component name.
    /// @return `"gpw_diagonalize_stage"`.
    std::string name() const override { return "gpw_diagonalize_stage"; }

    /// @brief Default binding: `gpw::Phase::diagonalize` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::diagonalize, 0)};
    }

    /// @brief Diagonalize H_ext → C.
    /// @param ctx Launch context; self-guards on `gpw::Phase::diagonalize`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::diagonalize))
            return;
        if (skip_update_)
            return;
        auto [eig, C] = diag_(H_ext_, C_);
        (void)eig;
        C_ = std::move(C);
    }

  private:
    DiagFn diag_;
    const linalg::RealMatrix& H_ext_;
    linalg::RealMatrix& C_;
    const bool& skip_update_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
