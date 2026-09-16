/// @file cpu/semiempirical/rm1_evaluator.h
/// @brief RM1 Evaluator component (sibling of AM1Evaluator).
#pragma once

#include <string>
#include <utility>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief RM1 semiempirical Evaluator (owns concrete `RM1` method).
class RM1Evaluator : public cpu::Semiempirical {
  public:
    RM1Evaluator() = default;
    explicit RM1Evaluator(RM1 method) : solver_(std::move(method)) {}

    std::string name() const override { return "rm1_evaluator"; }

    RM1Evaluator& with_gradient(bool enabled) & noexcept {
        compute_force_ = enabled;
        return *this;
    }
    RM1Evaluator&& with_gradient(bool enabled) && noexcept {
        compute_force_ = enabled;
        return std::move(*this);
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    const RM1Result& last_result() const { return solver_.last_result(); }
    const RM1& method() const { return solver_; }
    RM1& method() { return solver_; }
    bool gradient_enabled() const noexcept { return compute_force_; }

  private:
    RM1 solver_;
    bool compute_force_ = true;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
