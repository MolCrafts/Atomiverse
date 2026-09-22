/// @file cpu/electronic/second_gen_cp.h
#include <cmath>
/// @brief SecondGenCp — force-slot evaluator: ASPC + partial SCF on GPW.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Second-generation Car–Parrinello (Kühne) on a pure `GPWEvaluator`.
class SecondGenCp : public cpu::GPW {
  public:
    /// @param method Pure GPW evaluator (moved in).
    /// @param aspc_k ASPC extrapolation order K (Kolafa).
    /// @param corrector_max_iter Partial SCF budget per warm step.
    SecondGenCp(gpw::GPWEvaluator method, int aspc_k = 3,
                int corrector_max_iter = 5)
        : method_(std::move(method)) {
        if (aspc_k < 0)
            throw std::invalid_argument("SecondGenCp: aspc_k must be >= 0");
        if (corrector_max_iter < 1)
            throw std::invalid_argument(
                "SecondGenCp: corrector_max_iter must be >= 1");
        method_.second_gen_cp = true;
        method_.second_gen_cp_k = aspc_k;
        method_.second_gen_cp_max_iter = corrector_max_iter;
        method_.compute_forces = true;
    }

    std::string name() const override { return "second_gen_cp"; }

    void build(const cpu::BuildContext& ctx) override { method_.build(ctx); }
    void launch(const cpu::LaunchContext& ctx) override { method_.launch(ctx); }

    gpw::GPWEvaluator& method() { return method_; }
    const gpw::GPWEvaluator& method() const { return method_; }
    double last_energy_hartree() const { return method_.last_energy_hartree(); }
    const gpw::GpwResult& last_result() const { return method_.last_result(); }

  private:
    gpw::GPWEvaluator method_;
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
