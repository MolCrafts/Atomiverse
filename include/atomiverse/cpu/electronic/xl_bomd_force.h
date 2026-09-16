/// @file cpu/electronic/xl_bomd_force.h
/// @brief XlBomdForce — force-slot evaluator: Niklasson XL-BOMD on GPW.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Tag: zero-SCF linearized-functional XL forces (Γ and multi-k,
///        RKS and collinear UKS).
struct XlBomdLinearized {};

/// @brief Extended-Lagrangian BOMD evaluator wrapping a pure `GPWEvaluator`.
class XlBomdForce : public cpu::GPW {
  public:
    /// @param method Pure GPW evaluator (moved in).
    /// @param k Dissipation order K ∈ [3, 9] (Niklasson Table I).
    XlBomdForce(gpw::GPWEvaluator method, int k = 5)
        : method_(std::move(method)) {
        method_.xl_bomd = true;
        method_.xl_bomd_k = k;
        method_.xl_bomd_linearized = false;
    }

    /// @brief Linearized shadow-force path (requires forces; Γ and multi-k,
    ///        RKS and collinear UKS).
    XlBomdForce(gpw::GPWEvaluator method, int k, XlBomdLinearized)
        : XlBomdForce(std::move(method), k) {
        method_.xl_bomd_linearized = true;
        method_.compute_forces = true;
    }

    std::string name() const override { return "xl_bomd_force"; }

    void build(const cpu::BuildContext& ctx) override { method_.build(ctx); }
    void launch(const cpu::LaunchContext& ctx) override {
        method_.launch(ctx);
    }

    gpw::GPWEvaluator& method() { return method_; }
    const gpw::GPWEvaluator& method() const { return method_; }
    double last_energy_hartree() const {
        return method_.last_energy_hartree();
    }
    const gpw::GpwResult& last_result() const { return method_.last_result(); }

  private:
    gpw::GPWEvaluator method_;
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
