/// @file cpu/bias/harmonic_bias.h
/// @brief HarmonicBias — U = ½ k (ξ−ξ0)², F = −k(ξ−ξ0) ∇ξ.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/cpu/bias/collective_variable.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief Bias evaluator; writes Force only when ctx requests Force capability.
class HarmonicBias : public cpu::Bias {
  public:
    /// @param cv Owned CV (must outlive only via ownership here).
    /// @param k Force constant [eV / CV-unit²].
    /// @param center Bias centre ξ0 [CV unit].
    HarmonicBias(std::unique_ptr<CollectiveVariable> cv, double k,
                 double center);

    std::string name() const override { return "harmonic_bias"; }
    /* role via Potential/kind base */

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double last_cv() const { return last_cv_; }
    double last_bias_energy() const { return last_u_; }

  private:
    std::unique_ptr<CollectiveVariable> cv_;
    double k_;
    double center_;
    mutable std::vector<double> grad_;
    double last_cv_ = 0.0;
    double last_u_ = 0.0;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
