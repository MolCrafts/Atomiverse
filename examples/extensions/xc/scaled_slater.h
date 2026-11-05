#pragma once
#include "atomiverse/electronic/xc_functional.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace xc_example {

/// External spin-scaled Slater exchange, with no correlation or exact exchange.
/// E_x = -scale * (3/4)(6/pi)^(1/3) integral (rho_a^(4/3)+rho_b^(4/3)) dr.
class ScaledSlater final : public ATV::electronic::XcFunctional {
  public:
    explicit ScaledSlater(double scale = 0.8) : scale_(scale) {
        if (!std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument(
                "ScaledSlater: scale must be finite and positive");
    }
    ATV::electronic::XcIngredients ingredients() const noexcept override {
        return ATV::electronic::XcIngredients::Rho;
    }
    bool supports(ATV::electronic::XcSpin spin,
                  ATV::electronic::XcOrder order) const noexcept override {
        using namespace ATV::electronic;
        return (spin == XcSpin::Restricted || spin == XcSpin::Collinear)
               && (order == XcOrder::Energy || order == XcOrder::First);
    }

  protected:
    void evaluate_batch(const ATV::electronic::XcBatch& in,
                        ATV::electronic::XcOrder order,
                        ATV::electronic::XcOutput& out) const override {
        const double c = scale_ * 0.75 * std::cbrt(6.0 / std::numbers::pi);
        for (std::size_t i = 0; i < in.rho_alpha.size(); ++i) {
            const double a = std::max(in.rho_alpha[i], 0.0);
            const double b = std::max(in.rho_beta[i], 0.0);
            if (a + b < 1e-12) {
                out.energy_density[i] = 0;
                if (order == ATV::electronic::XcOrder::First)
                    out.vrho_alpha[i] = out.vrho_beta[i] = 0;
                continue;
            }
            const double aa = std::cbrt(a), bb = std::cbrt(b);
            out.energy_density[i] = -c * (a * aa + b * bb);
            if (order == ATV::electronic::XcOrder::First) {
                out.vrho_alpha[i] = -(4.0 / 3.0) * c * aa;
                out.vrho_beta[i] = -(4.0 / 3.0) * c * bb;
            }
        }
    }

  private:
    const double scale_;
};
} // namespace xc_example
