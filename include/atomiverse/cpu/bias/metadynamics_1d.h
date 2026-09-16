/// @file cpu/bias/metadynamics_1d.h
/// @brief Well-tempered 1D metadynamics Bias (CPU).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cpu/bias/collective_variable.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief History-dependent bias \(V(s)=\sum_k h_k e^{-(s-s_k)^2/2\sigma^2}\).
///
/// Well-tempered height: \(h_k = h_0 \exp(-V(s_k)/((γ-1)k_B T))\).
/// Hills deposited every `pace` launches.
class Metadynamics1D : public cpu::Bias {
  public:
    /// @param cv Owned collective variable.
    /// @param height Initial hill height h0 [eV].
    /// @param sigma Hill width [CV unit].
    /// @param temperature T [K] for well-tempered factor.
    /// @param bias_factor γ ≥ 1 (well-tempered; γ=1 → non-WT constant height).
    /// @param pace Deposit one hill every this many launches (≥ 1).
    Metadynamics1D(std::unique_ptr<CollectiveVariable> cv, double height,
                   double sigma, double temperature, double bias_factor = 10.0,
                   int pace = 10);

    std::string name() const override { return "metadynamics_1d"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    int n_hills() const { return static_cast<int>(centers_.size()); }
    double last_cv() const { return last_cv_; }
    double last_bias_energy() const { return last_u_; }

  private:
    double bias_at_(double s) const;

    std::unique_ptr<CollectiveVariable> cv_;
    double h0_ = 0;
    double sigma_ = 0;
    double temperature_ = 300;
    double gamma_ = 10;
    int pace_ = 10;
    int step_ = 0;
    std::vector<double> centers_;
    std::vector<double> heights_;
    mutable std::vector<double> grad_;
    double last_cv_ = 0;
    double last_u_ = 0;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
