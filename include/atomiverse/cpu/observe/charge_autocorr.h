/// @file cpu/observe/charge_autocorr.h
/// @brief Electrode total-charge autocorrelation \(C_{QQ}(t)\) foothold.
#pragma once

#include <deque>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

namespace md {
class ConstPotBO;
}

/// @brief Streaming \(C_{QQ}(\tau)=\langle \delta Q(t)\delta Q(t+\tau)\rangle\).
///
/// Borrows a non-owning `ConstPotBO*` (caller keeps it alive). Samples
/// \(Q=\sum_{\mathrm{group}} Q_g\) every `every` end_of_step launches.
class ChargeChargeAutocorr : public cpu::Observer {
  public:
    /// @param conp Non-owning pointer to const-pot solver (must outlive this).
    /// @param every Sample stride in MD steps (≥1).
    /// @param max_lag Maximum lag in samples (≥1).
    ChargeChargeAutocorr(md::ConstPotBO* conp, int every = 1, int max_lag = 50);

    std::string name() const override { return "charge_charge_autocorr"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +30)};
    }

    void launch(const LaunchContext& ctx) override;

    /// @brief \(C_{QQ}(0)\) after at least one sample.
    double C0() const;
    /// @brief \(C_{QQ}(\tau)\) for lag in samples; 0 if insufficient data.
    double C_lag(int lag) const;
    int n_samples() const { return static_cast<int>(history_.size()); }

  private:
    md::ConstPotBO* conp_ = nullptr;
    int every_ = 1;
    int max_lag_ = 50;
    std::deque<double> history_;
    double mean_ = 0.0;
    int n_mean_ = 0;
};

} // namespace cpu
} // namespace ATV
