/// @file cpu/observe/charge_autocorr.cpp
#include "atomiverse/cpu/observe/charge_autocorr.h"

#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {

ChargeChargeAutocorr::ChargeChargeAutocorr(md::ConstPotBO* conp, int every,
                                           int max_lag)
    : conp_(conp), every_(every), max_lag_(max_lag) {
    if (!conp_)
        throw std::invalid_argument("ChargeChargeAutocorr: null ConstPotBO");
    if (every_ < 1)
        throw std::invalid_argument("ChargeChargeAutocorr: every must be ≥ 1");
    if (max_lag_ < 1)
        throw std::invalid_argument(
            "ChargeChargeAutocorr: max_lag must be ≥ 1");
}

void ChargeChargeAutocorr::launch(const LaunchContext& ctx) {
    if (every_ > 1 && (ctx.step % every_) != 0)
        return;
    double Q = 0.0;
    // Prefer first electrode group if multi; else all electrodes.
    if (conp_->n_electrode() > 0) {
        // total on electrode 0 (one plate) for EDL charge series
        Q = conp_->total_charge(0);
    }
    // Online mean for δQ
    n_mean_ += 1;
    mean_ += (Q - mean_) / static_cast<double>(n_mean_);
    history_.push_back(Q);
    while (static_cast<int>(history_.size()) > max_lag_ + 1)
        history_.pop_front();
}

double ChargeChargeAutocorr::C0() const {
    if (history_.empty())
        return 0.0;
    double s = 0.0;
    for (double q : history_) {
        const double d = q - mean_;
        s += d * d;
    }
    return s / static_cast<double>(history_.size());
}

double ChargeChargeAutocorr::C_lag(int lag) const {
    if (lag < 0 || lag > max_lag_)
        return 0.0;
    const int n = static_cast<int>(history_.size());
    if (n <= lag)
        return 0.0;
    double s = 0.0;
    int cnt = 0;
    for (int t = 0; t + lag < n; ++t) {
        const double a = history_[static_cast<std::size_t>(t)] - mean_;
        const double b = history_[static_cast<std::size_t>(t + lag)] - mean_;
        s += a * b;
        ++cnt;
    }
    return cnt > 0 ? s / static_cast<double>(cnt) : 0.0;
}

} // namespace cpu
} // namespace ATV
