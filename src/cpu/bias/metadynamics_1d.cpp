/// @file src/cpu/bias/metadynamics_1d.cpp
#include "atomiverse/cpu/bias/metadynamics_1d.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace bias {

Metadynamics1D::Metadynamics1D(std::unique_ptr<CollectiveVariable> cv,
                               double height, double sigma, double temperature,
                               double bias_factor, int pace)
    : cv_(std::move(cv)), h0_(height), sigma_(sigma), temperature_(temperature),
      gamma_(bias_factor), pace_(pace) {
    if (!cv_)
        throw std::invalid_argument("Metadynamics1D: null CV");
    if (!(h0_ > 0.0) || !(sigma_ > 0.0) || !(temperature_ > 0.0)
        || !(gamma_ >= 1.0) || pace_ < 1)
        throw std::invalid_argument("Metadynamics1D: invalid parameters");
}

void Metadynamics1D::build(const BuildContext& ctx) {
    cv_->build(ctx.frame);
    grad_.assign(static_cast<std::size_t>(3 * ctx.frame.capacity()), 0.0);
    centers_.clear();
    heights_.clear();
    step_ = 0;
}

double Metadynamics1D::bias_at_(double s) const {
    double v = 0;
    const double inv2s2 = 1.0 / (2.0 * sigma_ * sigma_);
    for (std::size_t k = 0; k < centers_.size(); ++k) {
        const double d = s - centers_[k];
        v += heights_[k] * std::exp(-d * d * inv2s2);
    }
    return v;
}

void Metadynamics1D::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (static_cast<int>(grad_.size()) < 3 * f.nlocal())
        grad_.assign(static_cast<std::size_t>(3 * f.capacity()), 0.0);

    const bool want_force = ctx.forces_requested;
    last_cv_ = cv_->evaluate(f, want_force ? grad_.data() : nullptr);

    // Deposit hill before evaluating bias for this step (after pace).
    if (step_ > 0 && (step_ % pace_) == 0) {
        const double Vhere = bias_at_(last_cv_);
        double h = h0_;
        if (gamma_ > 1.0) {
            const double kT = units::kB_ev_per_k * temperature_;
            h = h0_ * std::exp(-Vhere / ((gamma_ - 1.0) * kT));
        }
        centers_.push_back(last_cv_);
        heights_.push_back(h);
    }
    ++step_;

    last_u_ = bias_at_(last_cv_);
    f.meta.get(kEnergy) += last_u_;

    if (want_force && !centers_.empty()) {
        // dV/ds = Σ h_k (−(s−s_k)/σ²) exp(...)
        double dVds = 0;
        const double inv_s2 = 1.0 / (sigma_ * sigma_);
        const double inv2s2 = 0.5 * inv_s2;
        for (std::size_t k = 0; k < centers_.size(); ++k) {
            const double d = last_cv_ - centers_[k];
            dVds += heights_[k] * (-d * inv_s2) * std::exp(-d * d * inv2s2);
        }
        const double fac = -dVds; // F = −∇V = −(dV/ds) ∇s
        const int n = f.nlocal();
        AccReal* fx = f.span(kAtom_fx);
        AccReal* fy = f.span(kAtom_fy);
        AccReal* fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            fx[i] += fac * grad_[3 * i + 0];
            fy[i] += fac * grad_[3 * i + 1];
            fz[i] += fac * grad_[3 * i + 2];
        }
    }
}

} // namespace bias
} // namespace cpu
} // namespace ATV
