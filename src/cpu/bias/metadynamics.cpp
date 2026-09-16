/// @file src/cpu/bias/metadynamics.cpp
#include "atomiverse/cpu/bias/metadynamics.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace bias {

Metadynamics::Metadynamics(std::vector<std::unique_ptr<CollectiveVariable>> cvs,
                           double height, std::vector<double> sigma,
                           double temperature, double bias_factor, int pace)
    : cvs_(std::move(cvs)), h0_(height), sigma_(std::move(sigma)),
      temperature_(temperature), gamma_(bias_factor), pace_(pace) {
    if (cvs_.empty())
        throw std::invalid_argument("Metadynamics: empty CV set");
    for (const auto& c : cvs_)
        if (!c)
            throw std::invalid_argument("Metadynamics: null CV");
    if (sigma_.size() != cvs_.size())
        throw std::invalid_argument("Metadynamics: sigma size != n_cv");
    if (!(h0_ > 0.0) || !(temperature_ > 0.0) || !(gamma_ >= 1.0) || pace_ < 1)
        throw std::invalid_argument("Metadynamics: invalid parameters");
    for (double s : sigma_)
        if (!(s > 0.0))
            throw std::invalid_argument("Metadynamics: sigma must be > 0");
}

void Metadynamics::build(const BuildContext& ctx) {
    const int n = ndim();
    last_s_.assign(static_cast<std::size_t>(n), 0.0);
    grads_.assign(static_cast<std::size_t>(n), {});
    for (int a = 0; a < n; ++a) {
        cvs_[static_cast<std::size_t>(a)]->build(ctx.frame);
        grads_[static_cast<std::size_t>(a)].assign(
            static_cast<std::size_t>(3 * ctx.frame.capacity()), 0.0);
    }
    centers_.clear();
    heights_.clear();
    step_ = 0;
}

double Metadynamics::bias_at_(const std::vector<double>& s) const {
    const int n = ndim();
    double v = 0;
    for (std::size_t k = 0; k < heights_.size(); ++k) {
        double arg = 0;
        for (int a = 0; a < n; ++a) {
            const double d = s[static_cast<std::size_t>(a)]
                             - centers_[k * static_cast<std::size_t>(n)
                                        + static_cast<std::size_t>(a)];
            const double sig = sigma_[static_cast<std::size_t>(a)];
            arg += (d * d) / (2.0 * sig * sig);
        }
        v += heights_[k] * std::exp(-arg);
    }
    return v;
}

void Metadynamics::dV_ds_(const std::vector<double>& s,
                          std::vector<double>& out) const {
    const int n = ndim();
    out.assign(static_cast<std::size_t>(n), 0.0);
    for (std::size_t k = 0; k < heights_.size(); ++k) {
        double arg = 0;
        std::vector<double> d(static_cast<std::size_t>(n));
        for (int a = 0; a < n; ++a) {
            d[static_cast<std::size_t>(a)] =
                s[static_cast<std::size_t>(a)]
                - centers_[k * static_cast<std::size_t>(n)
                           + static_cast<std::size_t>(a)];
            const double sig = sigma_[static_cast<std::size_t>(a)];
            arg += (d[static_cast<std::size_t>(a)]
                    * d[static_cast<std::size_t>(a)])
                   / (2.0 * sig * sig);
        }
        const double w = heights_[k] * std::exp(-arg);
        for (int a = 0; a < n; ++a) {
            const double sig = sigma_[static_cast<std::size_t>(a)];
            out[static_cast<std::size_t>(a)] +=
                w * (-d[static_cast<std::size_t>(a)] / (sig * sig));
        }
    }
}

void Metadynamics::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = ndim();
    const int nat = f.nlocal();
    const bool want_force = ctx.forces_requested;

    for (int a = 0; a < n; ++a) {
        if (static_cast<int>(grads_[static_cast<std::size_t>(a)].size())
            < 3 * nat)
            grads_[static_cast<std::size_t>(a)].assign(
                static_cast<std::size_t>(3 * f.capacity()), 0.0);
        last_s_[static_cast<std::size_t>(a)] =
            cvs_[static_cast<std::size_t>(a)]->evaluate(
                f, want_force ? grads_[static_cast<std::size_t>(a)].data()
                              : nullptr);
    }

    if (step_ > 0 && (step_ % pace_) == 0) {
        const double Vhere = bias_at_(last_s_);
        double h = h0_;
        if (gamma_ > 1.0) {
            const double kT = units::kB_ev_per_k * temperature_;
            h = h0_ * std::exp(-Vhere / ((gamma_ - 1.0) * kT));
        }
        for (int a = 0; a < n; ++a)
            centers_.push_back(last_s_[static_cast<std::size_t>(a)]);
        heights_.push_back(h);
    }
    ++step_;

    last_u_ = bias_at_(last_s_);
    f.meta.get(kEnergy) += last_u_;

    if (want_force && !heights_.empty()) {
        std::vector<double> dVds;
        dV_ds_(last_s_, dVds);
        AccReal* fx = f.span(kAtom_fx);
        AccReal* fy = f.span(kAtom_fy);
        AccReal* fz = f.span(kAtom_fz);
        for (int a = 0; a < n; ++a) {
            const double fac = -dVds[static_cast<std::size_t>(a)];
            const auto& g = grads_[static_cast<std::size_t>(a)];
            for (int i = 0; i < nat; ++i) {
                fx[i] += fac * g[3 * i + 0];
                fy[i] += fac * g[3 * i + 1];
                fz[i] += fac * g[3 * i + 2];
            }
        }
    }
}

} // namespace bias
} // namespace cpu
} // namespace ATV
