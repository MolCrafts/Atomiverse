/// @file src/cpu/bias/harmonic_bias.cpp
#include "atomiverse/cpu/bias/harmonic_bias.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/role.h"

namespace ATV {
namespace cpu {
namespace bias {

HarmonicBias::HarmonicBias(std::unique_ptr<CollectiveVariable> cv, double k,
                           double center)
    : cv_(std::move(cv)), k_(k), center_(center) {
    if (!cv_)
        throw std::invalid_argument("HarmonicBias: null CV");
    if (!(k_ > 0.0))
        throw std::invalid_argument("HarmonicBias: k must be > 0");
}

void HarmonicBias::build(const BuildContext& ctx) {
    cv_->build(ctx.frame);
    grad_.assign(static_cast<std::size_t>(3 * ctx.frame.capacity()), 0.0);
}

void HarmonicBias::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (static_cast<int>(grad_.size()) < 3 * f.nlocal())
        grad_.assign(static_cast<std::size_t>(3 * f.capacity()), 0.0);

    const bool want_force = ctx.forces_requested;

    last_cv_ = cv_->evaluate(f, want_force ? grad_.data() : nullptr);
    const double d = last_cv_ - center_;
    last_u_ = 0.5 * k_ * d * d;
    f.meta.get(kEnergy) += last_u_;

    if (want_force) {
        const double fac = -k_ * d;
        const int n = f.nlocal();
        for (int i = 0; i < n; ++i) {
            f.span(kAtom_fx)[i] += fac * grad_[3 * i + 0];
            f.span(kAtom_fy)[i] += fac * grad_[3 * i + 1];
            f.span(kAtom_fz)[i] += fac * grad_[3 * i + 2];
        }
    }
}

} // namespace bias
} // namespace cpu
} // namespace ATV
