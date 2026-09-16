/// @file src/cpu/path/ring_polymer_spring.cpp
#include "atomiverse/cpu/path/ring_polymer_spring.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace path {

RingPolymerSpring::RingPolymerSpring(ReplicaSet* band, int bead_index,
                                     double k_spring)
    : band_(band), bead_(bead_index), k_(k_spring) {
    if (!band_)
        throw std::invalid_argument("RingPolymerSpring: band is null");
    if (bead_index < 0 || bead_index >= band_->size())
        throw std::invalid_argument(
            "RingPolymerSpring: bead_index out of range");
    if (!(k_spring >= 0.0))
        throw std::invalid_argument("RingPolymerSpring: k_spring must be ≥ 0");
}

void RingPolymerSpring::launch(const LaunchContext& ctx) {
    if (!band_)
        return;
    const int P = band_->size();
    if (P < 2)
        return;
    const int prev = (bead_ + P - 1) % P;
    const int next = (bead_ + 1) % P;

    auto& f = ctx.frame;
    auto& fp = band_->frame(prev);
    auto& fn = band_->frame(next);

    const int n = f.nlocal();
    if (fp.nlocal() != n || fn.nlocal() != n)
        throw std::runtime_error(
            "RingPolymerSpring: bead atom counts disagree");

    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    const auto xp = fp.span(kAtom_x);
    const auto yp = fp.span(kAtom_y);
    const auto zp = fp.span(kAtom_z);
    const auto xn = fn.span(kAtom_x);
    const auto yn = fn.span(kAtom_y);
    const auto zn = fn.span(kAtom_z);
    auto fx = f.span(kAtom_fx);
    auto fy = f.span(kAtom_fy);
    auto fz = f.span(kAtom_fz);

    // F = −k (2r − r_prev − r_next); energy ½ k |r − r_next|² (bond once).
    const double k = k_;
    double e_spring = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dxn = x[i] - xn[i];
        const double dyn = y[i] - yn[i];
        const double dzn = z[i] - zn[i];
        e_spring += 0.5 * k * (dxn * dxn + dyn * dyn + dzn * dzn);

        fx[i] += -k * (2.0 * x[i] - xp[i] - xn[i]);
        fy[i] += -k * (2.0 * y[i] - yp[i] - yn[i]);
        fz[i] += -k * (2.0 * z[i] - zp[i] - zn[i]);
    }
    f.meta.get(kEnergy) += e_spring;
}

} // namespace path
} // namespace cpu
} // namespace ATV
