/// @file src/cpu/path/irc.cpp
#include "atomiverse/cpu/path/irc.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace path {

namespace {

void muller_brown_force(cpu::Frame& f) {
    constexpr double kA[4] = {-200.0, -100.0, -170.0, 15.0};
    constexpr double kAa[4] = {-1.0, -1.0, -6.5, 0.7};
    constexpr double kBb[4] = {0.0, 0.0, 11.0, 0.6};
    constexpr double kCc[4] = {-10.0, -10.0, -6.5, 0.7};
    constexpr double kX0[4] = {1.0, 0.0, -0.5, -1.0};
    constexpr double kY0[4] = {0.0, 0.5, 1.5, 1.0};
    const double x = f.span(kAtom_x)[0];
    const double y = f.span(kAtom_y)[0];
    double v = 0, dvdx = 0, dvdy = 0;
    for (int k = 0; k < 4; ++k) {
        const double dx = x - kX0[k];
        const double dy = y - kY0[k];
        const double e =
            kA[k]
            * std::exp(kAa[k] * dx * dx + kBb[k] * dx * dy + kCc[k] * dy * dy);
        v += e;
        dvdx += e * (2.0 * kAa[k] * dx + kBb[k] * dy);
        dvdy += e * (kBb[k] * dx + 2.0 * kCc[k] * dy);
    }
    f.meta.get(kEnergy) = v;
    f.span(kAtom_fx)[0] = -dvdx;
    f.span(kAtom_fy)[0] = -dvdy;
    f.span(kAtom_fz)[0] = 0;
}

} // namespace

Irc::Irc(double step_, int max_steps_, double f_tol_, double kick_)
    : step(step_), max_steps(max_steps_), f_tol(f_tol_), kick(kick_) {
    if (!(step > 0.0))
        throw std::invalid_argument("Irc: step > 0");
}

void Irc::set_force_fn(ForceFn fn) { force_ = std::move(fn); }
void Irc::use_muller_brown() { force_ = muller_brown_force; }

void Irc::branch_(cpu::Frame f, double mx, double my, bool forward) {
    auto& path = forward ? path_E_fwd_ : path_E_bwd_;
    path.clear();
    // Kick off the saddle along ±mode so we leave the stationary point.
    const double sign = forward ? 1.0 : -1.0;
    const double nrm = std::hypot(mx, my);
    if (nrm > 0) {
        f.span(kAtom_x)[0] += sign * kick * mx / nrm;
        f.span(kAtom_y)[0] += sign * kick * my / nrm;
    }
    force_(f);
    path.push_back(f.meta.get(kEnergy));

    for (int s = 0; s < max_steps; ++s) {
        const double fx = f.span(kAtom_fx)[0];
        const double fy = f.span(kAtom_fy)[0];
        const double fz = f.span(kAtom_fz)[0];
        const double fn = std::sqrt(fx * fx + fy * fy + fz * fz);
        if (fn < f_tol)
            break;
        // Mass-weighted SD with m=1: x += step * F/|F|
        f.span(kAtom_x)[0] += step * fx / fn;
        f.span(kAtom_y)[0] += step * fy / fn;
        f.span(kAtom_z)[0] += step * fz / fn;
        force_(f);
        const double E = f.meta.get(kEnergy);
        // Allow tiny numerical rises near flat regions; record always.
        path.push_back(E);
    }
    if (forward)
        E_fwd_ = path.back();
    else
        E_bwd_ = path.back();
}

void Irc::run(const cpu::Frame& saddle, double mode_x, double mode_y) {
    if (!force_)
        throw std::runtime_error("Irc: force function not set");
    branch_(saddle, mode_x, mode_y, true);
    branch_(saddle, mode_x, mode_y, false);
}

} // namespace path
} // namespace cpu
} // namespace ATV
