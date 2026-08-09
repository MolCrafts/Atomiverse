/// @file src/cpu/minimize/anisotropic_cell_relax.cpp
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace minimizer {

AnisotropicCellRelax::AnisotropicCellRelax(double P_target_ev_a3, double alpha)
    : P_target(P_target_ev_a3), alpha0(alpha) {
    if (!(alpha > 0.0) || !(max_scale > 0.0) || !(bulk_mod > 0.0))
        throw std::invalid_argument("AnisotropicCellRelax: invalid parameters");
}

void AnisotropicCellRelax::apply_scales_(Frame& f, double sx, double sy,
                                         double sz) const {
    if (!(sx > 0.0) || !(sy > 0.0) || !(sz > 0.0))
        throw std::runtime_error("AnisotropicCellRelax: invalid scale");
    f.box.scale_diagonal(sx, sy, sz);
    const int n = f.nlocal();
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        x[i] *= sx;
        y[i] *= sy;
        z[i] *= sz;
    }
}

void AnisotropicCellRelax::build(const BuildContext&) {
    if (!relax_x && !relax_y && !relax_z)
        throw std::invalid_argument(
            "AnisotropicCellRelax: every axis is masked (relax_x/y/z all "
            "false)");
    pressure_ = p_residual_ = 0;
    sx_ = sy_ = sz_ = 1;
    energy_prev_ = 0;
    step_ = 0;
    has_prev_ = false;
}

void AnisotropicCellRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const double V = f.box.volume();
    if (!(V > 0.0))
        throw std::runtime_error("AnisotropicCellRelax: non-positive volume");
    const auto& s = f.meta.get(kStress);

    // kStress is the extensive Voigt virial [eV] for every producer, so the
    // per-axis pressure is W_dd/V unconditionally.
    const double Px = s[0] / V;
    const double Py = s[1] / V;
    const double Pz = s[2] / V;

    // Residual over the RELAXED axes only (masked axes are not driven to
    // P_target — their stress is a constraint reaction, not an objective).
    auto residual = [&] {
        double r = 0.0;
        if (relax_x)
            r = std::max(r, std::fabs(Px - P_target));
        if (relax_y)
            r = std::max(r, std::fabs(Py - P_target));
        if (relax_z)
            r = std::max(r, std::fabs(Pz - P_target));
        return r;
    };

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        pressure_ = (Px + Py + Pz) / 3.0;
        p_residual_ = residual();
        auto one_s = [&](double P, bool relax) {
            if (!relax)
                return 1.0;
            double ds = alpha0 * (P - P_target) / bulk_mod;
            ds = std::max(-max_scale, std::min(max_scale, ds));
            return 1.0 + ds;
        };
        sx_ = one_s(Px, relax_x);
        sy_ = one_s(Py, relax_y);
        sz_ = one_s(Pz, relax_z);
        ++step_;
        return;
    }
    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        if (step_ > 0)
            apply_scales_(f, sx_, sy_, sz_);
        return;
    }
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        pressure_ = (Px + Py + Pz) / 3.0;
        p_residual_ = residual();
        const double energy = f.meta.get(kEnergy);
        const double dE = has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = p_residual_;
            m->rms_force = p_residual_;
            m->max_displacement = std::max(
                {std::fabs(sx_ - 1), std::fabs(sy_ - 1), std::fabs(sz_ - 1)});
            m->rms_displacement = m->max_displacement;
            m->energy_change = dE;
            m->has_displacement = has_prev_;
        }
        energy_prev_ = energy;
        has_prev_ = true;
    }
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
