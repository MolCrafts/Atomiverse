/// @file src/cpu/minimize/cell_relax.cpp
#include "atomiverse/cpu/minimize/cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace minimizer {

IsotropicCellRelax::IsotropicCellRelax(double P_target_ev_a3, double alpha)
    : P_target(P_target_ev_a3), alpha0(alpha) {
    if (!(alpha > 0.0))
        throw std::invalid_argument("IsotropicCellRelax: alpha must be > 0");
    if (!(max_scale > 0.0))
        throw std::invalid_argument(
            "IsotropicCellRelax: max_scale must be > 0");
    if (!(bulk_mod > 0.0))
        throw std::invalid_argument("IsotropicCellRelax: bulk_mod must be > 0");
}

double IsotropicCellRelax::pressure_from_frame_(const Frame& f) const {
    // kStress is the extensive Voigt virial [eV] for every producer, so
    // P = tr(W)/(3V) unconditionally. Kinetic neglected for 0 K geometry
    // optimization (it lives on kStressKinetic).
    const double V = f.box.volume();
    if (!(V > 0.0))
        throw std::runtime_error("IsotropicCellRelax: non-positive volume");
    const auto& s = f.meta.get(kStress);
    const double tr = s[0] + s[1] + s[2];
    return tr / (3.0 * V);
}

void IsotropicCellRelax::apply_scale_(Frame& f, double s) const {
    if (!(s > 0.0) || !std::isfinite(s))
        throw std::runtime_error("IsotropicCellRelax: invalid scale");
    f.box.scale_isotropic(s);
    const int n = f.nlocal();
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        x[i] *= s;
        y[i] *= s;
        z[i] *= s;
    }
}

void IsotropicCellRelax::build(const BuildContext&) {
    pressure_ = 0;
    p_residual_ = 0;
    scale_step_ = 1.0;
    energy_prev_ = 0;
    step_ = 0;
    has_prev_ = false;
}

void IsotropicCellRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        // Stress from the previous force phase (0 on first iter).
        pressure_ = pressure_from_frame_(f);
        p_residual_ = std::fabs(pressure_ - P_target);
        // Positive pressure (compressed) → expand: s > 1 when P > P_target.
        double ds = alpha0 * (pressure_ - P_target) / bulk_mod;
        ds = std::max(-max_scale, std::min(max_scale, ds));
        scale_step_ = 1.0 + ds;
        ++step_;
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        if (step_ > 0)
            apply_scale_(f, scale_step_);
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        pressure_ = pressure_from_frame_(f);
        p_residual_ = std::fabs(pressure_ - P_target);
        const double energy = f.meta.get(kEnergy);
        const double dE = has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
        // Publish pressure residual as max_force so ForceConverged *can*
        // judge it if the user sets ftol in eV/Å³; prefer PressureConverged.
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = p_residual_;
            m->rms_force = p_residual_;
            m->max_displacement = std::fabs(scale_step_ - 1.0);
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
