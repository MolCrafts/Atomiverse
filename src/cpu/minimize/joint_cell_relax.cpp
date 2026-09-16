/// @file src/cpu/minimize/joint_cell_relax.cpp
#include "atomiverse/cpu/minimize/joint_cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace minimizer {

JointIsotropicRelax::JointIsotropicRelax(double P_target_ev_a3,
                                         double atom_step_a)
    : P_target(P_target_ev_a3), atom_step(atom_step_a) {
    if (!(atom_step > 0.0))
        throw std::invalid_argument(
            "JointIsotropicRelax: atom_step must be > 0");
}

double JointIsotropicRelax::pressure_from_frame_(const Frame& f) {
    const double V = f.box.volume();
    if (!(V > 0.0))
        throw std::runtime_error("JointIsotropicRelax: non-positive volume");
    const auto& s = f.meta.get(kStress);
    return (s[0] + s[1] + s[2]) / (3.0 * V);
}

void JointIsotropicRelax::apply_scale_(Frame& f, double s) const {
    if (!(s > 0.0) || !std::isfinite(s))
        throw std::runtime_error("JointIsotropicRelax: invalid scale");
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

void JointIsotropicRelax::build(const BuildContext&) {
    pressure_ = 0;
    p_residual_ = 0;
    max_force_ = 0;
    scale_step_ = 1.0;
    energy_prev_ = 0;
    has_prev_ = false;
    step_ = 0;
}

void JointIsotropicRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = f.nlocal();

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        pressure_ = pressure_from_frame_(f);
        p_residual_ = std::fabs(pressure_ - P_target);
        double ds = alpha0 * (pressure_ - P_target) / bulk_mod;
        ds = std::max(-max_scale, std::min(max_scale, ds));
        scale_step_ = 1.0 + ds;

        // Max |F|
        max_force_ = 0;
        const auto fx = f.span(kAtom_fx);
        const auto fy = f.span(kAtom_fy);
        const auto fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            const double ff =
                std::sqrt(fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]);
            max_force_ = std::max(max_force_, ff);
        }
        ++step_;
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        // Atom SD: x += atom_step * F / |F|_max (or unit per-atom).
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        const auto fx = f.span(kAtom_fx);
        const auto fy = f.span(kAtom_fy);
        const auto fz = f.span(kAtom_fz);
        const double inv =
            (max_force_ > 1e-30) ? (atom_step / max_force_) : 0.0;
        for (int i = 0; i < n; ++i) {
            x[i] += inv * fx[i];
            y[i] += inv * fy[i];
            z[i] += inv * fz[i];
        }
        if (step_ > 0)
            apply_scale_(f, scale_step_);
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        pressure_ = pressure_from_frame_(f);
        p_residual_ = std::fabs(pressure_ - P_target);
        max_force_ = 0;
        const auto fx = f.span(kAtom_fx);
        const auto fy = f.span(kAtom_fy);
        const auto fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            const double ff =
                std::sqrt(fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]);
            max_force_ = std::max(max_force_, ff);
        }
        const double energy = f.meta.get(kEnergy);
        const double dE = has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = max_force_;
            m->rms_force = max_force_;
            m->max_displacement =
                std::max(atom_step, std::fabs(scale_step_ - 1.0));
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
