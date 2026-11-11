/// @file src/cpu/minimize/joint_cell_relax.cpp
#include "atomiverse/cpu/minimize/joint_cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/global_reduce.h"

namespace ATV {
namespace cpu {
namespace minimizer {

JointIsotropicRelax::JointIsotropicRelax(double P_target_ev_a3,
                                         double atom_step_a)
    : P_target(P_target_ev_a3), atom_step(atom_step_a) {
    if (!(atom_step > 0.0) || !std::isfinite(atom_step)
        || !std::isfinite(P_target))
        throw std::invalid_argument(
            "JointIsotropicRelax: finite target pressure and positive finite "
            "atom_step required");
}

double JointIsotropicRelax::pressure_from_frame_(const Frame& f) {
    const double V = f.box.volume();
    const auto& s = f.meta.get(kStress);
    return (s[0] + s[1] + s[2]) / (3.0 * V);
}

void JointIsotropicRelax::build(const BuildContext&) {
    pressure_ = 0;
    p_residual_ = 0;
    max_force_ = 0;
    max_disp_ = rms_disp_ = 0;
    scale_step_ = 1.0;
    energy_prev_ = 0;
    has_prev_ = false;
    step_ = 0;
}

void JointIsotropicRelax::launch(const LaunchContext& ctx) {
    require_finite_forces(ctx);
    auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Public parameters may change after build. Validate before any state or
    // geometry writes, and reduce the error flag before throwing on any rank.
    double invalid =
        !std::isfinite(P_target) || !(atom_step > 0.0)
        || !std::isfinite(atom_step) || !(alpha0 >= 0.0)
        || !std::isfinite(alpha0) || !(max_scale >= 0.0 && max_scale < 1.0)
        || !(bulk_mod > 0.0) || !std::isfinite(bulk_mod)
        || !(f.box.volume() > 0.0) || !std::isfinite(f.box.volume());
    for (double stress : f.meta.get(kStress))
        if (!std::isfinite(stress))
            invalid = 1;
    // In the run's pressure unit (LAMMPS fix box/relax: × nktv2p).
    const double pressure =
        invalid ? 0.0 : pressure_from_frame_(f) * ctx.runtime.units.nktv2p;
    const double residual = pressure - P_target;
    const double ds_unclamped = invalid ? 0.0 : alpha0 * residual / bulk_mod;
    if (!std::isfinite(pressure) || !std::isfinite(residual)
        || !std::isfinite(ds_unclamped))
        invalid = 1;
    if (global_max(ctx, invalid) != 0)
        throw std::runtime_error(
            "JointIsotropicRelax: invalid parameters, volume, or pressure");

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        pressure_ = pressure;
        p_residual_ = std::fabs(residual);
        const double ds =
            std::max(-max_scale, std::min(max_scale, ds_unclamped));
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
        max_force_ = global_max(ctx, max_force_);
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
        const double scale = step_ > 0 ? scale_step_ : 1.0;
        if (global_max(ctx,
                       (!(scale > 0.0) || !std::isfinite(scale)) ? 1.0 : 0.0)
            != 0)
            throw std::runtime_error("JointIsotropicRelax: invalid scale");
        double max_disp = 0, sum_d2 = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = (scale - 1) * x[i] + scale * inv * fx[i];
            const double dy = (scale - 1) * y[i] + scale * inv * fy[i];
            const double dz = (scale - 1) * z[i] + scale * inv * fz[i];
            x[i] += dx;
            y[i] += dy;
            z[i] += dz;
            max_disp = std::max(
                max_disp, std::max({std::abs(dx), std::abs(dy), std::abs(dz)}));
            sum_d2 += dx * dx + dy * dy + dz * dz;
        }
        if (scale != 1.0) {
            f.box.scale_isotropic(scale);
            ctx.runtime.md.reneighbor = true;
            if (multi_rank_dd(ctx.runtime))
                ctx.runtime.md.dd->compute_sublo_subhi(f.box);
        }
        max_disp_ = global_max(ctx, max_disp);
        double sums[2] = {sum_d2, static_cast<double>(n)};
        global_sum(ctx, sums, 2);
        rms_disp_ = std::sqrt(sums[0] / (3 * std::max(sums[1], 1.0)));
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        pressure_ = pressure;
        p_residual_ = std::fabs(residual);
        max_force_ = 0;
        double sum_f2 = 0;
        const auto fx = f.span(kAtom_fx);
        const auto fy = f.span(kAtom_fy);
        const auto fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            const double ff =
                std::sqrt(fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]);
            max_force_ = std::max(max_force_, ff);
            sum_f2 += ff * ff;
        }
        max_force_ = global_max(ctx, max_force_);
        double sums[2] = {sum_f2, static_cast<double>(n)};
        global_sum(ctx, sums, 2);
        const double rms_force =
            std::sqrt(sums[0] / (3 * std::max(sums[1], 1.0)));
        const double energy = f.meta.get(kEnergy);
        const double dE = has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = max_force_;
            m->rms_force = rms_force;
            m->max_displacement = max_disp_;
            m->rms_displacement = rms_disp_;
            m->energy_change = dE;
            m->has_displacement = step_ > 0;
        }
        energy_prev_ = energy;
        has_prev_ = true;
    }
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
