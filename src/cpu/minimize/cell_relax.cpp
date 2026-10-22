/// @file src/cpu/minimize/cell_relax.cpp
#include "atomiverse/cpu/minimize/cell_relax.h"

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

IsotropicCellRelax::IsotropicCellRelax(double P_target_ev_a3, double alpha)
    : P_target(P_target_ev_a3), alpha0(alpha) {
    if (!(alpha > 0.0) || !std::isfinite(alpha) || !std::isfinite(P_target))
        throw std::invalid_argument(
            "IsotropicCellRelax: finite target pressure and positive finite "
            "alpha required");
}

double IsotropicCellRelax::pressure_from_frame_(const Frame& f) const {
    // kStress is the extensive Voigt virial [eV] for every producer, so
    // P = tr(W)/(3V) unconditionally. Kinetic neglected for 0 K geometry
    // optimization (it lives on kStressKinetic).
    const double V = f.box.volume();
    const auto& s = f.meta.get(kStress);
    const double tr = s[0] + s[1] + s[2];
    return tr / (3.0 * V);
}

void IsotropicCellRelax::build(const BuildContext&) {
    pressure_ = 0;
    p_residual_ = 0;
    scale_step_ = 1.0;
    max_disp_ = rms_disp_ = 0;
    energy_prev_ = 0;
    step_ = 0;
    has_prev_ = false;
}

void IsotropicCellRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    // Reject bad input collectively before changing the cell or any private
    // history. In particular, min/max must never hide a NaN pressure as a cap.
    double invalid =
        !std::isfinite(P_target) || !(alpha0 > 0.0) || !std::isfinite(alpha0)
        || !(max_scale >= 0.0 && max_scale < 1.0) || !(bulk_mod > 0.0)
        || !std::isfinite(bulk_mod) || !(f.box.volume() > 0.0)
        || !std::isfinite(f.box.volume());
    for (double stress : f.meta.get(kStress))
        if (!std::isfinite(stress))
            invalid = 1;
    const double pressure = invalid ? 0.0 : pressure_from_frame_(f);
    const double residual = pressure - P_target;
    const double ds_unclamped = invalid ? 0.0 : alpha0 * residual / bulk_mod;
    if (!std::isfinite(pressure) || !std::isfinite(residual)
        || !std::isfinite(ds_unclamped))
        invalid = 1;
    if (global_max(ctx, invalid) != 0)
        throw std::runtime_error(
            "IsotropicCellRelax: invalid parameters, volume, or pressure");

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        // Stress from the previous force phase (0 on first iter).
        pressure_ = pressure;
        p_residual_ = std::fabs(residual);
        // Positive pressure (compressed) → expand: s > 1 when P > P_target.
        const double ds =
            std::max(-max_scale, std::min(max_scale, ds_unclamped));
        scale_step_ = 1.0 + ds;
        ++step_;
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        const double scale = step_ > 0 ? scale_step_ : 1.0;
        if (global_max(ctx,
                       (!(scale > 0.0) || !std::isfinite(scale)) ? 1.0 : 0.0)
            != 0)
            throw std::runtime_error("IsotropicCellRelax: invalid scale");
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        double max_disp = 0, sum_d2 = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = (scale - 1) * x[i];
            const double dy = (scale - 1) * y[i];
            const double dz = (scale - 1) * z[i];
            x[i] += dx;
            y[i] += dy;
            z[i] += dz;
            max_disp = std::max(
                max_disp, std::max({std::abs(dx), std::abs(dy), std::abs(dz)}));
            sum_d2 += dx * dx + dy * dy + dz * dz;
        }
        if (scale != 1.0) {
            f.box.scale_isotropic(scale);
            // Invalidate the old halo; the force boundary rebuilds neighbors
            // and recreates ghosts from the affinely moved owners.
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
        require_finite_forces(ctx);
        pressure_ = pressure;
        p_residual_ = std::fabs(residual);
        const double energy = f.meta.get(kEnergy);
        const double dE = has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
        double max_force = 0, sum_f2 = 0;
        const auto fx = f.span(kAtom_fx);
        const auto fy = f.span(kAtom_fy);
        const auto fz = f.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            const double f2 = fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i];
            max_force = std::max(max_force, std::sqrt(f2));
            sum_f2 += f2;
        }
        max_force = global_max(ctx, max_force);
        double sums[2] = {sum_f2, static_cast<double>(n)};
        global_sum(ctx, sums, 2);
        // These metrics retain their physical units even for a cell-only step.
        // PressureConverged/StressConverged judge the Frame stress separately.
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = max_force;
            m->rms_force = std::sqrt(sums[0] / (3 * std::max(sums[1], 1.0)));
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
