/// @file src/cpu/minimize/anisotropic_cell_relax.cpp
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/global_reduce.h"

namespace ATV {
namespace cpu {
namespace minimizer {

AnisotropicCellRelax::AnisotropicCellRelax(double P_target_ev_a3, double alpha)
    : P_target(P_target_ev_a3), alpha0(alpha) {
    if (!(alpha > 0.0) || !std::isfinite(alpha) || !std::isfinite(P_target))
        throw std::invalid_argument("AnisotropicCellRelax: invalid parameters");
}

void AnisotropicCellRelax::build(const BuildContext& ctx) {
    const LaunchContext reduction{ctx.runtime, ctx.frame, 0, 0};
    if (global_max(reduction, (!relax_x && !relax_y && !relax_z) ? 1.0 : 0.0)
        != 0)
        throw std::invalid_argument(
            "AnisotropicCellRelax: every axis is masked (relax_x/y/z all "
            "false)");
    pressure_ = p_residual_ = 0;
    sx_ = sy_ = sz_ = 1;
    max_disp_ = rms_disp_ = 0;
    energy_prev_ = 0;
    step_ = 0;
    has_prev_ = false;
}

void AnisotropicCellRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const double V = f.box.volume();
    const bool relaxed[3] = {relax_x, relax_y, relax_z};
    double invalid =
        !std::isfinite(P_target) || !(alpha0 > 0.0) || !std::isfinite(alpha0)
        || !(max_scale >= 0.0 && max_scale < 1.0) || !(bulk_mod > 0.0)
        || !std::isfinite(bulk_mod) || !(V > 0.0) || !std::isfinite(V)
        || !f.box.is_orthorhombic() || (!relax_x && !relax_y && !relax_z);
    const auto& stress = f.meta.get(kStress);
    for (double value : stress)
        if (!std::isfinite(value))
            invalid = 1;
    double pressures[3]{}, ds[3]{}, residual = 0;
    for (int d = 0; d < 3; ++d) {
        // In the run's pressure unit (LAMMPS fix box/relax: × nktv2p).
        pressures[d] = invalid ? 0.0 : stress[d] / V * ctx.runtime.units.nktv2p;
        if (relaxed[d]) {
            const double delta = pressures[d] - P_target;
            ds[d] = invalid ? 0.0 : alpha0 * delta / bulk_mod;
            if (!std::isfinite(delta) || !std::isfinite(ds[d]))
                invalid = 1;
            residual = std::max(residual, std::fabs(delta));
        }
        if (!std::isfinite(pressures[d]))
            invalid = 1;
    }
    const double pressure = (pressures[0] + pressures[1] + pressures[2]) / 3;
    if (!std::isfinite(pressure))
        invalid = 1;
    // Even a masked-axis NaN is an invalid observable, not zero residual.
    // Every rank rejects before private history or coordinates are changed.
    if (global_max(ctx, invalid) != 0)
        throw std::runtime_error("AnisotropicCellRelax: invalid parameters, "
                                 "orthorhombic cell, or pressure");

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        pressure_ = pressure;
        p_residual_ = residual;
        sx_ = 1 + std::clamp(ds[0], -max_scale, max_scale);
        sy_ = 1 + std::clamp(ds[1], -max_scale, max_scale);
        sz_ = 1 + std::clamp(ds[2], -max_scale, max_scale);
        ++step_;
        return;
    }
    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        const double sx = step_ > 0 ? sx_ : 1.0;
        const double sy = step_ > 0 ? sy_ : 1.0;
        const double sz = step_ > 0 ? sz_ : 1.0;
        const double bad_scale = !(sx > 0) || !std::isfinite(sx) || !(sy > 0)
                                 || !std::isfinite(sy) || !(sz > 0)
                                 || !std::isfinite(sz);
        if (global_max(ctx, bad_scale) != 0)
            throw std::runtime_error("AnisotropicCellRelax: invalid scale");
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        double max_disp = 0, sum_d2 = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = (sx - 1) * x[i];
            const double dy = (sy - 1) * y[i];
            const double dz = (sz - 1) * z[i];
            x[i] += dx;
            y[i] += dy;
            z[i] += dz;
            max_disp = std::max(
                max_disp, std::max({std::abs(dx), std::abs(dy), std::abs(dz)}));
            sum_d2 += dx * dx + dy * dy + dz * dz;
        }
        if (sx != 1.0 || sy != 1.0 || sz != 1.0) {
            f.box.scale_diagonal(sx, sy, sz);
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
        p_residual_ = residual;
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
        // StressConverged judges the cell objective separately. These remain
        // atomic force [eV/Å] and actual affine displacement [Å] metrics.
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
