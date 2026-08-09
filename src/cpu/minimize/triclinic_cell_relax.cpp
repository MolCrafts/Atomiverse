/// @file src/cpu/minimize/triclinic_cell_relax.cpp
#include "atomiverse/cpu/minimize/triclinic_cell_relax.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace minimizer {

TriclinicCellRelax::TriclinicCellRelax(double alpha, double bulk)
    : alpha0(alpha), bulk_mod(bulk) {
    if (!(alpha0 > 0.0) || !(bulk_mod > 0.0) || !(max_strain > 0.0))
        throw std::invalid_argument("TriclinicCellRelax: invalid parameters");
}

void TriclinicCellRelax::build(const BuildContext&) {
    residual_ = 0;
    energy_prev_ = 0;
    step_ = 0;
    has_prev_ = false;
    eps_ = Mat3::zeros();
}

void TriclinicCellRelax::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const double V = f.box.volume();
    if (!(V > 0.0))
        throw std::runtime_error("TriclinicCellRelax: non-positive volume");
    // kStress is the extensive Voigt virial W_ab [eV] (core/frame_schema.h),
    // Voigt order xx, yy, zz, xy, xz, yz. The intensive stress is σ = W/V.
    const auto& s = f.meta.get(kStress);
    auto clamp = [&](double e) {
        return std::max(-max_strain, std::min(max_strain, e));
    };

    if (ctx.at(ATV::minimizer::Phase::search_direction)) {
        // ε_αβ = α σ_αβ / B  with σ = s/V
        const double invVB = alpha0 / (bulk_mod * V);
        // Full symmetric strain from Voigt
        eps_(0, 0) = clamp(invVB * s[0]);
        eps_(1, 1) = clamp(invVB * s[1]);
        eps_(2, 2) = clamp(invVB * s[2]);
        const double exy = clamp(0.5 * invVB * s[3]);
        const double exz = clamp(0.5 * invVB * s[4]);
        const double eyz = clamp(0.5 * invVB * s[5]);
        eps_(0, 1) = eps_(1, 0) = exy;
        eps_(0, 2) = eps_(2, 0) = exz;
        eps_(1, 2) = eps_(2, 1) = eyz;
        residual_ = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                residual_ = std::max(residual_, std::fabs(eps_(a, b)));
        ++step_;
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::line_search) && step_ > 0) {
        // h' = (I+ε) h ; r' = (I+ε) r
        Mat3 h = f.box.cell();
        Mat3 Ipe = Mat3::zeros();
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b)
                Ipe(a, b) = eps_(a, b);
            Ipe(a, a) += 1.0;
        }
        Mat3 hnew = Ipe * h;
        f.box.set_cell(hnew);
        const int n = f.nlocal();
        auto x = f.span(kAtom_x);
        auto y = f.span(kAtom_y);
        auto z = f.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            const double rx = x[i], ry = y[i], rz = z[i];
            x[i] = Ipe(0, 0) * rx + Ipe(0, 1) * ry + Ipe(0, 2) * rz;
            y[i] = Ipe(1, 0) * rx + Ipe(1, 1) * ry + Ipe(1, 2) * rz;
            z[i] = Ipe(2, 0) * rx + Ipe(2, 1) * ry + Ipe(2, 2) * rz;
        }
        return;
    }

    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        const double energy = f.meta.get(kEnergy);
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = energy;
            m->max_force = residual_;
            m->rms_force = residual_;
            m->max_displacement = residual_;
            m->rms_displacement = residual_;
            m->energy_change =
                has_prev_ ? std::fabs(energy - energy_prev_) : 1e30;
            m->has_displacement = has_prev_;
        }
        energy_prev_ = energy;
        has_prev_ = true;
    }
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
