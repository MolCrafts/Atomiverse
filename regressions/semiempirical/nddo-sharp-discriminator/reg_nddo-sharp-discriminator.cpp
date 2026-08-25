/// @file regressions/semiempirical/nddo-sharp-discriminator/reg_nddo-sharp-discriminator.cpp
/// @brief RM1 CO₂ / CO — OpenMOPAC PARAM survey pins that discriminate the method.
///
/// Oracle: **OpenMOPAC** (NDDO family lock). Numbers are transcribed from
/// github.com/openmopac/param @ 9859905 `Analysis/Survey organic/RM1.heats`
/// and `RM1.geos` — the same survey as `reg_rm1-water-mopac-energy`. Water's
/// RM1 residual vs experiment is 0.06 kcal/mol; CO₂'s is 12.50. ctest never
/// shells out to mopac.
///
/// Exit non-zero on failure.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"

namespace {

using ATV::cpu::Frame;
using ATV::cpu::semiempirical::RM1;
using ATV::cpu::semiempirical::RM1Result;
using ATV::kAtom_type;
using ATV::kAtom_x;
using ATV::kAtom_y;
using ATV::kAtom_z;
using ATV::StorageReal;

// OpenMOPAC PARAM survey RM1.heats / RM1.geos @ 9859905.
constexpr double kCo2ExpHofKcal = -94.05;
constexpr double kCo2Rm1HofKcal = -81.55;
constexpr double kCo2Rm1RAngstrom = 1.173;
constexpr double kCoExpHofKcal = -26.42;
constexpr double kCoRm1HofKcal = -13.87;
constexpr double kCoRm1RAngstrom = 1.138;

constexpr double kHofTolKcal = 0.05;
constexpr double kRTolAngstrom = 0.005;
constexpr double kForceTolEvA = 0.05;
constexpr double kDiscriminatorFloorKcal = 10.0;

struct Point {
    bool converged = false;
    double hof_kcal = 0.0;
    double energy_ev = 0.0;
};

Frame linear_co2(double r_co) {
    Frame f(3);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_type)[1] = 8;
    f.span(kAtom_type)[2] = 8;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_co);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = static_cast<StorageReal>(-r_co);
    f.span(kAtom_y)[2] = 0;
    f.span(kAtom_z)[2] = 0;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

Frame diatomic_co(double r_co) {
    Frame f(2);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_type)[1] = 8;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = static_cast<StorageReal>(r_co);
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

Point rm1_on(const Frame& geom) {
    RM1 rm1;
    rm1.charge = 0;
    rm1.multiplicity = 1;
    rm1.density_tol = 1.0e-10;
    rm1.max_iter = 400;
    rm1.build(geom);
    const RM1Result& res = rm1.run(geom);
    Point p;
    p.converged = res.converged;
    p.hof_kcal = res.heat_of_formation_kcal_mol;
    p.energy_ev = res.total_energy_ev;
    return p;
}

/// @brief 1-D pattern search of r on the RM1 ΔHf surface.
double relax_r(Frame (*make)(double), double r, Point* best) {
    double dr = 0.02;
    *best = rm1_on(make(r));
    for (int sweep = 0; sweep < 80; ++sweep) {
        bool improved = false;
        for (int i : {-1, 1}) {
            const double rr = r + static_cast<double>(i) * dr;
            if (!(rr > 0.5 && rr < 2.0))
                continue;
            const Point p = rm1_on(make(rr));
            if (p.converged && p.hof_kcal < best->hof_kcal - 1.0e-12) {
                *best = p;
                r = rr;
                improved = true;
            }
        }
        if (!improved) {
            dr *= 0.5;
            if (dr < 1.0e-6)
                break;
        }
    }
    return r;
}

double max_force(Frame& geom) {
    RM1 rm1;
    rm1.charge = 0;
    rm1.multiplicity = 1;
    rm1.density_tol = 1.0e-10;
    rm1.max_iter = 400;
    rm1.build(geom);
    rm1.run(geom);
    const auto& fr = rm1.compute_forces(geom);
    double mx = 0.0;
    const std::size_t n = fr.forces_ev_a.size() / 3;
    for (std::size_t a = 0; a < n; ++a) {
        const double fx = fr.forces_ev_a[3 * a + 0];
        const double fy = fr.forces_ev_a[3 * a + 1];
        const double fz = fr.forces_ev_a[3 * a + 2];
        mx = std::max(mx, std::hypot(fx, fy, fz));
    }
    return mx;
}

} // namespace

int main() {
    if (!(std::abs(kCo2Rm1HofKcal - kCo2ExpHofKcal) > kDiscriminatorFloorKcal)) {
        std::printf("FAIL: published RM1 CO2 is no longer a discriminator vs experiment\n");
        return 1;
    }
    if (!(std::abs(kCoRm1HofKcal - kCoExpHofKcal) > kDiscriminatorFloorKcal)) {
        std::printf("FAIL: published RM1 CO is no longer a discriminator vs experiment\n");
        return 1;
    }

    Point co2;
    const double r_co2 = relax_r(linear_co2, 1.162, &co2);
    std::printf("RM1 CO2: r(C-O)=%.5f A  dHf=%.4f kcal/mol  E=%.6f eV\n",
                r_co2, co2.hof_kcal, co2.energy_ev);
    if (!co2.converged || !std::isfinite(co2.hof_kcal)) {
        std::printf("FAIL: RM1 CO2 did not converge\n");
        return 1;
    }
    if (!(std::abs(co2.hof_kcal - kCo2Rm1HofKcal) <= kHofTolKcal)) {
        std::printf("FAIL: dHf(CO2)=%.4f vs PARAM RM1 %.2f +- %.2f\n",
                    co2.hof_kcal, kCo2Rm1HofKcal, kHofTolKcal);
        return 1;
    }
    if (!(std::abs(co2.hof_kcal - kCo2ExpHofKcal) > kDiscriminatorFloorKcal)) {
        std::printf("FAIL: ATV RM1 CO2 (%.4f) collapsed onto experiment (%.2f)\n",
                    co2.hof_kcal, kCo2ExpHofKcal);
        return 1;
    }
    if (!(std::abs(r_co2 - kCo2Rm1RAngstrom) <= kRTolAngstrom)) {
        std::printf("FAIL: r(CO2)=%.5f vs RM1.geos %.3f +- %.3f\n",
                    r_co2, kCo2Rm1RAngstrom, kRTolAngstrom);
        return 1;
    }
    Frame co2_geom = linear_co2(r_co2);
    const double f_co2 = max_force(co2_geom);
    if (!(f_co2 < kForceTolEvA)) {
        std::printf("FAIL: |F|_max(CO2)=%.4f eV/A (not stationary)\n", f_co2);
        return 1;
    }

    Point co;
    const double r_co = relax_r(diatomic_co, 1.128, &co);
    std::printf("RM1 CO : r(C-O)=%.5f A  dHf=%.4f kcal/mol  E=%.6f eV\n",
                r_co, co.hof_kcal, co.energy_ev);
    if (!co.converged || !std::isfinite(co.hof_kcal)) {
        std::printf("FAIL: RM1 CO did not converge\n");
        return 1;
    }
    if (!(std::abs(co.hof_kcal - kCoRm1HofKcal) <= kHofTolKcal)) {
        std::printf("FAIL: dHf(CO)=%.4f vs PARAM RM1 %.2f +- %.2f\n",
                    co.hof_kcal, kCoRm1HofKcal, kHofTolKcal);
        return 1;
    }
    if (!(std::abs(r_co - kCoRm1RAngstrom) <= kRTolAngstrom)) {
        std::printf("FAIL: r(CO)=%.5f vs RM1.geos %.3f +- %.3f\n",
                    r_co, kCoRm1RAngstrom, kRTolAngstrom);
        return 1;
    }

    std::printf("PASS: nddo-sharp-discriminator "
                "(CO2 dHf %.4f vs RM1 %.2f, exp %.2f; CO dHf %.4f vs RM1 %.2f)\n",
                co2.hof_kcal, kCo2Rm1HofKcal, kCo2ExpHofKcal, co.hof_kcal,
                kCoRm1HofKcal);
    return 0;
}
