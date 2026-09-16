/// @file regressions/scf/wannier-centers/reg_wannier-centers.cpp
/// @brief Foster–Boys MLWF foothold: H₂O + H₂ / STO-3G RHF geometric pins.
///
/// Out-of-library consumer (public headers + link atomiverse). Pins:
///   - H₂: single center at bond midpoint (unit algebra).
///   - H₂O: 5 centers, total spread decreases, classic Boys geometry
///     (core + rabbit-ear lone pairs + two OH-bond centers).
/// No CP2K binary on the pin host — geometric / unit-algebra pins
/// (perf_exempt). Exit non-zero on failure. ≤ 60 s, single CPU.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/wannier.h"

namespace {

bool near(double a, double b, double tol) { return std::fabs(a - b) < tol; }

double dist(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double d = a[k] - b[k];
        s += d * d;
    }
    return std::sqrt(s);
}

std::array<double, 3> sum_centers(const std::vector<std::array<double, 3>>& c) {
    std::array<double, 3> s = {0.0, 0.0, 0.0};
    for (const auto& p : c)
        for (int k = 0; k < 3; ++k)
            s[k] += p[k];
    return s;
}

int check_h2() {
    using namespace ATV;
    // H—H = 0.74 Å along z; midpoint at z = 0.37 Å.
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.max_iter = 50;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::printf("FAIL H2: SCF not converged\n");
        return 1;
    }

    cpu::scf::WannierCenters wb;
    auto r = wb.compute(rhf.result_handle(), frame);
    if (r.centers.size() != 1u) {
        std::printf("FAIL H2: expected 1 center, got %zu\n", r.centers.size());
        return 1;
    }
    const double mid = 0.37;
    const double err = std::fabs(r.centers[0][2] - mid);
    std::printf("H2 center = (%.6f, %.6f, %.6f) Å  |z-0.37|=%.3e  "
                "spread=%.6f Å²\n",
                r.centers[0][0], r.centers[0][1], r.centers[0][2], err,
                r.spreads[0]);
    if (err > 1e-4 || std::fabs(r.centers[0][0]) > 1e-6
        || std::fabs(r.centers[0][1]) > 1e-6 || !(r.spreads[0] > 0.0)) {
        std::printf("FAIL H2: midpoint / spread pin\n");
        return 1;
    }
    std::printf("H2 PASS\n");
    return 0;
}

int check_h2o() {
    using namespace ATV;
    // Same geometry as h2o_rhf_dipole regression.
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    const std::array<double, 3> O = {0.0, 0.0, 0.1173};
    const std::array<double, 3> H1 = {0.0, 0.7572, -0.4692};
    const std::array<double, 3> H2 = {0.0, -0.7572, -0.4692};
    const std::array<double, 3> mid1 = {
        0.5 * (O[0] + H1[0]), 0.5 * (O[1] + H1[1]), 0.5 * (O[2] + H1[2])};
    const std::array<double, 3> mid2 = {
        0.5 * (O[0] + H2[0]), 0.5 * (O[1] + H2[1]), 0.5 * (O[2] + H2[2])};

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.max_iter = 100;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::printf("FAIL H2O: SCF not converged (iters=%d)\n",
                    rhf.iterations());
        return 1;
    }
    std::printf("H2O RHF E = %.10f Ha  iters=%d\n", rhf.energy(),
                rhf.iterations());

    cpu::scf::WannierCenters wb;
    wb.tol = 1e-12;
    wb.max_sweeps = 200;
    auto r = wb.compute(rhf.result_handle(), frame);

    std::printf("n_centers=%zu  sweeps=%d  conv=%d  "
                "Ω_init=%.6f  Ω_loc=%.6f Å²\n",
                r.centers.size(), r.sweeps, int(r.converged),
                r.total_spread_initial, r.total_spread);

    if (r.centers.size() != 5u || r.spreads.size() != 5u) {
        std::printf("FAIL H2O: expected 5 centers\n");
        return 1;
    }
    if (!(r.total_spread < r.total_spread_initial)) {
        std::printf("FAIL H2O: total spread did not decrease\n");
        return 1;
    }
    for (double s : r.spreads) {
        if (!(s > 0.0) || !std::isfinite(s)) {
            std::printf("FAIL H2O: non-positive/non-finite spread\n");
            return 1;
        }
    }

    // Sort by distance to O for geometric classification.
    std::vector<std::size_t> order(5);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return dist(r.centers[a], O) < dist(r.centers[b], O);
    });

    for (std::size_t k = 0; k < 5; ++k) {
        const auto& c = r.centers[order[k]];
        std::printf("  center[%zu] (%.4f, %.4f, %.4f) Å  |r-O|=%.4f  "
                    "Ω=%.4f\n",
                    k, c[0], c[1], c[2], dist(c, O), r.spreads[order[k]]);
    }

    // Core: closest to O.
    if (dist(r.centers[order[0]], O) > 0.15) {
        std::printf("FAIL H2O: core center too far from O\n");
        return 1;
    }

    // Remaining four: two bond (near OH midpoints, large |y|) and two LP
    // (large |x|, opposite signs).
    std::vector<std::array<double, 3>> rest;
    for (std::size_t k = 1; k < 5; ++k)
        rest.push_back(r.centers[order[k]]);

    // Bond candidates: match each OH midpoint to nearest remaining center.
    auto take_nearest =
        [](std::vector<std::array<double, 3>>& pool,
           const std::array<double, 3>& ref) -> std::array<double, 3> {
        std::size_t best = 0;
        double bd = 1e300;
        for (std::size_t i = 0; i < pool.size(); ++i) {
            const double d = dist(pool[i], ref);
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        auto out = pool[best];
        pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(best));
        return out;
    };

    auto pool = rest;
    const auto b1 = take_nearest(pool, mid1);
    const auto b2 = take_nearest(pool, mid2);
    const double db1 = dist(b1, mid1);
    const double db2 = dist(b2, mid2);
    std::printf("  bond1 |c-midH1|=%.4f  bond2 |c-midH2|=%.4f\n", db1, db2);
    if (db1 > 0.35 || db2 > 0.35) {
        std::printf("FAIL H2O: OH-bond centers too far from midpoints\n");
        return 1;
    }
    // y signs should track the hydrogens.
    if (!(b1[1] * H1[1] > 0.0) || !(b2[1] * H2[1] > 0.0)) {
        std::printf("FAIL H2O: bond centers y-sign mismatch\n");
        return 1;
    }

    // Lone pairs: remaining two — opposite x, |x| > 0.15 Å.
    if (pool.size() != 2u) {
        std::printf("FAIL H2O: expected 2 lone-pair centers left\n");
        return 1;
    }
    const auto& lp1 = pool[0];
    const auto& lp2 = pool[1];
    std::printf("  LP1 x=%.4f  LP2 x=%.4f\n", lp1[0], lp2[0]);
    if (!(std::fabs(lp1[0]) > 0.15) || !(std::fabs(lp2[0]) > 0.15)
        || !(lp1[0] * lp2[0] < 0.0)) {
        std::printf("FAIL H2O: lone-pair rabbit-ear x pattern\n");
        return 1;
    }

    // Sum of centers finite and not NaN (localization preserves electronic
    // dipole / sum of centroids for unitary occupied mix).
    const auto s = sum_centers(r.centers);
    std::printf("  sum centers = (%.6f, %.6f, %.6f) Å\n", s[0], s[1], s[2]);
    if (!std::isfinite(s[0]) || !std::isfinite(s[1]) || !std::isfinite(s[2])) {
        std::printf("FAIL H2O: non-finite center sum\n");
        return 1;
    }
    // Symmetry: Σx ≈ 0, Σy ≈ 0 for C2v water.
    if (!near(s[0], 0.0, 1e-5) || !near(s[1], 0.0, 1e-5)) {
        std::printf("FAIL H2O: center sum broke C2v (x,y)\n");
        return 1;
    }

    if (!r.converged) {
        std::printf("WARN H2O: Jacobi not fully converged (sweeps=%d) — "
                    "geometry pins still required\n",
                    r.sweeps);
        // Still require decrease; do not fail solely on conv if pins pass.
    }

    std::printf("H2O PASS\n");
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= check_h2();
    rc |= check_h2o();
    if (rc == 0)
        std::printf("PASS\n");
    return rc;
}
