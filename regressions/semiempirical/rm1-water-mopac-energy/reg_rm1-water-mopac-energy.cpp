/// @file regressions/semiempirical/rm1-water-mopac-energy/reg_rm1-water-mopac-energy.cpp
/// @brief END-TO-END: RM1 heat of formation of water, pinned to OpenMOPAC.
///
/// See REFERENCE.md for the citations, the geometry convention and the
/// tolerance argument. Public API only: `cpu::Frame` and
/// `cpu::semiempirical::RM1`.
///
/// A MOPAC heat of formation is quoted **at the method's own stationary
/// geometry** (OpenMOPAC `data/README.md`: "the reference geometries are
/// relaxed to the local minimum of the semiempirical model being trained"), so
/// this case relaxes water's two symmetry-unique internal coordinates on the
/// RM1 surface before comparing, and pins the relaxed geometry too. The
/// relaxation is a plain pattern search over `RM1::run` — composing primitives
/// is the caller's job, and it keeps the case independent of whether the
/// minimizer module is built.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

namespace {

/// Published RM1 heat of formation of water [kcal/mol] — OpenMOPAC PARAM
/// survey table `Analysis/Survey organic/RM1.heats`, row `H2O Water`
/// (github.com/openmopac/param @ 9859905, Apache-2.0; the sibling `RM1.dat`
/// is `SURVEY RM1`).
constexpr double kPublishedHofKcal = -57.86;

/// The same quantity from an independent peer-reviewed source [kJ/mol] —
/// L. S. Martins & A. H. Lima, Scientia Plena 12(6), 069903 (2016),
/// Tabela 1 ("Valores em kJ/mol"), RM1 row, H2O column.
constexpr double kPublishedHofKjMol = -242.04;

/// Thermochemical calorie [J] — exact by definition.
constexpr double kJoulePerCalorie = 4.184;

/// Published RM1 stationary geometry of water — OpenMOPAC PARAM survey table
/// `Analysis/Survey organic/RM1.geos`, rows `H2O Water O-H` / `H-O-H`
/// (calculated column; the experimental column is 0.957 Å / 104.500 deg).
constexpr double kPublishedRohAngstrom = 0.956;
constexpr double kPublishedAngleDeg = 103.610;

/// Experimental gas-phase ΔHf(H2O, 298 K) [kcal/mol], JANAF Third Edition
/// (Chase et al. 1986) — the datum RM1 was fitted against, carried in
/// OpenMOPAC `data/PM6-ORG/Core/water.mop` as `H=-57.8000 HR=JANAF86`.
/// Present only so the case can assert that its pin is method-specific.
constexpr double kExperimentalHofKcal = -57.800;

/// Tolerance on ΔHf [kcal/mol]. Justified in REFERENCE.md; must stay well
/// below |published RM1 - experiment| = 0.06 or the pin stops being a pin on
/// *RM1* and becomes a pin on thermochemistry in general.
constexpr double kHofTolKcal = 0.02;

/// @brief Gas-phase water in the xy plane.
/// @param r O-H bond length [Å] (both bonds, C2v).
/// @param theta_deg H-O-H angle [degrees].
/// @return A 3-atom Frame in a 20 Å cubic box (isolation only).
cpu::Frame water(double r, double theta_deg) {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    const double th = theta_deg * M_PI / 180.0;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_x)[2] = r * std::cos(th);
    f.span(kAtom_y)[2] = r * std::sin(th);
    f.span(kAtom_z)[2] = 0.0;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

/// @brief One converged RM1 point.
struct Point {
    bool converged = false;
    double hof_kcal = 0.0;  ///< MOPAC-style heat of formation [kcal/mol]
    double energy_ev = 0.0; ///< NDDO electronic + core energy [eV]
};

/// @brief Run RM1 at one (r, theta).
/// @param r O-H bond length [Å].
/// @param theta_deg H-O-H angle [degrees].
/// @return Converged flag, ΔHf [kcal/mol] and total energy [eV].
Point rm1_point(double r, double theta_deg) {
    cpu::Frame f = water(r, theta_deg);
    RM1 rm1; // concrete method — parameters are members, no *Config bag
    rm1.charge = 0;
    rm1.multiplicity = 1;
    rm1.density_tol = 1.0e-10;
    rm1.max_iter = 400;
    rm1.build(f);
    const RM1Result& res = rm1.run(f);
    Point p;
    p.converged = res.converged;
    p.hof_kcal = res.heat_of_formation_kcal_mol;
    p.energy_ev = res.total_energy_ev;
    return p;
}

} // namespace

int main() {
    // ---- relax (r, theta) on the RM1 surface -------------------------------
    // Start from the experimental geometry (Herzberg 1966), exactly as
    // OpenMOPAC's reference decks do; the relaxed point is method-owned.
    double r = 0.957, theta = 104.5;
    double dr = 0.02, dtheta = 2.0;
    Point best = rm1_point(r, theta);
    if (!best.converged) {
        std::printf("FAIL: RM1 SCF did not converge at the start geometry\n");
        return 1;
    }
    for (int sweep = 0; sweep < 80; ++sweep) {
        bool improved = false;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0)
                    continue;
                const double rr = r + i * dr;
                const double tt = theta + j * dtheta;
                const Point p = rm1_point(rr, tt);
                if (p.converged && p.hof_kcal < best.hof_kcal - 1.0e-12) {
                    best = p;
                    r = rr;
                    theta = tt;
                    improved = true;
                }
            }
        }
        if (!improved) {
            dr *= 0.5;
            dtheta *= 0.5;
            if (dr < 1.0e-6 && dtheta < 1.0e-4)
                break;
        }
    }

    const double published_kcal_from_kj = kPublishedHofKjMol / kJoulePerCalorie;
    std::printf("RM1 water stationary point: r(OH)=%.5f A  angle(HOH)=%.3f deg"
                "   [MOPAC PARAM survey: %.3f A, %.3f deg]\n",
                r, theta, kPublishedRohAngstrom, kPublishedAngleDeg);
    std::printf("  E(NDDO)  = %.6f eV\n", best.energy_ev);
    std::printf("  dHf(RM1) = %.4f kcal/mol"
                "   [RM1.heats %.2f | Scientia Plena %.2f kJ/mol = %.4f]\n",
                best.hof_kcal, kPublishedHofKcal, kPublishedHofKjMol,
                published_kcal_from_kj);

    if (!best.converged || !std::isfinite(best.hof_kcal)
        || !std::isfinite(best.energy_ev)) {
        std::printf("FAIL: RM1 water did not converge to a finite result\n");
        return 1;
    }

    // ---- the pin must be a pin on RM1, not on thermochemistry --------------
    if (!(std::abs(kPublishedHofKcal - kExperimentalHofKcal) > kHofTolKcal)) {
        std::printf("FAIL: tolerance %.3f kcal/mol no longer separates the RM1 "
                    "value from the experimental datum\n",
                    kHofTolKcal);
        return 1;
    }

    // ---- gate 1: OpenMOPAC PARAM survey heat of formation ------------------
    if (!(std::abs(best.hof_kcal - kPublishedHofKcal) <= kHofTolKcal)) {
        std::printf("FAIL: dHf(RM1) = %.4f kcal/mol, OpenMOPAC PARAM survey "
                    "%.2f +- %.2f\n",
                    best.hof_kcal, kPublishedHofKcal, kHofTolKcal);
        return 1;
    }

    // ---- gate 2: the same quantity from an independent publication ---------
    if (!(std::abs(best.hof_kcal - published_kcal_from_kj) <= kHofTolKcal)) {
        std::printf("FAIL: dHf(RM1) = %.4f kcal/mol, Scientia Plena %.2f kJ/mol"
                    " (= %.4f kcal/mol) +- %.2f\n",
                    best.hof_kcal, kPublishedHofKjMol, published_kcal_from_kj,
                    kHofTolKcal);
        return 1;
    }

    // ---- gate 3: the geometry the heat of formation belongs to -------------
    if (!(std::abs(r - kPublishedRohAngstrom) <= 0.002)) {
        std::printf("FAIL: r(OH) = %.5f A, RM1.geos %.3f +- 0.002\n", r,
                    kPublishedRohAngstrom);
        return 1;
    }
    if (!(std::abs(theta - kPublishedAngleDeg) <= 0.10)) {
        std::printf("FAIL: angle(HOH) = %.3f deg, RM1.geos %.3f +- 0.10\n",
                    theta, kPublishedAngleDeg);
        return 1;
    }

    std::printf("PASS: rm1-water-mopac-energy "
                "(dHf %.4f kcal/mol vs published RM1 %.2f; experiment %.3f)\n",
                best.hof_kcal, kPublishedHofKcal, kExperimentalHofKcal);
    return 0;
}
