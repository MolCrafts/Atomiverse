/// @file regressions/semiempirical/am1_water_force/reg_am1_water_force.cpp
/// @brief AM1 water: literature ΔHf-scale energy + analytic force self-consistency.
///
/// Literature / textbook anchors:
/// - Dewar, Zoebisch, Healy, Stewart, J. Am. Chem. Soc. 107, 3902 (1985)
///   (AM1 parameterization). Water experimental ΔHf(gas) ≈ −57.8 kcal/mol;
///   AM1 reproduces small-molecule thermochemistry to a few kcal/mol.
/// - Force check: SCF analytic gradient (fixed-density Hellmann–Feynman NDDO
///   force) must match re-SCF central finite-difference of the total energy
///   (Pulay response vanishes at SCF stationarity for variational methods).
///
/// Units: energy compared after converting total electronic+nuclear energy to
/// a heat-of-formation proxy is NOT used here — we pin the **force** identity
/// and that the total energy is negative and finite (bound molecule).
///
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/facade/stage.h"

namespace {

ATV::cpu::Frame make_water() {
    // Experimental-like geometry (Å), slightly off-equilibrium so |F| ≠ 0.
    ATV::cpu::Frame f(3);
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = 1.05;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -0.26;
    f.span(ATV::kAtom_y)[2] = 1.0;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.simbox = ATV::cpu::Box::cubic(20.0);
    return f;
}

ATV::cpu::semiempirical::AM1Result run_am1(ATV::cpu::Frame frame) {
    ATV::cpu::semiempirical::AM1Solver solver;
    solver.build(frame);
    return solver.run(frame);
}

double total_energy_ev(ATV::cpu::Frame frame) {
    return run_am1(std::move(frame)).total_energy_ev;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = make_water();

    // Public Evaluate path with force-capable AM1Evaluator.
    {
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
        eval.build();
        eval.run(frame);

        const double e = frame.meta.get(kEnergy);
        if (!(e < 0.0) || !std::isfinite(e)) {
            std::fprintf(stderr, "FAIL: AM1 water energy not bound (E=%g eV)\n",
                         e);
            return 1;
        }

        double f2 = 0.0;
        double net[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i) {
            const double fx = frame.span(kAtom_fx)[i];
            const double fy = frame.span(kAtom_fy)[i];
            const double fz = frame.span(kAtom_fz)[i];
            f2 += fx * fx + fy * fy + fz * fz;
            net[0] += fx;
            net[1] += fy;
            net[2] += fz;
        }
        if (f2 < 1.0e-10) {
            std::fprintf(stderr,
                         "FAIL: forces vanished on distorted water (f2=%g)\n",
                         f2);
            return 1;
        }
        for (int d = 0; d < 3; ++d) {
            if (std::abs(net[d]) > 1.0e-5) {
                std::fprintf(stderr, "FAIL: net force component %d = %g eV/Å\n",
                             d, net[d]);
                return 1;
            }
        }
        std::printf("Evaluate path: E=%.6f eV  |F|=%.6f eV/Å  netF ok\n", e,
                    std::sqrt(f2));
    }

    // MOPAC-style heat of formation (Dewar AM1; openmopac calc_of_hof).
    {
        const auto r = run_am1(make_water());
        const double hof = r.heat_of_formation_kcal_mol;
        // Expt. H2O(g) ΔHf ≈ −57.8 kcal/mol; AM1 is typically within a few
        // kcal/mol at near-experimental geometry.
        if (!std::isfinite(hof) || std::abs(hof + 59.0) > 8.0) {
            std::fprintf(stderr,
                         "FAIL: AM1 water ΔHf = %g kcal/mol outside "
                         "[-67, -51] (expect ~-59)\n",
                         hof);
            return 1;
        }
        std::printf("ΔHf(H2O) = %.3f kcal/mol (AM1, MOPAC-style)\n", hof);
    }

    // Analytic vs re-SCF energy finite difference (literature identity:
    // variational SCF gradient ≡ energy derivative).
    {
        auto f0 = make_water();
        cpu::semiempirical::AM1Solver solver;
        solver.build(f0);
        solver.run(f0);
        solver.compute_forces(f0, 1.0e-4);
        const auto& forces = solver.last_result().forces_ev_a;

        constexpr double h = 5.0e-4;
        double max_err = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int d = 0; d < 3; ++d) {
                auto fp = make_water();
                auto fm = make_water();
                StorageReal* cp = d == 0   ? fp.span(kAtom_x)
                                  : d == 1 ? fp.span(kAtom_y)
                                           : fp.span(kAtom_z);
                StorageReal* cm = d == 0   ? fm.span(kAtom_x)
                                  : d == 1 ? fm.span(kAtom_y)
                                           : fm.span(kAtom_z);
                cp[i] =
                    static_cast<StorageReal>(static_cast<double>(cp[i]) + h);
                cm[i] =
                    static_cast<StorageReal>(static_cast<double>(cm[i]) - h);
                const double fd =
                    -(total_energy_ev(fp) - total_energy_ev(fm)) / (2.0 * h);
                const double an = forces[static_cast<std::size_t>(3 * i + d)];
                max_err = std::max(max_err, std::abs(an - fd));
            }
        }
        constexpr double tol = 5.0e-3; // eV/Å
        if (max_err > tol) {
            std::fprintf(stderr,
                         "FAIL: max |F_analytic - F_FD| = %g > %g eV/Å\n",
                         max_err, tol);
            return 1;
        }
        std::printf("PASS: AM1 water force (max|ΔF|=%.3e eV/Å) [Dewar AM1 "
                    "JACS 1985 method; SCF gradient identity]\n",
                    max_err);
    }
    return 0;
}
