/// @file regressions/md/eam_cu/reg_eam_cu.cpp
/// @brief END-TO-END REGRESSION: EAM copper — cohesive energy and equilibrium
///        lattice constant against the values the potential was FITTED to.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/eam.h"
#include "atomiverse/cpu/potentials/eam_potential.h"

using namespace ATV;

int main() {
    // Sutton-Chen fitted copper: E_coh = 3.54 eV/atom at a0 = 3.61 A (fcc).
    const double a0_ref = 3.61;   // [A]
    const double ecoh_ref = 3.54; // [eV/atom], binding (positive by convention)
    const double cutoff = 10.0;   // [A] — the SC power law needs a long cutoff
    const int n_cell = 6;         // box = 6*a0 = 21.7 A > 2*cutoff (min image)

    // Energy per atom of fcc Cu at lattice constant a, over a scan. The scan is
    // the point of the case, so this loop is the case — no helper function.
    const int n_scan = 13;
    double a_scan[n_scan], e_scan[n_scan];
    double best_a = 0.0, best_e = 1e30;

    for (int s = 0; s < n_scan; ++s) {
        const double a = 3.45 + 0.025 * double(s); // 3.45 .. 3.75 A
        const int n = 4 * n_cell * n_cell * n_cell;
        const double basis[4][3] = {
            {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};

        cpu::Frame frame(n);
        int idx = 0;
        for (int ix = 0; ix < n_cell; ++ix)
            for (int iy = 0; iy < n_cell; ++iy)
                for (int iz = 0; iz < n_cell; ++iz)
                    for (int k = 0; k < 4; ++k) {
                        frame.span(ATV::kAtom_x)[idx] =
                            (double(ix) + basis[k][0]) * a;
                        frame.span(ATV::kAtom_y)[idx] =
                            (double(iy) + basis[k][1]) * a;
                        frame.span(ATV::kAtom_z)[idx] =
                            (double(iz) + basis[k][2]) * a;
                        frame.span(ATV::kAtom_mass)[idx] = 63.546;
                        frame.span(ATV::kAtom_type)[idx] = 29;
                        ++idx;
                    }
        frame.simbox = ATV::cpu::Box::cubic(a * n_cell);

        if (frame.simbox.length(0) <= 2.0 * cutoff) {
            std::printf("FAIL: box %.2f A too small for cutoff %.2f A\n",
                        frame.simbox.length(0), cutoff);
            return 1;
        }

        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"},
                          cutoff});
        eval.enable_timing(false);
        eval.build();
        eval.run(frame);

        a_scan[s] = a;
        e_scan[s] = frame.meta.get(ATV::kEnergy) / double(n); // [eV/atom]
        if (e_scan[s] < best_e) {
            best_e = e_scan[s];
            best_a = a;
        }
    }

    // Parabolic refinement of the minimum through the three points around it.
    double a_min = best_a;
    for (int s = 1; s < n_scan - 1; ++s) {
        if (a_scan[s] != best_a)
            continue;
        const double y0 = e_scan[s - 1], y1 = e_scan[s], y2 = e_scan[s + 1];
        const double denom = y0 - 2.0 * y1 + y2;
        if (std::fabs(denom) > 1e-12) {
            const double da = a_scan[s] - a_scan[s - 1];
            a_min = a_scan[s] + 0.5 * da * (y0 - y2) / denom;
        }
    }

    const double ecoh = -best_e; // binding energy, positive
    const double a_err = std::fabs(a_min - a0_ref) / a0_ref;
    const double e_err = std::fabs(ecoh - ecoh_ref) / ecoh_ref;

    // Tolerances: 2% — the potential is FITTED to these two numbers, so the only
    // things separating us from them are the finite cutoff (the Sutton-Chen power
    // law has no intrinsic range) and the discrete lattice scan. See REFERENCE.md.
    const double tol = 0.02;

    std::printf("EAM Cu (Sutton-Chen), fcc, cutoff = %.1f A, %d atoms\n",
                cutoff, 4 * n_cell * n_cell * n_cell);
    std::printf("  equilibrium a0 = %.4f A     want %.4f A   (err %.2f%%, tol "
                "%.0f%%)\n",
                a_min, a0_ref, 100.0 * a_err, 100.0 * tol);
    std::printf("  cohesive E     = %.4f eV    want %.4f eV  (err %.2f%%, tol "
                "%.0f%%)\n",
                ecoh, ecoh_ref, 100.0 * e_err, 100.0 * tol);

    bool ok = true;
    if (!(a_err <= tol)) {
        std::printf("FAIL: equilibrium lattice constant off by %.2f%%\n",
                    100.0 * a_err);
        ok = false;
    }
    if (!(e_err <= tol)) {
        std::printf("FAIL: cohesive energy off by %.2f%%\n", 100.0 * e_err);
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
