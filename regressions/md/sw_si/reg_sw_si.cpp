/// @file regressions/md/sw_si/reg_sw_si.cpp
/// @brief END-TO-END REGRESSION: Stillinger-Weber silicon — cohesive energy,
///        equilibrium lattice constant, and the diamond-over-fcc preference that
///        the three-body term exists to produce.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"

using namespace ATV;

int main() {
    const double a0_ref = 5.431;    // [A]   diamond Si lattice constant
    const double ecoh_ref = 4.3363; // [eV/atom]  SW's fitted cohesive energy
    const int n_cell = 3;           // 8*27 = 216 atoms; box 16.3 A > 2*cutoff

    const cpu::SWPotential pot(molrs::Element::Si, "stillinger-weber");
    const double cutoff = pot.cutoff(); // a*sigma = 3.771 A

    const double diamond[8][3] = {
        {0, 0, 0},     {0.25, 0.25, 0.25}, {0.5, 0.5, 0}, {0.75, 0.75, 0.25},
        {0.5, 0, 0.5}, {0.75, 0.25, 0.75}, {0, 0.5, 0.5}, {0.25, 0.75, 0.75}};

    // Lattice scan of diamond Si.
    const int n_scan = 13;
    double a_scan[n_scan], e_scan[n_scan];
    double best_a = 0.0, best_e = 1e30;

    for (int s = 0; s < n_scan; ++s) {
        const double a = 5.20 + 0.04 * double(s); // 5.20 .. 5.68 A
        const int n = 8 * n_cell * n_cell * n_cell;

        cpu::Frame frame(n);
        int idx = 0;
        for (int ix = 0; ix < n_cell; ++ix)
            for (int iy = 0; iy < n_cell; ++iy)
                for (int iz = 0; iz < n_cell; ++iz)
                    for (int k = 0; k < 8; ++k) {
                        frame.span(ATV::kAtom_x)[idx] =
                            (double(ix) + diamond[k][0]) * a;
                        frame.span(ATV::kAtom_y)[idx] =
                            (double(iy) + diamond[k][1]) * a;
                        frame.span(ATV::kAtom_z)[idx] =
                            (double(iz) + diamond[k][2]) * a;
                        frame.span(ATV::kAtom_mass)[idx] = 28.0855;
                        frame.span(ATV::kAtom_type)[idx] = 14;
                        ++idx;
                    }
        frame.box = ATV::cpu::Box::cubic(a * n_cell);

        if (frame.box.length(0) <= 2.0 * cutoff) {
            std::printf("FAIL: box too small for the cutoff\n");
            return 1;
        }

        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
        eval.on(evaluate::Phase::evaluate).add(cpu::StillingerWeber{pot});
        eval.enable_timing(false);
        eval.build();
        eval.run(frame);

        a_scan[s] = a;
        e_scan[s] = frame.meta.get(ATV::kEnergy) / double(n);
        if (e_scan[s] < best_e) {
            best_e = e_scan[s];
            best_a = a;
        }
    }

    // Parabolic refinement through the three points bracketing the minimum.
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
    const double ecoh = -best_e;

    // The structural test: SW's whole purpose is that silicon prefers the OPEN
    // tetrahedral diamond lattice over a close-packed one, which no pair potential
    // can reproduce — pair potentials always favour close packing. Compare diamond
    // against fcc at the SAME number density, so the comparison is about structure
    // and not about volume.
    const int n_fcc_cell = 4; // 4*64 = 256 atoms
    const double n_density = 8.0 / (a0_ref * a0_ref * a0_ref); // diamond Si
    const double a_fcc = std::cbrt(4.0 / n_density);
    const double fcc[4][3] = {
        {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};

    const int n_f = 4 * n_fcc_cell * n_fcc_cell * n_fcc_cell;
    cpu::Frame ff(n_f);
    int fi = 0;
    for (int ix = 0; ix < n_fcc_cell; ++ix)
        for (int iy = 0; iy < n_fcc_cell; ++iy)
            for (int iz = 0; iz < n_fcc_cell; ++iz)
                for (int k = 0; k < 4; ++k) {
                    ff.span(ATV::kAtom_x)[fi] =
                        (double(ix) + fcc[k][0]) * a_fcc;
                    ff.span(ATV::kAtom_y)[fi] =
                        (double(iy) + fcc[k][1]) * a_fcc;
                    ff.span(ATV::kAtom_z)[fi] =
                        (double(iz) + fcc[k][2]) * a_fcc;
                    ff.span(ATV::kAtom_mass)[fi] = 28.0855;
                    ff.span(ATV::kAtom_type)[fi] = 14;
                    ++fi;
                }
    ff.box = ATV::cpu::Box::cubic(a_fcc * n_fcc_cell);

    cpu::Evaluate eval_fcc;
    eval_fcc.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval_fcc.on(evaluate::Phase::evaluate).add(cpu::StillingerWeber{pot});
    eval_fcc.enable_timing(false);
    eval_fcc.build();
    eval_fcc.run(ff);
    const double e_fcc = ff.meta.get(ATV::kEnergy) / double(n_f);

    const double a_err = std::fabs(a_min - a0_ref) / a0_ref;
    const double e_err = std::fabs(ecoh - ecoh_ref) / ecoh_ref;
    const double tol = 0.01; // 1%: SW is FITTED to both numbers, and it is
                             // strictly cutoff-terminated, so there is no
                             // truncation tail to excuse an error.

    std::printf("SW silicon, diamond, cutoff = %.4f A (= a*sigma), %d atoms\n",
                cutoff, 8 * n_cell * n_cell * n_cell);
    std::printf("  equilibrium a0 = %.4f A     want %.4f A   (err %.2f%%, tol "
                "%.0f%%)\n",
                a_min, a0_ref, 100.0 * a_err, 100.0 * tol);
    std::printf("  cohesive E     = %.4f eV    want %.4f eV  (err %.2f%%, tol "
                "%.0f%%)\n",
                ecoh, ecoh_ref, 100.0 * e_err, 100.0 * tol);
    std::printf("  E(diamond)     = %.4f eV/atom\n", best_e);
    std::printf("  E(fcc, same density) = %.4f eV/atom   [must be HIGHER]\n",
                e_fcc);

    bool ok = true;
    if (!(a_err <= tol)) {
        std::printf("FAIL: lattice constant off by %.2f%%\n", 100.0 * a_err);
        ok = false;
    }
    if (!(e_err <= tol)) {
        std::printf("FAIL: cohesive energy off by %.2f%%\n", 100.0 * e_err);
        ok = false;
    }
    if (!(e_fcc > best_e)) {
        std::printf("FAIL: fcc is not higher in energy than diamond — the "
                    "three-body term is not doing its job\n");
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
