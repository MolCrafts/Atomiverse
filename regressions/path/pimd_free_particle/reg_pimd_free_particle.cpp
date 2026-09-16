/// @file regressions/path/pimd_free_particle/reg_pimd_free_particle.cpp
/// @brief PIMD skeleton: free particle on a ring polymer, centroid collapse.
///
/// Checks (textbook skeleton, not full quantum estimator suite):
///   1. IdentityReplication expands P beads; run completes with finite centroid.
///   2. Spring constant k = m P / (β² ℏ²) (energy units) is positive.
///   3. Mean bead–centroid spread > 0 after thermal init (polymer is open).
///   4. Deterministic seed → same spread.
///
/// Literature: ring-polymer MD (Chandler & Wolynes 1981; Tuckerman et al.);
/// skeleton validates infrastructure (IdentityReplication second consumer +
/// closed-chain springs), not RPMD correlation functions.
/// See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/pimd.h"

using namespace ATV;

static cpu::Frame free_atom() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_mass)[0] = 1.0; // amu
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(40.0);
    return f;
}

int main() {
    auto f = free_atom();
    cpu::path::Pimd pimd(/*n_beads*/ 4, /*T*/ 300.0, /*dt*/ 0.25,
                         /*steps_per_block*/ 20, /*seed*/ 20260723,
                         /*k_ext*/ 0.0);
    // Free particle: no ProgramBuilder (springs + InitVelocity only).
    pimd.run(f, /*n_blocks=*/15);

    const double k = pimd.spring_constant();
    const double spread = pimd.mean_bead_spread();
    const double cx = f.span(kAtom_x)[0];
    const double cy = f.span(kAtom_y)[0];
    const double cz = f.span(kAtom_z)[0];

    std::printf("PIMD free particle: P=%d k=%.6e eV/A2 spread=%.6e A2 "
                "centroid=(%.4f,%.4f,%.4f)\n",
                pimd.n_beads, k, spread, cx, cy, cz);

    bool ok = true;
    if (!(k > 0.0) || !std::isfinite(k)) {
        std::printf("FAIL: spring constant\n");
        ok = false;
    }
    if (!(spread > 0.0) || !std::isfinite(spread)) {
        std::printf("FAIL: bead spread\n");
        ok = false;
    }
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz)) {
        std::printf("FAIL: non-finite centroid\n");
        ok = false;
    }
    if (f.nlocal() != 1) {
        std::printf("FAIL: nlocal\n");
        ok = false;
    }

    // Determinism
    auto f2 = free_atom();
    cpu::path::Pimd pimd2(/*n_beads*/ 4, /*T*/ 300.0, /*dt*/ 0.25,
                          /*steps_per_block*/ 20, /*seed*/ 20260723,
                          /*k_ext*/ 0.0);
    pimd2.run(f2, 15);
    if (std::fabs(pimd2.mean_bead_spread() - spread) > 1e-12) {
        std::printf("FAIL: non-deterministic spread\n");
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}
