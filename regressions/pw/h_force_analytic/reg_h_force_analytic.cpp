/// @file regressions/pw/h_force_analytic/reg_h_force_analytic.cpp
/// @brief END-TO-END: analytic PW forces vanish for a single atom (symmetry).
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.

#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    // eV/Å. A single atom at the origin of a cubic cell: the force is zero by
    // point symmetry, not by convergence, so this is a machine-precision
    // identity and NOT an SCF-quality bound — it holds even though the 8-iter
    // SCF below stops short of `cfg.tol`. Measured 2026-08-15 with GTH-PBE-q1
    // at ecut_ry=5, spin-polarized doublet: |F| = 6.6e-16 for both the
    // analytic and the evaluator path.
    // (The pre-GTH value of 0.5 was sized for the deleted toy potential and
    // was ~14 orders of magnitude too loose to detect a broken force term.)
    const double tol = 1e-9;

    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.box = cpu::Box::cubic(6.0);

    PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    // One valence electron is open-shell: the restricted force path rejects an
    // odd electron count (src/cpu/planewave/internal/pw_forces.cpp:266). A lone
    // H is a doublet (S = 1/2), so N↑−N↓ = 1 ⇒ multiplicity 2; the default 1
    // would fail the parity check in the same guard.
    cfg.spin_polarized = true;
    cfg.multiplicity = 2;

    const auto F = pw_forces_analytic(frame, cfg);
    if (F.size() != 1u) {
        std::printf("FAIL: expected 1 force, got %zu\n", F.size());
        return 1;
    }
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    if (!(mag < tol)) {
        std::printf("FAIL: |F|=%.6g eV/Å  tol=%g (symmetry)\n", mag, tol);
        return 1;
    }

    for (int i = 0; i < 3; ++i)
        frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
            frame.span(kAtom_fz)[0] = 0;
    PWEvaluator pwe{std::move(cfg)};
    pwe.with_gradient(true);
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(pwe);
    eval.enable_timing(false);
    eval.build();
    eval.run(frame);
    if (!std::isfinite(frame.meta.get(kEnergy))) {
        std::printf("FAIL: non-finite energy\n");
        return 1;
    }
    const double mag2 = std::sqrt(static_cast<double>(
        frame.span(kAtom_fx)[0] * frame.span(kAtom_fx)[0]
        + frame.span(kAtom_fy)[0] * frame.span(kAtom_fy)[0]
        + frame.span(kAtom_fz)[0] * frame.span(kAtom_fz)[0]));
    if (!(mag2 < tol)) {
        std::printf("FAIL: evaluator |F|=%.6g eV/Å  tol=%g\n", mag2, tol);
        return 1;
    }

    std::printf("ok: |F|_analytic=%.4g  |F|_eval=%.4g eV/Å  E=%.6g eV\n", mag,
                mag2, frame.meta.get(kEnergy));
    std::printf(
        "PASS: pw analytic forces (single-atom symmetry + evaluator)\n");
    return 0;
}
