/// @file regressions/pw/h_force_fd/reg_h_force_fd.cpp
/// @brief END-TO-END: plane-wave FD forces vanish for a single atom (symmetry).
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
    // eV/Å. A single atom at the origin of a cubic cell: the force vanishes by
    // point symmetry, not by convergence, so both bounds below are symmetry
    // identities rather than SCF-quality bounds.
    //
    // The two paths have genuinely different floors, so they get different
    // bounds instead of one blanket number:
    //   * tol_fd    — central differences with the 5e-3 Å step below, so the
    //                 floor is set by the FD step and the 1e-4 SCF tolerance,
    //                 not by machine epsilon. Measured 2026-08-15: 1.4e-11.
    //   * tol_eval  — the evaluator path returns the analytic gradient here.
    //                 Measured 2026-08-15: 6.6e-16.
    // The pre-GTH value of 0.5 was sized for the deleted toy potential and was
    // far too loose to detect a dropped or asymmetric force term.
    const double tol_fd = 1e-4;
    const double tol_eval = 1e-9;

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

    const auto F = pw_forces_fd(frame, cfg, 5e-3);
    if (F.size() != 1u) {
        std::printf("FAIL: expected 1 force, got %zu\n", F.size());
        return 1;
    }
    const double mag =
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]);
    if (!(mag < tol_fd)) {
        std::printf("FAIL: |F|=%.6g eV/Å  tol=%g (symmetry)\n", mag, tol_fd);
        return 1;
    }

    // Public evaluator path.
    for (int i = 0; i < 3; ++i)
        frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
            frame.span(kAtom_fz)[0] = 0;
    PWEvaluator pwe{std::move(cfg)};
    pwe.with_gradient(true).set_fd_step(5e-3);
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
    if (!(mag2 < tol_eval)) {
        std::printf("FAIL: evaluator |F|=%.6g eV/Å  tol=%g\n", mag2, tol_eval);
        return 1;
    }

    std::printf("ok: |F|_fd=%.4g  |F|_eval=%.4g eV/Å  E=%.6g eV\n", mag, mag2,
                frame.meta.get(kEnergy));
    std::printf("PASS: pw FD forces (single-atom symmetry + evaluator)\n");
    return 0;
}
