/// @file regressions/pw/h_force_fd/reg_h_force_fd.cpp
/// @brief END-TO-END: plane-wave FD forces vanish for a single atom (symmetry).
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_evaluator.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    const double tol = 0.5; // eV/Å — residual of toy-PP + short SCF

    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.simbox = cpu::Box::cubic(6.0);

    PWConfig cfg;
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;

    const auto F = pw_forces_fd(frame, cfg, 5e-3);
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

    // Public evaluator path.
    for (int i = 0; i < 3; ++i)
        frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
            frame.span(kAtom_fz)[0] = 0;
    PWEvaluator pwe{cfg};
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
    if (!(mag2 < tol)) {
        std::printf("FAIL: evaluator |F|=%.6g eV/Å  tol=%g\n", mag2, tol);
        return 1;
    }

    std::printf("ok: |F|_fd=%.4g  |F|_eval=%.4g eV/Å  E=%.6g eV\n", mag, mag2,
                frame.meta.get(kEnergy));
    std::printf("PASS: pw FD forces (single-atom symmetry + evaluator)\n");
    return 0;
}
