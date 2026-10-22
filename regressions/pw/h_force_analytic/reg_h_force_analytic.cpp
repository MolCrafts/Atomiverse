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
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"

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

    PW cfg{GthPseudopotential{"GTH-PBE-q1"}, 5.0, 1};
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    // One valence electron is open-shell: the restricted force path rejects an
    // odd electron count (`PW::forces` throws). A lone
    // H is a doublet (S = 1/2), so N↑−N↓ = 1 ⇒ multiplicity 2; the default 1
    // would fail the parity check in the same guard.
    cfg.spin_polarized = true;
    cfg.multiplicity = 2;

    cfg.forces(frame); // writes kAtom_fx/fy/fz [eV/Å] + kEnergy
    const double F0[3] = {static_cast<double>(frame.span(kAtom_fx)[0]),
                          static_cast<double>(frame.span(kAtom_fy)[0]),
                          static_cast<double>(frame.span(kAtom_fz)[0])};
    const double mag = std::sqrt(F0[0] * F0[0] + F0[1] * F0[1] + F0[2] * F0[2]);
    if (!(mag < tol)) {
        std::printf("FAIL: |F|=%.6g eV/Å  tol=%g (symmetry)\n", mag, tol);
        return 1;
    }

    for (int i = 0; i < 3; ++i)
        frame.span(kAtom_fx)[0] = frame.span(kAtom_fy)[0] =
            frame.span(kAtom_fz)[0] = 0;
    // The method itself, nested in an Evaluate: energy and forces accumulate.
    PW& pwe = cfg;
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
        std::printf("FAIL: nested |F|=%.6g eV/Å  tol=%g\n", mag2, tol);
        return 1;
    }

    const auto same_result = [&]() {
        return std::abs(frame.meta.get(kEnergy)
                        - pwe.last_result().energy * qe_units::rydberg_to_ev)
               < 1e-12;
    };
    if (!same_result()) {
        std::printf("FAIL: combined outputs do not publish the owned result\n");
        return 1;
    }
    // A changed cell must replace the public result too.
    frame.box = cpu::Box::cubic(7.0);
    eval.run(frame);
    if (!same_result() || pwe.last_result().iterations <= 0) {
        std::printf("FAIL: changed cell left a stale owned result\n");
        return 1;
    }

    // Exercise both metallic occupation routes against the existing displaced-
    // geometry oracle. Keep the established one-atom symmetry gates: analytic
    // 1e-9 eV/A here, FD 1e-4 eV/A at h=5e-3 in h_force_fd. This is an
    // occupation/state-reuse gate, not coverage of nonzero bond or KB forces.
    for (const bool spin : {false, true}) {
        cpu::Frame metallic = frame;
        metallic.box = cpu::Box::cubic(6.0);
        PW smear{GthPseudopotential{"GTH-PBE-q1"}, 5.0, 1};
        smear.tol = 1e-8;
        smear.max_iter = 100;
        smear.mix_beta = 0.5;
        smear.spin_polarized = spin;
        smear.multiplicity = spin ? 2 : 1;
        smear.smearing = Smearing::fermi_dirac;
        smear.degauss = 0.02;
        smear.extra_bands = 2;
        const auto fd = pw_forces_fd(metallic, smear, 5e-3);
        // Restricted metallic stress is supported; spin+smear stress is not.
        std::array<double, 6> stress_fd{};
        if (!spin)
            stress_fd = pw_stress_fd(metallic, smear, 2e-3);
        PW& metallic_eval = smear;
        metallic_eval.forces(metallic);
        if (!spin)
            metallic_eval.stress(metallic);
        if (!metallic_eval.last_result().converged || fd.size() != 1u) {
            std::printf("FAIL: metallic spin=%d state did not converge\n",
                        spin);
            return 1;
        }
        const double force[3] = {metallic.span(kAtom_fx)[0],
                                 metallic.span(kAtom_fy)[0],
                                 metallic.span(kAtom_fz)[0]};
        for (int k = 0; k < 3; ++k) {
            if (!(std::abs(force[k]) < tol) || !(std::abs(fd[0][k]) < 1e-4)
                || !(std::abs(force[k] - fd[0][k]) < 1e-4)) {
                std::printf("FAIL: metallic spin=%d force[%d]=%g FD=%g\n", spin,
                            k, force[k], fd[0][k]);
                return 1;
            }
        }
        if (!spin) {
            const auto& virial = metallic.meta.get(kStress);
            for (int k = 0; k < 6; ++k) {
                const double stress = virial[k] / metallic.box.volume();
                // Same live strain step and dual-gate bound as the existing
                // pw-analytic-stress-si FD pin (0.05 eV/A^3). Symmetry has the
                // stricter independent off-diagonal bound from h_stress_cubic.
                if (!(std::abs(stress - stress_fd[k]) < 0.05)
                    || (k >= 3 && !(std::abs(stress) < 1e-9))) {
                    std::printf("FAIL: metallic stress[%d]=%g FD=%g\n", k,
                                stress, stress_fd[k]);
                    return 1;
                }
            }
        }
        if (!(std::abs(metallic.meta.get(kEnergy)
                       - metallic_eval.last_result().energy
                             * qe_units::rydberg_to_ev)
              < 1e-12)) {
            std::printf("FAIL: metallic spin=%d result ownership\n", spin);
            return 1;
        }
    }

    std::printf("ok: |F|_analytic=%.4g  |F|_eval=%.4g eV/Å  E=%.6g eV\n", mag,
                mag2, frame.meta.get(kEnergy));
    std::printf(
        "PASS: pw analytic forces (single-atom symmetry + nested method)\n");
    return 0;
}
