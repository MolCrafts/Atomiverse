/// @file regressions/dynamics/xl-bomd-linearized-forces/reg_xl-bomd-linearized-forces.cpp
/// @brief END-TO-END: zero-SCF linearized-functional XL-BOMD NVE on H₂O
///        with GPW shadow forces — energy-drift + iters==0 pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Isolated water, CP2K getting-started geometry (Å) — same twin family
    // as xl-bomd-water, with zero-SCF linearized shadow forces engaged.
    cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double x0[3] = {5.0000, 5.7586, 4.2414};
    const double y0[3] = {5.0000, 5.0000, 5.0000};
    const double z0[3] = {5.0000, 5.5043, 5.5043};
    const double mass[3] = {15.999, 1.008, 1.008};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_mass)[i] = mass[i];
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = y0[i];
        frame.span(kAtom_z)[i] = z0[i];
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // COM-preserving H stretch mode [Å/fs].
    frame.span(kAtom_vx)[1] = 0.0015;
    frame.span(kAtom_vx)[2] = -0.0015;
    frame.box = cpu::Box::cubic(10.0);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator gpw_base(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    cpu::electronic::XlBomdForce eval{std::move(gpw_base), 5, cpu::electronic::XlBomdLinearized{}};
    constexpr double dt = 0.5; // fs
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // Shadow PE + nuclear KE: looser than full-SCF XL (0.05) — one-shot
    // residual heat; still catches sign/shape bugs immediately.
    constexpr double drift_tol = 0.10;
    constexpr double oh_max = 1.8; // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold full SCF seeds the XL history.
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: warm-up step did not run\n");
        return 1;
    }
    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !(ke0 >= 0.0)) {
        std::printf("FAIL: non-finite energy after warm-up PE=%g KE=%g\n", pe0,
                    ke0);
        return 1;
    }
    if (!(ke0 > 1e-12)) {
        std::printf("FAIL: expected non-trivial KE after warm-up, got %g eV\n",
                    ke0);
        return 1;
    }
    if (!(pe0 < -100.0)) {
        std::printf("FAIL: PE not bound after warm-up PE=%g eV\n", pe0);
        return 1;
    }
    // Cold seed still does iterative SCF (iters > 0).
    if (!(eval.last_result().iters > 0) || !eval.last_result().converged) {
        std::printf("FAIL: warm-up should be full SCF seed (iters=%d "
                    "converged=%d)\n",
                    eval.last_result().iters,
                    static_cast<int>(eval.last_result().converged));
        return 1;
    }
    const double e0 = pe0 + ke0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    int max_warm_iters = 0;
    const int steps_per = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (sim.run(frame, steps_per) != steps_per) {
            std::printf("FAIL: production chunk %d incomplete\n", c);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite energy at chunk %d PE=%g KE=%g\n", c,
                        pe, ke);
            return 1;
        }
        // Warm linearized steps: zero iterative SCF.
        if (eval.last_result().iters != 0) {
            std::printf("FAIL: warm linearized step had iters=%d (want 0) "
                        "at chunk %d\n",
                        eval.last_result().iters, c);
            return 1;
        }
        if (!eval.last_result().converged) {
            std::printf("FAIL: shadow step not marked converged at chunk %d\n",
                        c);
            return 1;
        }
        max_warm_iters = std::max(max_warm_iters, eval.last_result().iters);
        const double drift = std::fabs((pe + ke) - e0) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;

        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[i]))
                || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[i]))
                || !std::isfinite(
                    static_cast<double>(frame.span(kAtom_z)[i]))) {
                std::printf("FAIL: non-finite coordinate at chunk %d atom %d\n",
                            c, i);
                return 1;
            }
        }
    }

    auto oh = [&](int h) {
        const double dx = static_cast<double>(frame.span(kAtom_x)[h]
                                              - frame.span(kAtom_x)[0]);
        const double dy = static_cast<double>(frame.span(kAtom_y)[h]
                                              - frame.span(kAtom_y)[0]);
        const double dz = static_cast<double>(frame.span(kAtom_z)[h]
                                              - frame.span(kAtom_z)[0]);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    const double oh1 = oh(1);
    const double oh2 = oh(2);
    if (!(oh1 < oh_max) || !(oh2 < oh_max)) {
        std::printf("FAIL: O–H bond exploded oh1=%.3f oh2=%.3f Å (max %g)\n",
                    oh1, oh2, oh_max);
        return 1;
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    std::printf("XL-BOMD linearized H2O NVE (K=5): steps=%d  E0=%.6g eV  "
                "Ef=%.6g eV  worst |dE|/scale=%.3e  tol=%.2f  "
                "warm_iters=%d  oh=%.3f/%.3f Å\n",
                n_prod, e0, pe_f + ke_f, worst_drift, drift_tol, max_warm_iters,
                oh1, oh2);
    std::printf("  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV\n", pe0, ke0, pe_f,
                ke_f);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: energy drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }
    if (max_warm_iters != 0) {
        std::printf("FAIL: warm max iters=%d (want 0)\n", max_warm_iters);
        return 1;
    }

    std::printf("PASS: xl-bomd-linearized-forces (zero-SCF shadow XL NVE)\n");
    return 0;
}
