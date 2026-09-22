/// @file regressions/dynamics/cpmd-rattle-water/reg_cpmd-rattle-water.cpp
/// @brief END-TO-END: Car–Parrinello MD with electronic RATTLE on H₂O NVE
///        (GPW) — CP conserved-energy drift envelope pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Isolated water, CP2K getting-started geometry (Å) — same twin family as
    // cpmd-water, now with electronic RATTLE (constraint-preserving orbitals).
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
    // COM-preserving H stretch mode [Å/fs] — small KE, no COM translation.
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

    constexpr double dt = 0.1; // fs — short step for electronic stability
    cpu::gpw::GPWEvaluator gpw_base(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    // Same μ [a.u.] as the Löwdin foothold pin for fair envelope comparison —
    // a constructor argument on the CP wrapper, not a public flag.
    cpu::electronic::CarParrinello eval{std::move(gpw_base), /*mu=*/2000.0};
    // dt is pushed in from `VelocityVerlet` by MD at build (TimestepProvider).

    // Drift production after the baseline CP step that engages Te.
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // Tighter than Löwdin foothold (0.10) — RATTLE conserves better.
    constexpr double drift_tol = 0.05;
    constexpr double oh_max = 1.8; // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold SCF seed + BO force evaluation (Te still 0).
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: warm-up step did not run\n");
        return 1;
    }
    {
        const double pe_w = frame.meta.get(kEnergy);
        const double ke_w = frame.meta.get(kKinetic);
        if (!std::isfinite(pe_w) || !std::isfinite(ke_w) || !(ke_w >= 0.0)) {
            std::printf("FAIL: non-finite energy after warm-up PE=%g KE=%g\n",
                        pe_w, ke_w);
            return 1;
        }
        if (!(ke_w > 1e-12)) {
            std::printf(
                "FAIL: expected non-trivial KE after warm-up, got %g eV\n",
                ke_w);
            return 1;
        }
        if (!(pe_w < -100.0)) {
            std::printf("FAIL: PE not bound after warm-up PE=%g eV\n", pe_w);
            return 1;
        }
    }

    // First CP production step engages fixed-density eval + electronic advance.
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: first CP production step did not run\n");
        return 1;
    }
    if (eval.last_result().iters != 0) {
        std::printf(
            "FAIL: first CP step re-ran SCF (iters=%d) — not fixed-density\n",
            eval.last_result().iters);
        return 1;
    }
    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    const double te0 = eval.last_electronic_ke_ev();
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !std::isfinite(te0)
        || !(ke0 >= 0.0) || !(te0 >= 0.0)) {
        std::printf("FAIL: non-finite energy after first CP step PE=%g KE=%g "
                    "Te=%g\n",
                    pe0, ke0, te0);
        return 1;
    }
    const double e0 = pe0 + ke0 + te0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    int last_iters = eval.last_result().iters;
    const int steps_per = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (sim.run(frame, steps_per) != steps_per) {
            std::printf("FAIL: production chunk %d incomplete\n", c);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        const double te = eval.last_electronic_ke_ev();
        if (!std::isfinite(pe) || !std::isfinite(ke) || !std::isfinite(te)
            || !(ke >= 0.0) || !(te >= 0.0)) {
            std::printf(
                "FAIL: non-finite energy at chunk %d PE=%g KE=%g Te=%g\n", c,
                pe, ke, te);
            return 1;
        }
        last_iters = eval.last_result().iters;
        if (last_iters != 0) {
            std::printf(
                "FAIL: production step re-ran SCF (iters=%d) — CPMD must not "
                "be a BOMD rename\n",
                last_iters);
            return 1;
        }
        const double e_cons = pe + ke + te;
        const double drift = std::fabs(e_cons - e0) / e_scale;
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
    const double te_f = eval.last_electronic_ke_ev();
    std::printf(
        "CPMD+RATTLE H2O NVE (mu=2000 a.u.): steps=%d  E0=%.6g eV  Ef=%.6g eV  "
        "worst |dE_cons|/scale=%.3e  tol=%.2f  oh=%.3f/%.3f Å\n",
        n_prod, e0, pe_f + ke_f + te_f, worst_drift, drift_tol, oh1, oh2);
    std::printf(
        "  PE0=%.6g  KE0=%.6g  Te0=%.6g  PEf=%.6g  KEf=%.6g  Tef=%.6g eV  "
        "prod SCF iters=%d (expect 0)\n",
        pe0, ke0, te0, pe_f, ke_f, te_f, last_iters);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: CP energy drift %.3e exceeds tol %.2f\n",
                    worst_drift, drift_tol);
        return 1;
    }

    std::printf("PASS: cpmd-rattle-water (CPMD+RATTLE NVE conserved-energy "
                "envelope)\n");
    return 0;
}
