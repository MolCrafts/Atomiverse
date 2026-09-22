/// @file regressions/dynamics/cpmd-uks/reg_cpmd-uks.cpp
/// @brief END-TO-END: Car–Parrinello MD (fictitious electronic DOF) NVE on
///        O₂ triplet with Γ-UKS GPW — CP conserved-energy drift envelope.
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
#include "atomiverse/molrs.hpp"

using namespace ATV;

int main() {
    // O₂ triplet, centered in 12 Å cubic MT cell (same family as
    // xl-bomd-uks / gpw-uks-forces). Dual α/β CPMD propagators engage via
    // n_alpha/n_beta + cpmd.
    cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    const double mass = 15.999;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = mass;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = 0.5 * L;
        frame.span(kAtom_y)[i] = 0.5 * L;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    // Antisymmetric stretch [Å/fs] — no COM translation, small KE.
    frame.span(kAtom_vz)[0] = -0.0008;
    frame.span(kAtom_vz)[1] = 0.0008;
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    // Envelope run: 200 Ry / tol 1e-5 keeps multi-step UKS inside ≤60 s.
    // Absolute E/F parity is gpw-uks-forces / gpw-uks-o2-gamma's job.
    constexpr double dt = 0.1; // fs — short step for electronic stability
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
        /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    gpw.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    gpw.n_beta = 5;
    cpu::electronic::CarParrinello eval{std::move(gpw), 2000.0};
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // CP conserved-energy envelope (same class as closed-shell CPMD).
    constexpr double drift_tol = 0.10;
    constexpr double oo_max = 1.6;
    constexpr double oo_min = 0.9;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold UKS SCF seed + BO force evaluation (Te still 0).
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
        if (!eval.last_result().converged) {
            std::printf("FAIL: SCF not converged after warm-up (iters=%d)\n",
                        eval.last_result().iters);
            return 1;
        }
        const double s2_0 = eval.method().last_s2();
        if (!std::isfinite(s2_0) || s2_0 < 1.5 || s2_0 > 2.5) {
            std::printf("FAIL: ⟨S²⟩ out of triplet window after warm-up: %g\n",
                        s2_0);
            return 1;
        }
    }

    // First CP production step engages fixed-density dual-KS + electronic
    // advance (Te becomes nonzero). Baseline E_cons after this step so the
    // Te turn-on is not counted as secular drift.
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

        for (int i = 0; i < 2; ++i) {
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

    const double dz =
        static_cast<double>(frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0]);
    const double dx =
        static_cast<double>(frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0]);
    const double dy =
        static_cast<double>(frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0]);
    const double r_oo = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(r_oo < oo_max) || !(r_oo > oo_min)) {
        std::printf("FAIL: O–O bond exploded/collapsed r=%.3f Å "
                    "(window %g..%g)\n",
                    r_oo, oo_min, oo_max);
        return 1;
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    const double te_f = eval.last_electronic_ke_ev();
    // ⟨S²⟩ is only refreshed on full SCF; fixed-density steps leave the
    // seed value. Re-check the warm-up diagnostic was already gate-passed;
    // here we only require the final energy surface stayed bound.
    std::printf(
        "CPMD O2 triplet NVE (mu=2000 a.u., UKS): steps=%d  E0=%.6g eV  "
        "Ef=%.6g eV  worst |dE_cons|/scale=%.3e  tol=%.2f  rOO=%.3f Å\n",
        n_prod, e0, pe_f + ke_f + te_f, worst_drift, drift_tol, r_oo);
    std::printf(
        "  PE0=%.6g  KE0=%.6g  Te0=%.6g  PEf=%.6g  KEf=%.6g  Tef=%.6g eV  "
        "prod SCF iters=%d (expect 0)\n",
        pe0, ke0, te0, pe_f, ke_f, te_f, last_iters);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: CP energy drift %.3e exceeds tol %.2f\n",
                    worst_drift, drift_tol);
        return 1;
    }

    std::printf("PASS: cpmd-uks (CPMD UKS NVE conserved-energy envelope, "
                "dual fictitious e- DOF)\n");
    return 0;
}
