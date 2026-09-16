/// @file regressions/dynamics/xl-bomd-linearized-uks/reg_xl-bomd-linearized-uks.cpp
/// @brief END-TO-END: zero-SCF linearized-functional XL-BOMD NVE on the O₂
///        triplet (Γ-UKS GPW shadow forces) — energy-drift + iters==0 +
///        ⟨S²⟩ pin for dual-spin P^σ propagation.
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
#include "atomiverse/molrs.hpp"

using namespace ATV;

int main() {
    // O₂ triplet, centered in 12 Å cubic MT cell — the identical twin of
    // xl-bomd-uks with ONLY the XlBomdLinearized tag added (SCF-XL vs
    // zero-SCF linearized-XL A/B on the same open-shell system).
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

    // xl-bomd-uks envelope settings (200 Ry / tol 1e-5 / mix 0.4 / PADE);
    // cold seed converges in ~23 dual-spin DIIS iterations.
    cpu::gpw::GPWEvaluator gpw_base(
        /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
        /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    gpw_base.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    gpw_base.n_beta = 5;
    cpu::electronic::XlBomdForce eval{std::move(gpw_base), 5,
                                      cpu::electronic::XlBomdLinearized{}};
    constexpr double dt = 0.5; // fs
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // Shadow PE + nuclear KE: looser than full-SCF XL-UKS (0.05) — one-shot
    // residual heat, same class as xl-bomd-linearized-forces; still catches
    // sign/shape/spin-swap bugs immediately.
    constexpr double drift_tol = 0.10;
    // Soft O–O bond window [Å] after production (no dissociation/collapse).
    constexpr double oo_max = 1.6;
    constexpr double oo_min = 0.9;
    // Löwdin ⟨S²⟩ triplet window (exact 2; small contamination allowed).
    constexpr double s2_lo = 1.5;
    constexpr double s2_hi = 2.5;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold full SCF seeds both spin XL histories.
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
    // Open-shell diagnostic: ⟨S²⟩ near 2 for a triplet.
    const double s2_0 = eval.method().last_s2();
    if (!std::isfinite(s2_0) || s2_0 < s2_lo || s2_0 > s2_hi) {
        std::printf("FAIL: ⟨S²⟩ out of triplet window after warm-up: %g\n",
                    s2_0);
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
        // Warm linearized steps: zero iterative SCF on the dual-spin path.
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
        // ⟨S²⟩ at the shadow densities stays in the triplet window.
        const double s2_c = eval.method().last_s2();
        if (!std::isfinite(s2_c) || s2_c < s2_lo || s2_c > s2_hi) {
            std::printf("FAIL: ⟨S²⟩ left triplet window at chunk %d: %g\n", c,
                        s2_c);
            return 1;
        }
        max_warm_iters = std::max(max_warm_iters, eval.last_result().iters);
        const double drift = std::fabs((pe + ke) - e0) / e_scale;
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

    const double dxf = static_cast<double>(frame.span(kAtom_x)[1]
                                           - frame.span(kAtom_x)[0]);
    const double dyf = static_cast<double>(frame.span(kAtom_y)[1]
                                           - frame.span(kAtom_y)[0]);
    const double dzf = static_cast<double>(frame.span(kAtom_z)[1]
                                           - frame.span(kAtom_z)[0]);
    const double r_oo = std::sqrt(dxf * dxf + dyf * dyf + dzf * dzf);
    if (!(r_oo < oo_max) || !(r_oo > oo_min)) {
        std::printf("FAIL: O–O bond exploded/collapsed r=%.3f Å "
                    "(window %g..%g)\n",
                    r_oo, oo_min, oo_max);
        return 1;
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    const double s2_f = eval.method().last_s2();
    std::printf("XL-BOMD linearized O2 triplet NVE (K=5, UKS): steps=%d  "
                "E0=%.6g eV  Ef=%.6g eV  worst |dE|/scale=%.3e  tol=%.2f  "
                "warm_iters=%d  rOO=%.3f Å  S2=%.4f→%.4f\n",
                n_prod, e0, pe_f + ke_f, worst_drift, drift_tol,
                max_warm_iters, r_oo, s2_0, s2_f);
    std::printf("  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV\n", pe0, ke0,
                pe_f, ke_f);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: energy drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }
    if (max_warm_iters != 0) {
        std::printf("FAIL: warm max iters=%d (want 0)\n", max_warm_iters);
        return 1;
    }

    std::printf("PASS: xl-bomd-linearized-uks (zero-SCF dual-spin shadow XL "
                "NVE)\n");
    return 0;
}
