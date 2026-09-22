/// @file regressions/dynamics/second-gen-cpmd-uks/reg_second-gen-cpmd-uks.cpp
/// @brief END-TO-END: second-generation Car–Parrinello MD (ASPC + partial
///        SCF corrector) NVE on O₂ triplet with Γ-UKS GPW — energy-drift
///        envelope + corrector-budget identity pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/second_gen_cp.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

int main() {
    // O₂ triplet, centered in 12 Å cubic MT cell (same family as
    // gpw-uks-o2-gamma / xl-bomd-uks). Dual ASPC α/β + partial SCF.
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

    constexpr double dt = 0.5; // fs — BOMD-class step
    constexpr int corrector_budget = 5;
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
        /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    gpw.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    gpw.n_beta = 5;
    cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                      corrector_budget};

    constexpr int n_seed = 2; // warm ASPC history (order K=1 needs K+1)
    constexpr int n_prod = 2;
    constexpr int n_chunks = 1;
    // Soft relative drift (partial SCF heat) — same class as RKS SGCP water.
    constexpr double drift_tol = 0.10;
    constexpr double oo_max = 1.6;
    constexpr double oo_min = 0.9;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Seed: full-SCF steps fill dual ASPC history.
    if (sim.run(frame, n_seed) != n_seed) {
        std::printf("FAIL: seed steps incomplete\n");
        return 1;
    }
    {
        const double pe_w = frame.meta.get(kEnergy);
        const double ke_w = frame.meta.get(kKinetic);
        if (!std::isfinite(pe_w) || !std::isfinite(ke_w) || !(ke_w >= 0.0)) {
            std::printf("FAIL: non-finite energy after seed PE=%g KE=%g\n",
                        pe_w, ke_w);
            return 1;
        }
        if (!(ke_w > 1e-12)) {
            std::printf("FAIL: expected non-trivial KE after seed, got %g eV\n",
                        ke_w);
            return 1;
        }
        if (!(pe_w < -100.0)) {
            std::printf("FAIL: PE not bound after seed PE=%g eV\n", pe_w);
            return 1;
        }
        const double s2_w = eval.method().last_s2();
        if (!std::isfinite(s2_w) || s2_w < 1.5 || s2_w > 2.5) {
            std::printf("FAIL: ⟨S²⟩ out of triplet window after seed: %g\n",
                        s2_w);
            return 1;
        }
    }

    // Baseline E after history is warm (first production step uses dual ASPC).
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: first SGCP production step did not run\n");
        return 1;
    }
    const int iters_first = eval.last_result().iters;
    if (!(iters_first <= corrector_budget)) {
        std::printf(
            "FAIL: first SGCP step iters=%d exceeds corrector budget %d "
            "(not partial SCF)\n",
            iters_first, corrector_budget);
        return 1;
    }
    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !(ke0 >= 0.0)) {
        std::printf("FAIL: non-finite energy after first SGCP PE=%g KE=%g\n",
                    pe0, ke0);
        return 1;
    }
    const double e0 = pe0 + ke0;
    const double e_scale = std::max(1.0, std::fabs(e0));
    const double s2_0 = eval.method().last_s2();

    double worst_drift = 0.0;
    int max_prod_iters = iters_first;
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
        const int last_iters = eval.last_result().iters;
        if (last_iters > max_prod_iters)
            max_prod_iters = last_iters;
        if (last_iters > corrector_budget) {
            std::printf(
                "FAIL: production iters=%d > corrector budget %d — SGCP must "
                "not run full BOMD SCF\n",
                last_iters, corrector_budget);
            return 1;
        }
        const double e_cons = pe + ke;
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
    const double s2_f = eval.method().last_s2();
    std::printf(
        "SGCP O2 triplet NVE (ASPC K=1, corrector=%d, UKS): seed=%d prod=%d  "
        "E0=%.6g eV  Ef=%.6g eV  worst |dE|/scale=%.3e  tol=%.2f  rOO=%.3f Å  "
        "S2=%.4f→%.4f\n",
        corrector_budget, n_seed, n_prod, e0, pe_f + ke_f, worst_drift,
        drift_tol, r_oo, s2_0, s2_f);
    std::printf(
        "  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV  max prod SCF iters=%d "
        "(budget %d)\n",
        pe0, ke0, pe_f, ke_f, max_prod_iters, corrector_budget);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: SGCP energy drift %.3e exceeds tol %.2f\n",
                    worst_drift, drift_tol);
        return 1;
    }
    if (!std::isfinite(s2_f) || s2_f < 1.5 || s2_f > 2.5) {
        std::printf("FAIL: ⟨S²⟩ left triplet window: %g\n", s2_f);
        return 1;
    }

    std::printf("PASS: second-gen-cpmd-uks (SGCP UKS NVE ASPC+partial SCF)\n");
    return 0;
}
