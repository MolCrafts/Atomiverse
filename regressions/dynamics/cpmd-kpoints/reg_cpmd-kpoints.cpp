/// @file regressions/dynamics/cpmd-kpoints/reg_cpmd-kpoints.cpp
/// @brief END-TO-END: multi-k Car–Parrinello MD (Bloch C_k) NVE on periodic
///        H₂ with GPW forces — CP conserved-energy drift envelope pin.
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
    // Periodic H₂ — twin family of gpw-kpoint-forces / xl-bomd-kpoints
    // (DZV-GTH-PADE, PADE XC, mesh 1×1×2 unreduced). CPMD multi-k Bloch C_k.
    cpu::Frame frame(2);
    constexpr double bohr2ang = 0.52917720859;
    constexpr double L_bohr = 12.0;
    constexpr double L_ang = L_bohr * bohr2ang;
    constexpr double r0_bohr = 1.0;
    const double x0[2] = {(L_bohr / 2.0 - 0.5 * r0_bohr) * bohr2ang,
                          (L_bohr / 2.0 + 0.5 * r0_bohr) * bohr2ang};
    const double y0[2] = {L_ang / 2.0, L_ang / 2.0};
    const double z0[2] = {L_ang / 2.0, L_ang / 2.0};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = y0[i];
        frame.span(kAtom_z)[i] = z0[i];
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_ang);

    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

    constexpr double dt = 0.1; // fs — short step for electronic stability
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false;
    // Fictitious electron mass [a.u.]; a constructor argument on the CP
    // wrapper (no public `cpmd_mu` flag), so the case owns the value it prints.
    constexpr double mu_au = 400.0;
    cpu::electronic::CarParrinello eval{std::move(gpw), mu_au};

    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // Soft CP conserved-energy envelope (multi-k RATTLE foothold).
    constexpr double drift_tol = 0.10;
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold multi-k SCF seed + BO forces (Te still 0).
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
        if (!(pe_w < -5.0)) {
            std::printf("FAIL: PE not bound after warm-up PE=%g eV\n", pe_w);
            return 1;
        }
        if (eval.last_result().n_kpts < 2) {
            std::printf("FAIL: expected n_kpts >= 2 after seed, got %d\n",
                        eval.last_result().n_kpts);
            return 1;
        }
    }

    // First CP production step engages fixed-density multi-k + electronic
    // advance. Baseline E_cons after this step so Te turn-on is not drift.
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
    if (eval.last_result().n_kpts < 2) {
        std::printf("FAIL: n_kpts < 2 on CP step (%d)\n",
                    eval.last_result().n_kpts);
        return 1;
    }

    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    const double te0 = eval.last_electronic_ke_ev();
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !std::isfinite(te0)
        || !(ke0 >= 0.0) || !(te0 >= 0.0)) {
        std::printf("FAIL: non-finite baseline PE=%g KE=%g Te=%g\n", pe0, ke0,
                    te0);
        return 1;
    }
    const double e0 = pe0 + ke0 + te0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    const int steps_per = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (sim.run(frame, steps_per) != steps_per) {
            std::printf("FAIL: production chunk %d incomplete\n", c);
            return 1;
        }
        if (eval.last_result().iters != 0) {
            std::printf("FAIL: production chunk %d re-ran SCF (iters=%d)\n", c,
                        eval.last_result().iters);
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
        const double drift = std::fabs((pe + ke + te) - e0) / e_scale;
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

    const double dx =
        static_cast<double>(frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0]);
    const double dy =
        static_cast<double>(frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0]);
    const double dz =
        static_cast<double>(frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0]);
    const double rhh = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(rhh > rmin) || !(rhh < rmax)) {
        std::printf("FAIL: H–H bond out of window r=%.3f Å (%.2f–%.2f)\n", rhh,
                    rmin, rmax);
        return 1;
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    const double te_f = eval.last_electronic_ke_ev();
    std::printf(
        "CPMD multi-k H2 NVE (mu=%.0f, nk=%d): steps=%d  E0=%.6g eV  "
        "Ef=%.6g eV  worst |dE_cons|/scale=%.3e  tol=%.2f  rHH=%.3f Å\n",
        mu_au, eval.last_result().n_kpts, n_prod, e0, pe_f + ke_f + te_f,
        worst_drift, drift_tol, rhh);
    std::printf("  PE0=%.6g  KE0=%.6g  Te0=%.6g  PEf=%.6g  KEf=%.6g  "
                "Tef=%.6g eV\n",
                pe0, ke0, te0, pe_f, ke_f, te_f);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: E_cons drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }

    std::printf(
        "PASS: cpmd-kpoints (multi-k Bloch CPMD NVE E_cons envelope)\n");
    return 0;
}
