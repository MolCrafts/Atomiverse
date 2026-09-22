/// @file regressions/dynamics/xl-bomd-kpoints/reg_xl-bomd-kpoints.cpp
/// @brief END-TO-END: multi-k XL-BOMD (Niklasson dissipative extended-
///        Lagrangian BOMD) NVE on periodic H₂ with GPW forces — energy-drift
///        envelope pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Periodic H₂ — twin family of gpw-kpoint-forces (DZV-GTH-PADE, PADE XC,
    // mesh 1×1×2, unreduced). XL-BOMD K=5 on complex P_k.
    cpu::Frame frame(2);
    // Bond along x, centered in a cubic 12 Bohr ≈ 6.35 Å cell. Use Å for Frame.
    constexpr double bohr2ang = 0.52917720859;
    constexpr double L_bohr = 12.0;
    constexpr double L_ang = L_bohr * bohr2ang;
    constexpr double r0_bohr = 1.0; // compressed bond (same as kpoint-forces)
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
    // Antisymmetric stretch [Å/fs] — small KE, no COM translation.
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_ang);

    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

    // Cheap multi-k envelope (forces quality is pinned in gpw-kpoint-forces).
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false; // two sampled k
    cpu::electronic::XlBomdForce eval{std::move(gpw), 5};

    constexpr double dt = 0.5; // fs
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    constexpr double drift_tol = 0.05;
    // Soft H–H window [Å] after production (no dissociation / collapse).
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: warm-up step did not run\n");
        return 1;
    }
    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !(ke0 >= 0.0)) {
        std::printf("FAIL: non-finite or negative energy after warm-up "
                    "PE=%g KE=%g\n",
                    pe0, ke0);
        return 1;
    }
    if (!(ke0 > 1e-12)) {
        std::printf("FAIL: expected non-trivial KE after warm-up, got %g eV\n",
                    ke0);
        return 1;
    }
    // Bound PE: H₂ PADE is O(−10¹) eV, not near zero / unbound.
    if (!(pe0 < -5.0)) {
        std::printf("FAIL: PE not bound after warm-up PE=%g eV\n", pe0);
        return 1;
    }
    if (eval.last_result().n_kpts < 2) {
        std::printf("FAIL: expected n_kpts >= 2, got %d\n",
                    eval.last_result().n_kpts);
        return 1;
    }
    if (static_cast<int>(eval.last_result().P_k.size())
        != eval.last_result().n_kpts) {
        std::printf("FAIL: P_k payload missing after multi-k SCF\n");
        return 1;
    }
    const double e0 = pe0 + ke0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
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
        if (!eval.last_result().converged) {
            std::printf("FAIL: SCF not converged at chunk %d (iters=%d)\n", c,
                        eval.last_result().iters);
            return 1;
        }
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
    std::printf("XL-BOMD multi-k H2 NVE (K=5, nk=%d): steps=%d  E0=%.6g eV  "
                "Ef=%.6g eV  worst |dE|/scale=%.3e  tol=%.2f  rHH=%.3f Å\n",
                eval.last_result().n_kpts, n_prod, e0, pe_f + ke_f, worst_drift,
                drift_tol, rhh);
    std::printf("  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV  "
                "final SCF iters=%d\n",
                pe0, ke0, pe_f, ke_f, eval.last_result().iters);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: energy drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }

    std::printf("PASS: xl-bomd-kpoints (multi-k XL-BOMD NVE energy-drift "
                "envelope, K=5)\n");
    return 0;
}
