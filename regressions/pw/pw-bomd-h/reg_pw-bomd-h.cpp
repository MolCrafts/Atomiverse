/// @file regressions/pw/pw-bomd-h/reg_pw-bomd-h.cpp
/// @brief END-TO-END: short PW BOMD NVE on H (GTH PP) — energy-drift pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW (+ MD product).
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;

int main() {
    // Single H at the cubic-cell origin: lattice symmetry ⇒ F ≈ 0
    // (same foothold as regressions/pw/h_force_*). Free-particle
    // NVE with a hand-set velocity (InitVelocity always zeros COM, which
    // kills KE for N=1). BOMD product path still runs full SCF+analytic
    // forces every step.
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_tag)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    // Mild drift velocity [Å/fs] so KE is non-trivial while F≈0 keeps PE flat.
    frame.span(kAtom_vx)[0] = 0.002;
    frame.span(kAtom_vy)[0] = 0.0;
    frame.span(kAtom_vz)[0] = 0.0;
    frame.box = cpu::Box::cubic(6.0);

    cpu::pw::PW pw;
    pw.gth_name = "GTH-PBE-q1"; // analytic GTH for H (see REFERENCE.md)
    pw.ecut_ry = 5.0;
    pw.k_mesh = 1;
    pw.tol = 1e-4;
    pw.max_iter = 12;
    pw.mix_beta = 0.5;

    constexpr double dt = 0.5; // fs
    constexpr int n_prod = 10;
    constexpr int n_chunks = 5;
    // Relative |ΔE|/max(1,|E0|). Single-atom F≈0 → near-perfect NVE;
    // envelope still catches NaN / force-binding failure / secular leak.
    constexpr double drift_tol = 0.05;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.on(md::Phase::force)
        .add(cpu::pw::PWEvaluator{std::move(pw)}.with_gradient(true));
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
    // Hand-set velocity must produce a measurable KE (not COM-killed zero).
    if (!(ke0 > 1e-12)) {
        std::printf("FAIL: expected non-trivial KE after warm-up, got %g eV\n",
                    ke0);
        return 1;
    }
    const double e0 = pe0 + ke0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    // Residual force after warm-up (symmetry pin, eV/Å).
    const double fmag0 = std::sqrt(static_cast<double>(
        frame.span(kAtom_fx)[0] * frame.span(kAtom_fx)[0]
        + frame.span(kAtom_fy)[0] * frame.span(kAtom_fy)[0]
        + frame.span(kAtom_fz)[0] * frame.span(kAtom_fz)[0]));
    constexpr double f_tol = 0.5; // matches h_force_analytic residual bound
    if (!(fmag0 < f_tol)) {
        std::printf("FAIL: |F| after warm-up = %.4g eV/Å (tol %g, symmetry)\n",
                    fmag0, f_tol);
        return 1;
    }

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
        const double drift = std::fabs((pe + ke) - e0) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;

        if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[0]))
            || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[0]))
            || !std::isfinite(static_cast<double>(frame.span(kAtom_z)[0]))) {
            std::printf("FAIL: non-finite coordinate at chunk %d\n", c);
            return 1;
        }
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    std::printf("PW BOMD H NVE: steps=%d  E0=%.6g eV  Ef=%.6g eV  "
                "worst |dE|/scale=%.3e  tol=%.2f  |F0|=%.3g eV/A\n",
                n_prod, e0, pe_f + ke_f, worst_drift, drift_tol, fmag0);
    std::printf("  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV\n", pe0, ke0, pe_f,
                ke_f);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: energy drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }

    std::printf("PASS: pw-bomd-h (PW BOMD NVE energy-drift envelope)\n");
    return 0;
}
