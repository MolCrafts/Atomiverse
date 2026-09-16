/// @file regressions/gpw/gpw-bomd-uks-kpoints/reg_gpw-bomd-uks-kpoints.cpp
/// @brief END-TO-END: plain multi-k collinear UKS Born–Oppenheimer MD (full
///        SCF each step, previous-step dual \(P_k^\alpha/P_k^\beta\) warm
///        start) NVE on periodic H₂ with GPW forces — step-0 energy/force
///        and per-step E_pot trace pinned vs CP2K 2026.1 Quickstep UKS AIMD
///        with &KPOINTS, plus the NVE drift envelope and dual-spin payload.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Periodic H₂ — same cell/bond family as gpw-bomd-kpoints /
    // gpw-kpoint-uks-forces (DZV-GTH-PADE, PADE XC, mesh 1×1×2 unreduced) at
    // CUTOFF 300 Ry, but on the collinear UKS path with closed-shell
    // occupations (n_α=n_β=1) so both spin channels are non-empty and the
    // molecule stays bound. High-spin H₂ is unbound and cannot pin NVE.
    cpu::Frame frame(2);
    constexpr double bohr2ang = 0.52917720859;
    constexpr double L_bohr = 12.0;
    const double x0[2] = {5.5 * bohr2ang, 6.5 * bohr2ang};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = 6.0 * bohr2ang;
        frame.span(kAtom_z)[i] = 6.0 * bohr2ang;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // Antisymmetric stretch [Å/fs] — CP2K twin: ∓4.5710290e-5 Bohr/au_t.
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_bohr * bohr2ang);

    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 60, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.n_alpha = 1;
    gpw.n_beta = 1;
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false; // two sampled k (CP2K FULL_GRID .TRUE.)

    // ── CP2K 2026.1 pins (offline oracle; tools/cp2k_oracle/inputs/
    //    gpw-bomd-uks-kpoints/h2_uks_bomd_kpoints_nve.inp at CUTOFF 300) ─
    // Closed-shell UKS twin prints the same .ener Pot rows as the RKS
    // multi-k BOMD twin (identical valence filling on this H₂); the dual
    // gate is still the UKS product path (dual P_k^σ warm start).
    constexpr double e0_cp2k_ha = -1.074050184366365; // step-0 Pot
    constexpr double f0x_cp2k = -0.345535381;         // atom 0 F_x [Ha/Bohr]
    constexpr double pot_cp2k_ha[4] = {-1.097682104, -1.125544577,
                                       -1.130531455, -1.121507812};
    // Same-method twin bands (REFERENCE.md): Γ-point periodic-GPW residual
    // on this compressed H₂ (~4e-4 Ha / ~6e-3 Ha/Bohr); multi-k adds <1e-5.
    constexpr double tol_e0 = 1e-3;    // [Ha] (RKS twin measured 4.07e-4)
    constexpr double tol_f0 = 8e-3;    // [Ha/Bohr]
    constexpr double tol_trace = 2e-3; // [Ha] per-step
    constexpr double drift_tol = 0.05; // relative NVE envelope (family)
    constexpr double dt = 0.5;         // fs
    constexpr int n_prod = 4;
    constexpr double rmin = 0.4; // soft H–H window [Å] after production
    constexpr double rmax = 2.5;

    // ── Step 0: single point at t=0 — energy + forces seed the VV bootstrap
    //    (CP2K MD evaluates forces at step 0 before the first VV step). ────
    cpu::single_point(frame, gpw);
    const auto& r0 = gpw.last_result();
    if (!r0.converged) {
        std::printf("FAIL: step-0 multi-k UKS SCF not converged (iters=%d)\n",
                    r0.iters);
        return 1;
    }
    if (r0.n_kpts != 2 || static_cast<int>(r0.P_k.size()) != r0.n_kpts
        || static_cast<int>(r0.P_k_beta.size()) != r0.n_kpts) {
        std::printf("FAIL: expected n_kpts==2 with full dual P_k / P_k_beta, "
                    "got n_kpts=%d P_k=%zu P_k_beta=%zu\n",
                    r0.n_kpts, r0.P_k.size(), r0.P_k_beta.size());
        return 1;
    }
    const int iters_cold = r0.iters;
    const double e0_atv_ha = r0.e_total;
    if (!(std::fabs(e0_atv_ha - e0_cp2k_ha) <= tol_e0)) {
        std::printf("FAIL: step-0 energy vs CP2K |%.9f - (%.9f)| = %.3e Ha "
                    "> %.1e\n",
                    e0_atv_ha, e0_cp2k_ha,
                    std::fabs(e0_atv_ha - e0_cp2k_ha), tol_e0);
        return 1;
    }
    const double f0x_atv = static_cast<double>(frame.span(kAtom_fx)[0])
                           / ATV::units::kHartreePerBohrInEvPerAngstrom;
    if (!(std::fabs(f0x_atv - f0x_cp2k) <= tol_f0)) {
        std::printf("FAIL: step-0 F_x(atom0) vs CP2K |%.8f - (%.8f)| = %.3e "
                    "Ha/Bohr > %.1e\n",
                    f0x_atv, f0x_cp2k, std::fabs(f0x_atv - f0x_cp2k), tol_f0);
        return 1;
    }
    // Multi-k UKS stores integer-seed ⟨S²⟩ = s_z(s_z+1)+N_β with
    // s_z=(N_α−N_β)/2 — for N_α=N_β=1 that is 1. Require finite ≥ 0.
    const double s2_0 = gpw.last_s2();
    if (!std::isfinite(s2_0) || s2_0 < 0.0) {
        std::printf("FAIL: ⟨S²⟩ non-finite or negative after step-0: %g\n",
                    s2_0);
        return 1;
    }

    // E(t=0) for the drift envelope: PE from the single point + KE from the
    // initial velocities (1 eV = 103.6427 amu Å²/fs² — units::force_to_accel).
    const double pe0_ev = e0_atv_ha * ATV::units::kHartreeInEv;
    double ke0_ev = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double vx = static_cast<double>(frame.span(kAtom_vx)[i]);
        ke0_ev += 0.5 * 1.008 * vx * vx / ATV::units::force_to_accel;
    }
    const double e0_tot = pe0_ev + ke0_ev;
    const double e_scale = std::max(1.0, std::fabs(e0_tot));

    // ── NVE production: 4 × 1-step chunks; per-step trace vs CP2K. ───────
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(gpw);
    sim.enable_timing(false);
    sim.build();

    double worst_drift = 0.0;
    double worst_trace = 0.0;
    int worst_iters = 0;
    for (int s = 0; s < n_prod; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: production step %d did not run\n", s + 1);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite energy at step %d PE=%g KE=%g\n",
                        s + 1, pe, ke);
            return 1;
        }
        const auto& rs = gpw.last_result();
        if (!rs.converged) {
            std::printf("FAIL: SCF not converged at step %d (iters=%d)\n",
                        s + 1, rs.iters);
            return 1;
        }
        if (static_cast<int>(rs.P_k.size()) != rs.n_kpts
            || static_cast<int>(rs.P_k_beta.size()) != rs.n_kpts) {
            std::printf("FAIL: dual P_k payload missing at step %d "
                        "(P_k=%zu P_k_beta=%zu n_kpts=%d)\n",
                        s + 1, rs.P_k.size(), rs.P_k_beta.size(), rs.n_kpts);
            return 1;
        }
        // Warm start: production SCF (previous-step dual P_k^σ as the
        // initial density) must stay near the cold step-0 budget. Geometry
        // motion can cost a few extra DIIS steps vs the t=0 cold solve;
        // allow +3. The strict same-geometry iteration *win* is unit-pinned
        // in test_cpu_gpw_bomd
        // (GpwBomd.PlainMultiKUksRelaunchWarmStartsBothSpins).
        constexpr int warm_margin = 3;
        if (!(rs.iters <= iters_cold + warm_margin)) {
            std::printf("FAIL: step %d SCF iters=%d above cold+margin=%d "
                        "(dual P_k^σ warm start regressed the SCF)\n",
                        s + 1, rs.iters, iters_cold + warm_margin);
            return 1;
        }
        if (rs.iters > worst_iters)
            worst_iters = rs.iters;
        const double pot_atv_ha = pe / ATV::units::kHartreeInEv;
        const double dtrace = std::fabs(pot_atv_ha - pot_cp2k_ha[s]);
        if (dtrace > worst_trace)
            worst_trace = dtrace;
        if (!(dtrace <= tol_trace)) {
            std::printf("FAIL: step %d E_pot vs CP2K |%.9f - (%.9f)| = %.3e "
                        "Ha > %.1e\n",
                        s + 1, pot_atv_ha, pot_cp2k_ha[s], dtrace, tol_trace);
            return 1;
        }
        const double drift = std::fabs((pe + ke) - e0_tot) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;
        for (int i = 0; i < 2; ++i) {
            if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[i]))
                || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[i]))
                || !std::isfinite(
                    static_cast<double>(frame.span(kAtom_z)[i]))) {
                std::printf("FAIL: non-finite coordinate at step %d atom %d\n",
                            s + 1, i);
                return 1;
            }
        }
    }

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: NVE drift %.3e exceeds envelope %.2f\n",
                    worst_drift, drift_tol);
        return 1;
    }

    const double dx = static_cast<double>(frame.span(kAtom_x)[1]
                                          - frame.span(kAtom_x)[0]);
    const double dy = static_cast<double>(frame.span(kAtom_y)[1]
                                          - frame.span(kAtom_y)[0]);
    const double dz = static_cast<double>(frame.span(kAtom_z)[1]
                                          - frame.span(kAtom_z)[0]);
    const double rhh = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(rhh > rmin) || !(rhh < rmax)) {
        std::printf("FAIL: H–H bond out of window r=%.3f Å (%.2f–%.2f)\n",
                    rhh, rmin, rmax);
        return 1;
    }

    const double s2_f = gpw.last_s2();
    if (!std::isfinite(s2_f) || s2_f < 0.0) {
        std::printf("FAIL: ⟨S²⟩ non-finite or negative after production: %g\n",
                    s2_f);
        return 1;
    }

    std::printf("BOMD multi-k UKS H2 NVE (nk=%d, Na=1 Nb=1): E0=%.9f Ha "
                "(CP2K %.9f, d=%.2e)  F0x=%.8f Ha/Bohr (CP2K %.8f, d=%.2e)\n",
                r0.n_kpts, e0_atv_ha, e0_cp2k_ha,
                std::fabs(e0_atv_ha - e0_cp2k_ha), f0x_atv, f0x_cp2k,
                std::fabs(f0x_atv - f0x_cp2k));
    std::printf("  steps=%d  worst |dPot| vs CP2K=%.3e Ha (tol %.0e)  "
                "worst NVE drift=%.3e (tol %.2f)  cold iters=%d  worst warm "
                "iters=%d  rHH=%.3f Å  S2=%.4f→%.4f\n",
                n_prod, worst_trace, tol_trace, worst_drift, drift_tol,
                iters_cold, worst_iters, rhh, s2_0, s2_f);
    std::printf("PASS: gpw-bomd-uks-kpoints (plain multi-k UKS BOMD NVE vs "
                "CP2K Quickstep UKS AIMD)\n");
    return 0;
}
