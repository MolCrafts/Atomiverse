/// @file regressions/dynamics/cpmd-uks-thermostat-kpoints/reg_cpmd-uks-thermostat-kpoints.cpp
/// @brief END-TO-END: multi-k UKS Car–Parrinello MD with the Nosé–Hoover
///        electronic thermostat engaged per (k, σ) channel — extended-energy
///        drift + Te-control envelope + UKS-vs-RKS closed-shell consistency
///        on periodic H₂ (GPW).
///
/// Combo pin of `cpmd-uks-kpoints` × `cpmd-thermostat-kpoints`. See
/// REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD. Exit non-zero
/// on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr double kBohr2Ang = 0.52917720859;
constexpr double kLBohr = 12.0;
constexpr double kLAng = kLBohr * kBohr2Ang;
constexpr double kR0Bohr = 1.0;

/// Periodic H₂ (compressed bond along x, antisymmetric stretch velocity) —
/// twin geometry of cpmd-uks-kpoints / cpmd-thermostat-kpoints.
cpu::Frame h2_frame() {
    cpu::Frame frame(2);
    const double x0[2] = {(kLBohr / 2.0 - 0.5 * kR0Bohr) * kBohr2Ang,
                          (kLBohr / 2.0 + 0.5 * kR0Bohr) * kBohr2Ang};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = kLAng / 2.0;
        frame.span(kAtom_z)[i] = kLAng / 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(kLAng);
    return frame;
}

/// Multi-k GPW evaluator on the family settings (DZV-GTH-PADE, PADE,
/// 100 Ry, mesh 1×1×2 unreduced). Spin fillings and `tol` are the caller's.
cpu::gpw::GPWEvaluator h2_gpw(double tol) {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 80, /*tol*/ tol,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false;
    return gpw;
}

} // namespace

int main() {
    // ── UKS-vs-RKS closed-shell consistency (cold SCF identity) ─────────
    // Same gate class as cpmd-uks-kpoints: multi-k UKS (n_alpha = n_beta = 1)
    // and multi-k RKS (n_occ = 1) single points at tol = 1e-7 must agree.
    constexpr double consistency_tol_ev = 3e-4; // ≈ 1.1e-5 Ha
    {
        cpu::Frame frame_rks = h2_frame();
        cpu::gpw::GPWEvaluator rks = h2_gpw(/*tol=*/1e-7);
        cpu::single_point(frame_rks, rks);
        const double pe_rks = frame_rks.meta.get(kEnergy);

        cpu::Frame frame_uks = h2_frame();
        cpu::gpw::GPWEvaluator uks = h2_gpw(/*tol=*/1e-7);
        uks.n_alpha = 1;
        uks.n_beta = 1;
        cpu::single_point(frame_uks, uks);
        const double pe_uks = frame_uks.meta.get(kEnergy);

        if (!std::isfinite(pe_rks) || !(pe_rks < -5.0)) {
            std::printf("FAIL: multi-k RKS single point unbound PE=%g eV\n",
                        pe_rks);
            return 1;
        }
        if (!std::isfinite(pe_uks) || !(pe_uks < -5.0)) {
            std::printf("FAIL: multi-k UKS single point unbound PE=%g eV\n",
                        pe_uks);
            return 1;
        }
        const double d_cons = std::fabs(pe_uks - pe_rks);
        if (!(d_cons <= consistency_tol_ev)) {
            std::printf("FAIL: UKS-vs-RKS closed-shell inconsistency "
                        "|%.6f - %.6f| = %.3e eV > %.1e\n",
                        pe_uks, pe_rks, d_cons, consistency_tol_ev);
            return 1;
        }
        std::printf("closed-shell consistency: UKS-k %.6f eV vs RKS-k %.6f "
                    "eV (|d|=%.2e, tol=%.0e)\n",
                    pe_uks, pe_rks, d_cons, consistency_tol_ev);
    }

    // ── Multi-k UKS CPMD + electronic NH trajectory ─────────────────────
    cpu::Frame frame = h2_frame();
    cpu::gpw::GPWEvaluator gpw = h2_gpw(/*tol=*/1e-5);
    // Closed-shell fillings on the UKS path: both spin channels non-empty
    // (dual per-(k, σ) Bloch CP + NH), molecule bound.
    gpw.n_alpha = 1;
    gpw.n_beta = 1;

    constexpr double mu_au = 400.0;
    constexpr double k0_ha = 1e-5;
    constexpr double tau_e_fs = 50.0;
    cpu::electronic::CarParrinello eval{
        std::move(gpw), mu_au,
        cpu::electronic::ElectronicThermostat{k0_ha, tau_e_fs}};

    constexpr double dt = 0.1; // fs — short step for electronic stability
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // Family envelope (REFERENCE.md): extended-energy drift + Te control.
    constexpr double drift_tol = 0.10;
    constexpr double te_ceil_ev = 0.05;
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

    // Warm-up: cold multi-k UKS SCF seed + BO forces (Te and Eth still 0).
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
        if (!eval.last_result().converged) {
            std::printf("FAIL: UKS SCF not converged after warm-up "
                        "(iters=%d)\n",
                        eval.last_result().iters);
            return 1;
        }
        if (eval.last_result().n_kpts < 2) {
            std::printf("FAIL: expected n_kpts >= 2 after seed, got %d\n",
                        eval.last_result().n_kpts);
            return 1;
        }
    }

    // First CP production step engages dual-spin fixed-density multi-k +
    // per-(k, σ) electronic NH advance. Baseline E_ext after this step so
    // the Te turn-on is not counted as drift.
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
    const double eth0 = eval.last_electronic_thermostat_ev();
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !std::isfinite(te0)
        || !std::isfinite(eth0) || !(ke0 >= 0.0) || !(te0 >= 0.0)) {
        std::printf("FAIL: non-finite baseline PE=%g KE=%g Te=%g Eth=%g\n", pe0,
                    ke0, te0, eth0);
        return 1;
    }
    const double e0 = pe0 + ke0 + te0 + eth0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    double worst_te = te0;
    int last_iters = eval.last_result().iters;
    const int steps_per = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (sim.run(frame, steps_per) != steps_per) {
            std::printf("FAIL: production chunk %d incomplete\n", c);
            return 1;
        }
        last_iters = eval.last_result().iters;
        if (last_iters != 0) {
            std::printf("FAIL: production chunk %d re-ran SCF (iters=%d)\n", c,
                        last_iters);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        const double te = eval.last_electronic_ke_ev();
        const double eth = eval.last_electronic_thermostat_ev();
        if (!std::isfinite(pe) || !std::isfinite(ke) || !std::isfinite(te)
            || !std::isfinite(eth) || !(ke >= 0.0) || !(te >= 0.0)) {
            std::printf("FAIL: non-finite energy at chunk %d PE=%g KE=%g Te=%g "
                        "Eth=%g\n",
                        c, pe, ke, te, eth);
            return 1;
        }
        const double drift = std::fabs((pe + ke + te + eth) - e0) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;
        if (te > worst_te)
            worst_te = te;

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
    const double eth_f = eval.last_electronic_thermostat_ev();
    std::printf(
        "CPMD+eNH multi-k UKS H2 NVE (mu=%.0f, K0=%.0e Ha, tau=%.0f fs, "
        "nk=%d, na=nb=1): steps=%d  E0=%.6g eV  Ef=%.6g eV  "
        "worst |dE_ext|/scale=%.3e  tol=%.2f  max Te=%.3e eV  rHH=%.3f Å\n",
        mu_au, k0_ha, tau_e_fs, eval.last_result().n_kpts, n_prod, e0,
        pe_f + ke_f + te_f + eth_f, worst_drift, drift_tol, worst_te, rhh);
    std::printf("  PE0=%.6g  KE0=%.6g  Te0=%.6g  Eth0=%.6g  PEf=%.6g  "
                "KEf=%.6g  Tef=%.6g  Ethf=%.6g eV  prod SCF iters=%d "
                "(expect 0)\n",
                pe0, ke0, te0, eth0, pe_f, ke_f, te_f, eth_f, last_iters);

    if (!(worst_te <= te_ceil_ev)) {
        std::printf("FAIL: Te runaway max Te=%.3e eV > ceiling %.3e eV\n",
                    worst_te, te_ceil_ev);
        return 1;
    }
    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: E_ext drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }

    std::printf("PASS: cpmd-uks-thermostat-kpoints (multi-k UKS CP + "
                "electronic NH E_ext envelope + closed-shell consistency)\n");
    return 0;
}
