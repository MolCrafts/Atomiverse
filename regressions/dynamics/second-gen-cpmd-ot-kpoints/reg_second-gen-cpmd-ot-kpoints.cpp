/// @file regressions/dynamics/second-gen-cpmd-ot-kpoints/reg_second-gen-cpmd-ot-kpoints.cpp
/// @brief END-TO-END: multi-k second-generation CP with **OT partial-SCF
///        corrector** (Kolafa ASPC on complex \(P_k\) → natural-orbital
///        warm start of the per-k OT manifold) NVE on periodic H₂ with GPW
///        forces — corrector A/B identity + energy-drift + budget pin.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/electronic/second_gen_cp.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Periodic H₂ — the second-gen-cpmd-kpoints input with the corrector
    // switched to OT (DZV-GTH-PADE, PADE XC, mesh 1×1×2, unreduced).
    constexpr double bohr2ang = 0.52917720859;
    constexpr double L_bohr = 12.0;
    constexpr double L_ang = L_bohr * bohr2ang;
    constexpr double r0_bohr = 1.0;
    const double x0[2] = {(L_bohr / 2.0 - 0.5 * r0_bohr) * bohr2ang,
                          (L_bohr / 2.0 + 0.5 * r0_bohr) * bohr2ang};

    const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");

    // ── A: corrector A/B identity at the initial geometry ───────────────
    // Full-budget multi-k OT vs multi-k diagonalization/DIIS must reach the
    // same SCF fixed point (the ot-kpoints pin level, |ΔE| ≤ 1e-4 Ha) —
    // the OT corrector is the same functional, only the minimizer differs.
    {
        const double ra[3] = {0.5 * L_bohr - 0.5 * r0_bohr, 0.5 * L_bohr,
                              0.5 * L_bohr};
        const double rb[3] = {0.5 * L_bohr + 0.5 * r0_bohr, 0.5 * L_bohr,
                              0.5 * L_bohr};
        std::vector<cpu::gpw::GpwAtom> atoms;
        atoms.emplace_back(ra, bH, pH, 1);
        atoms.emplace_back(rb, bH, pH, 1);

        auto make = [&](bool ot) {
            cpu::gpw::GpwDft m;
            m.cell[0] = m.cell[1] = m.cell[2] = L_bohr;
            m.cutoff_ry = 100.0;
            m.n_occ = 1;
            m.max_iter = 200;
            m.tol = 1e-7;
            m.xc = cpu::gpw::XcKind::PADE;
            m.k_mesh[0] = 1;
            m.k_mesh[1] = 1;
            m.k_mesh[2] = 2;
            m.k_time_reversal = false;
            m.use_ot = ot;
            return m;
        };
        cpu::gpw::GpwDft diag = make(false);
        diag.build(atoms);
        const auto rd = diag.run(atoms);
        cpu::gpw::GpwDft ot = make(true);
        ot.build(atoms);
        const auto ro = ot.run(atoms);
        if (!rd.converged || !ro.converged) {
            std::printf("FAIL: A/B baseline SCF not converged "
                        "(diag=%d ot=%d)\n",
                        rd.converged ? 1 : 0, ro.converged ? 1 : 0);
            return 1;
        }
        const double dab = std::fabs(rd.e_total - ro.e_total);
        std::printf("A/B: E_diag=%.10f Ha  E_OT=%.10f Ha  |Δ|=%.3e Ha  "
                    "iters diag/ot=%d/%d  nk=%d\n",
                    rd.e_total, ro.e_total, dab, rd.iters, ro.iters,
                    ro.n_kpts);
        if (!(dab <= 1e-4)) {
            std::printf("FAIL: |E_diag - E_OT| = %.3e Ha exceeds 1e-4 Ha\n",
                        dab);
            return 1;
        }
        if (ro.n_kpts != 2) {
            std::printf("FAIL: expected n_kpts == 2 on 1×1×2 TR-off mesh, "
                        "got %d\n",
                        ro.n_kpts);
            return 1;
        }
    }

    // ── B: SGCP NVE with the OT corrector on the k path ─────────────────
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = L_ang / 2.0;
        frame.span(kAtom_z)[i] = L_ang / 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_ang);

    const cpu::gpw::Kind H{molrs::Element::H, bH, pH};

    constexpr double dt = 0.5; // fs
    constexpr int corrector_budget = 5;
    constexpr int n_seed = 2; // ASPC K=1 needs K+1 densities
    constexpr int n_prod = 2;
    constexpr double drift_tol = 0.10;
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false;
    gpw.use_ot = true; // OT partial-SCF corrector (this slug)
    cpu::electronic::SecondGenCp eval{std::move(gpw), 1, corrector_budget};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();

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
            std::printf(
                "FAIL: expected non-trivial KE after seed, got %g eV\n", ke_w);
            return 1;
        }
        if (!(pe_w < -5.0)) {
            std::printf("FAIL: PE not bound after seed PE=%g eV\n", pe_w);
            return 1;
        }
        if (eval.last_result().n_kpts < 2) {
            std::printf("FAIL: expected n_kpts >= 2, got %d\n",
                        eval.last_result().n_kpts);
            return 1;
        }
        if (static_cast<int>(eval.last_result().P_k.size())
            != eval.last_result().n_kpts) {
            std::printf("FAIL: P_k payload missing after multi-k seed SCF\n");
            return 1;
        }
    }

    // First warm ASPC step — the OT corrector must obey the budget.
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: first SGCP-OT multi-k production step "
                    "did not run\n");
        return 1;
    }
    const int iters_first = eval.last_result().iters;
    if (!(iters_first <= corrector_budget)) {
        std::printf(
            "FAIL: first multi-k SGCP-OT step iters=%d exceeds corrector "
            "budget %d (not partial OT)\n",
            iters_first, corrector_budget);
        return 1;
    }
    const double pe0 = frame.meta.get(kEnergy);
    const double ke0 = frame.meta.get(kKinetic);
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !(ke0 >= 0.0)) {
        std::printf("FAIL: non-finite energy after first SGCP-OT PE=%g "
                    "KE=%g\n",
                    pe0, ke0);
        return 1;
    }
    const double e0 = pe0 + ke0;
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    int max_prod_iters = iters_first;
    if (sim.run(frame, n_prod) != n_prod) {
        std::printf("FAIL: production steps incomplete\n");
        return 1;
    }
    {
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite energy in production PE=%g KE=%g\n",
                        pe, ke);
            return 1;
        }
        const int last_iters = eval.last_result().iters;
        if (last_iters > max_prod_iters)
            max_prod_iters = last_iters;
        if (last_iters > corrector_budget) {
            std::printf(
                "FAIL: production iters=%d > corrector budget %d — multi-k "
                "SGCP-OT must not run full BOMD OT\n",
                last_iters, corrector_budget);
            return 1;
        }
        worst_drift = std::fabs((pe + ke) - e0) / e_scale;
        for (int i = 0; i < 2; ++i) {
            if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[i]))
                || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[i]))
                || !std::isfinite(
                    static_cast<double>(frame.span(kAtom_z)[i]))) {
                std::printf("FAIL: non-finite coordinate atom %d\n", i);
                return 1;
            }
        }
    }

    const double dx = static_cast<double>(frame.span(kAtom_x)[1]
                                          - frame.span(kAtom_x)[0]);
    const double dy = static_cast<double>(frame.span(kAtom_y)[1]
                                          - frame.span(kAtom_y)[0]);
    const double dz = static_cast<double>(frame.span(kAtom_z)[1]
                                          - frame.span(kAtom_z)[0]);
    const double rhh = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(rhh > rmin) || !(rhh < rmax)) {
        std::printf("FAIL: H–H bond out of window r=%.3f Å (%.2f–%.2f)\n", rhh,
                    rmin, rmax);
        return 1;
    }

    const double pe_f = frame.meta.get(kEnergy);
    const double ke_f = frame.meta.get(kKinetic);
    std::printf(
        "SGCP-OT multi-k H2 NVE (ASPC K=1, OT corrector=%d, nk=%d): seed=%d "
        "prod=%d  E0=%.6g eV  Ef=%.6g eV  worst |dE|/scale=%.3e  tol=%.2f  "
        "rHH=%.3f Å\n",
        corrector_budget, eval.last_result().n_kpts, n_seed, n_prod, e0,
        pe_f + ke_f, worst_drift, drift_tol, rhh);
    std::printf(
        "  PE0=%.6g  KE0=%.6g  PEf=%.6g  KEf=%.6g eV  max prod OT iters=%d "
        "(budget %d)\n",
        pe0, ke0, pe_f, ke_f, max_prod_iters, corrector_budget);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: energy drift %.3e exceeds tol %.2f\n", worst_drift,
                    drift_tol);
        return 1;
    }

    std::printf("PASS: second-gen-cpmd-ot-kpoints (multi-k SGCP NVE, ASPC on "
                "P_k + OT partial-SCF corrector, A/B corrector identity)\n");
    return 0;
}
