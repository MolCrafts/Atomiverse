/// @file regressions/dynamics/cpmd-water/reg_cpmd-water.cpp
/// @brief END-TO-END: Car–Parrinello MD (fictitious electronic DOF) NVE on
///        H₂O with GPW — CP conserved-energy drift envelope pin.
///
/// \f$E_{\mathrm{cons}}\f$ is not re-added here from evaluator readbacks: the
/// engine publishes it on `frame.meta[kConserved]` [eV] every step through
/// `cpu::md::CpmdConserved`, and this case reads that one channel — the same
/// number a user sees in Thermo's `Conserved` column or from a drift
/// `Criterion`.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_GPW + MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/cpmd_conserved.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Isolated water, CP2K getting-started geometry (Å) — BOMD/XL twin family,
    // now with Car–Parrinello fictitious electronic dynamics (not SCF each
    // production step, not XL auxiliary density).
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

    // Cheaper than forces pin (200 Ry / tol 1e-5) so cold SCF seed stays
    // inside the ≤60 s gate. CPMD production steps are single KS evals.
    constexpr double dt = 0.1; // fs — short step for electronic stability
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
    // Heavier fictitious mass than plane-wave textbook 400 a.u. — AO residual
    // curvature on DZVP water is stiffer; 2000 keeps short NVE stable at 0.1 fs.
    cpu::electronic::CarParrinello eval{std::move(gpw), /*mu=*/2000.0};
    // No electronic dt here: `MD` pushes the nuclear dt from the integrator
    // (TimestepProvider) into the evaluator at build, so `dt` has one owner.

    // Drift production after the baseline CP step that engages Te.
    constexpr int n_prod = 4;
    constexpr int n_chunks = 2;
    // CP conserved-energy envelope (softer than BO 0.05 — Löwdin foothold).
    constexpr double drift_tol = 0.10;
    constexpr double oh_max = 1.8; // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(eval);
    sim.set_observer(cpu::md::CpmdConserved{});
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

    // First CP production step engages fixed-density eval + electronic advance
    // (Te becomes nonzero). Baseline E_cons after this step so the Te turn-on
    // is not counted as secular drift.
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
    // T_e comes off the frame, not off the evaluator: the CPMD evaluator
    // publishes kEnergyElectronKinetic / kEnergyElectronThermostat on every
    // launch, and CpmdConserved has already folded both into kConserved below.
    const double te0 = frame.meta.get(kEnergyElectronKinetic);
    if (!std::isfinite(pe0) || !std::isfinite(ke0) || !std::isfinite(te0)
        || !(ke0 >= 0.0) || !(te0 >= 0.0)) {
        std::printf("FAIL: non-finite energy after first CP step PE=%g KE=%g "
                    "Te=%g\n",
                    pe0, ke0, te0);
        return 1;
    }
    const double e0 = frame.meta.get(kConserved);
    if (!std::isfinite(e0)) {
        std::printf("FAIL: non-finite E_cons after first CP step: %g eV\n", e0);
        return 1;
    }
    const double e_scale = std::max(1.0, std::fabs(e0));

    double worst_drift = 0.0;
    int last_iters = eval.last_result().iters;
    // Remaining production steps for the drift envelope (after baseline).
    const int steps_per = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (sim.run(frame, steps_per) != steps_per) {
            std::printf("FAIL: production chunk %d incomplete\n", c);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        const double te = frame.meta.get(kEnergyElectronKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !std::isfinite(te)
            || !(ke >= 0.0) || !(te >= 0.0)) {
            std::printf(
                "FAIL: non-finite energy at chunk %d PE=%g KE=%g Te=%g\n", c,
                pe, ke, te);
            return 1;
        }
        // Production CP steps must be fixed-density evals (iters==0), not SCF.
        last_iters = eval.last_result().iters;
        if (last_iters != 0) {
            std::printf(
                "FAIL: production step re-ran SCF (iters=%d) — CPMD must not "
                "be a BOMD rename\n",
                last_iters);
            return 1;
        }
        const double e_cons = frame.meta.get(kConserved);
        if (!std::isfinite(e_cons)) {
            std::printf("FAIL: non-finite E_cons at chunk %d: %g eV\n", c,
                        e_cons);
            return 1;
        }
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
    const double te_f = frame.meta.get(kEnergyElectronKinetic);
    std::printf(
        "CPMD H2O NVE (mu=2000 a.u.): steps=%d  E0=%.6g eV  Ef=%.6g eV  "
        "worst |dE_cons|/scale=%.3e  tol=%.2f  oh=%.3f/%.3f Å\n",
        n_prod, e0, frame.meta.get(kConserved), worst_drift, drift_tol, oh1,
        oh2);
    std::printf(
        "  PE0=%.6g  KE0=%.6g  Te0=%.6g  PEf=%.6g  KEf=%.6g  Tef=%.6g eV  "
        "prod SCF iters=%d (expect 0)\n",
        pe0, ke0, te0, pe_f, ke_f, te_f, last_iters);

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: CP energy drift %.3e exceeds tol %.2f\n",
                    worst_drift, drift_tol);
        return 1;
    }

    std::printf("PASS: cpmd-water (CPMD NVE conserved-energy envelope, "
                "fictitious e- DOF)\n");
    return 0;
}
