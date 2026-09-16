/// @file regressions/dynamics/cpmd-rks-water/reg_cpmd-rks-water.cpp
/// @brief END-TO-END: Car–Parrinello MD on an AO-basis Kohn–Sham method —
///        RKS/LSDA/STO-3G H₂O NVE, CP conserved-energy drift envelope pin.
///
/// The Kohn–Sham leg of the AO Car–Parrinello family (`cpmd-rhf-h2` pins the
/// machinery; `cpmd-uhf-o2` pins the dual-spin channel). Unlike H₂, water has
/// no persymmetry to freeze the occupied subspace, so the fictitious
/// electronic degrees of freedom genuinely move: \f$T_e\f$ is strictly
/// positive on every production step, which is what makes this case a test of
/// the **fixed-density KS energy branch and the XC gradient at a
/// non-stationary density** rather than of a disguised Born–Oppenheimer run.
///
/// Same nuclei as the shipped GPW Car–Parrinello case
/// (`dynamics/cpmd-water`, CP2K getting-started geometry): the AO twin of that
/// H₂O NVE trajectory, with the Gaussian SCF method in the force slot instead
/// of the GPW evaluator.
///
/// \f$E_{\mathrm{cons}}\f$ comes off `frame.meta[kConserved]` [eV], published
/// each step by `cpu::md::CpmdConserved` — the case re-assembles the four
/// terms only to prove the channel is the documented sum.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_HF + ATV_BUILD_MD.
/// Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/electronic/aimd_method.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/cpmd_conserved.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;

int main() {
    // Isolated water, CP2K getting-started geometry (Å) — the same nuclei as
    // the GPW twin `dynamics/cpmd-water`.
    cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double mass[3] = {15.999, 1.008, 1.008};
    const double x0[3] = {5.0000, 5.7586, 4.2414};
    const double y0[3] = {5.0000, 5.0000, 5.0000};
    const double z0[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i]; // Z
        frame.span(kAtom_mass)[i] = mass[i];
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = y0[i];
        frame.span(kAtom_z)[i] = z0[i];
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // COM-preserving H stretch mode [Å/fs] — same kick as the GPW twin.
    frame.span(kAtom_vx)[1] = 0.0015;
    frame.span(kAtom_vx)[2] = -0.0015;
    frame.box = cpu::Box::cubic(10.0); // isolated molecule (SCF ignores PBC)

    // Nuclear timestep — pushed into the method by `MD` at build
    // (TimestepProvider → TimestepConsumer); there is no `cpmd_dt` member.
    constexpr double dt = 0.05; // fs

    cpu::scf::RKS rks_base{cpu::scf::XCType::LSDA};
    rks_base.basis = cpu::scf::BasisSet::STO3G;
    rks_base.tol = 1e-9;
    // Scanned μ ∈ {400, 1000, 2000} a.u. × dt ∈ {0.05, 0.1} fs (REFERENCE.md):
    // the plane-wave textbook μ = 400 a.u. loses adiabaticity on the stiffer AO
    // residual and diverges inside 120 steps; μ = 1000 at dt = 0.05 fs is the
    // stable window and is the family-wide setting.
    constexpr double mu_au = 1000.0;
    cpu::electronic::CarParrinelloMethod<cpu::scf::RKS> rks{std::move(rks_base),
                                                            mu_au};

    constexpr int n_prod = 40;
    // Envelope: 8.5× the worst observed relative drift (5.85e-06) over these
    // 40 production steps — headroom for BLAS/threading variance, still three
    // orders of magnitude tighter than the GPW CP case (cpmd-water, 0.10).
    constexpr double drift_tol = 5.0e-5;
    constexpr double sum_tol = 1.0e-9; // eV, kConserved re-assembly residual
    constexpr double te_max_tol = 1.0; // eV — adiabaticity: electrons stay cold
    constexpr double oh_max = 1.30;    // Å
    constexpr double oh_min = 0.70;    // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(rks);
    // CpmdConserved sums E_KS + T_nuc + T_e + E_therm^e onto
    // frame.meta[kConserved] [eV] at end_of_step @ -90 (after KineticEnergy
    // @ -100, so the nuclear term is this step's).
    sim.set_observer(cpu::md::CpmdConserved{});
    sim.enable_timing(false);
    sim.build();

    // ── Step 1: the cold seed — one full converged KS SCF + BO forces ─────
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: seed step did not run\n");
        return 1;
    }
    if (!(rks.iterations() > 0)) {
        std::printf("FAIL: seed step ran %d SCF iterations — the cold seed "
                    "must converge a real SCF\n",
                    rks.iterations());
        return 1;
    }
    if (!rks.converged()) {
        std::printf("FAIL: seed KS SCF did not converge (%s)\n",
                    rks.stop_reason().c_str());
        return 1;
    }
    {
        const double pe_seed = frame.meta.get(kEnergy);
        // LSDA/STO-3G water is ≈ −74.7 Ha ≈ −2034 eV: a bound molecule.
        if (!std::isfinite(pe_seed) || !(pe_seed < -1900.0)
            || !(pe_seed > -2200.0)) {
            std::printf("FAIL: seed PE %g eV outside the bound-H2O window "
                        "(−2200, −1900)\n",
                        pe_seed);
            return 1;
        }
        if (!(frame.meta.get(kEnergyElectronKinetic) == 0.0)) {
            std::printf("FAIL: seed step already carries T_e = %g eV\n",
                        frame.meta.get(kEnergyElectronKinetic));
            return 1;
        }
    }

    // ── Step 2: the first CP production step — baseline for the drift ─────
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: first CP production step did not run\n");
        return 1;
    }
    if (rks.iterations() != 0) {
        std::printf("FAIL: first CP step re-ran SCF (iters=%d) — CPMD must not "
                    "be a BOMD rename\n",
                    rks.iterations());
        return 1;
    }
    const double e0 = frame.meta.get(kConserved);
    if (!std::isfinite(e0)) {
        std::printf("FAIL: non-finite E_cons after the first CP step: %g eV\n",
                    e0);
        return 1;
    }
    const double e_scale = std::max(1.0, std::fabs(e0));

    // ── Production: every post-seed step is a fixed-density KS evaluation ─
    double worst_drift = 0.0;
    double worst_sum_residual = 0.0;
    double te_max = 0.0;
    double te_min = 1e300;
    for (int s = 0; s < n_prod; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: production step %d did not run\n", s + 1);
            return 1;
        }
        if (rks.iterations() != 0) {
            std::printf("FAIL: production step %d re-ran SCF (iters=%d)\n",
                        s + 1, rks.iterations());
            return 1;
        }

        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        const double te = frame.meta.get(kEnergyElectronKinetic);
        const double eth = frame.meta.get(kEnergyElectronThermostat);
        const double ec = frame.meta.get(kConserved);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !std::isfinite(te)
            || !std::isfinite(eth) || !std::isfinite(ec) || !(ke >= 0.0)
            || !(te >= 0.0)) {
            std::printf("FAIL: non-finite/negative energy at step %d "
                        "PE=%g KE=%g Te=%g Eth=%g Ec=%g\n",
                        s + 1, pe, ke, te, eth, ec);
            return 1;
        }
        if (!(eth == 0.0)) {
            std::printf("FAIL: NVE electrons but E_therm = %g eV at step %d\n",
                        eth, s + 1);
            return 1;
        }
        const double residual = std::fabs(ec - (pe + ke + te + eth));
        if (residual > worst_sum_residual)
            worst_sum_residual = residual;

        const double drift = std::fabs(ec - e0) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;
        if (te > te_max)
            te_max = te;
        if (te < te_min)
            te_min = te;
    }

    // ── Structural sanity: the molecule is still a molecule ───────────────
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
    if (!(oh1 > oh_min) || !(oh1 < oh_max) || !(oh2 > oh_min)
        || !(oh2 < oh_max)) {
        std::printf("FAIL: O–H distances %.4f / %.4f Å left the physical "
                    "window (%.2f, %.2f)\n",
                    oh1, oh2, oh_min, oh_max);
        return 1;
    }

    std::printf("CPMD H2O RKS(LSDA)/STO-3G NVE (mu=%.0f a.u., dt=%.3g fs): "
                "seed+1+%d steps  E0=%.8g eV  Ef=%.8g eV\n",
                mu_au, dt, n_prod, e0, frame.meta.get(kConserved));
    std::printf("  worst |dE_cons|/scale=%.3e (tol %.1e)  "
                "worst |Ec-(PE+KE+Te+Eth)|=%.3e eV (tol %.1e)\n",
                worst_drift, drift_tol, worst_sum_residual, sum_tol);
    std::printf("  PEf=%.8g eV  KEf=%.6g eV  Te in [%.3e, %.3e] eV  "
                "OH=%.4f/%.4f Å  prod SCF iters=%d (expect 0)\n",
                frame.meta.get(kEnergy), frame.meta.get(kKinetic), te_min,
                te_max, oh1, oh2, rks.iterations());

    // The electronic DOF must be alive (this is not H2's frozen subspace) …
    if (!(te_min > 0.0)) {
        std::printf("FAIL: T_e stayed at %g eV — the fictitious electronic "
                    "DOF never moved, so this is BOMD in disguise\n",
                    te_min);
        return 1;
    }
    // … and adiabatically cold (CP's whole premise: T_e << T_nuc scale).
    if (!(te_max < te_max_tol)) {
        std::printf("FAIL: T_e reached %.3e eV (limit %.2f) — adiabaticity "
                    "lost, the electrons are heating\n",
                    te_max, te_max_tol);
        return 1;
    }
    if (!(worst_sum_residual <= sum_tol)) {
        std::printf("FAIL: kConserved is not E+KE+Te+Eth (residual %.3e eV "
                    "exceeds %.1e)\n",
                    worst_sum_residual, sum_tol);
        return 1;
    }
    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: CP energy drift %.3e exceeds tol %.1e\n",
                    worst_drift, drift_tol);
        return 1;
    }

    std::printf("PASS: cpmd-rks-water (AO Car–Parrinello NVE on RKS/LSDA, "
                "fixed-density KS branch + XC gradient at a moving density)\n");
    return 0;
}
