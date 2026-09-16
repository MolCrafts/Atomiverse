/// @file regressions/dynamics/cpmd-uhf-o2/reg_cpmd-uhf-o2.cpp
/// @brief END-TO-END: Car–Parrinello MD on an unrestricted AO method —
///        UHF triplet O₂/STO-3G NVE, CP conserved-energy drift envelope pin.
///
/// The dual-channel leg of the AO Car–Parrinello family (`cpmd-rhf-h2` pins
/// the machinery, `cpmd-rks-water` the KS/XC branch). Triplet O₂ propagates
/// **two independent fictitious electronic subsystems** — 9 α and 7 β occupied
/// coefficient columns, each at occupancy 1 — so this case is the one that
/// fails if the β channel is dropped, mis-occupied, or aliased onto α.
///
/// O₂ is homonuclear like H₂, but unlike H₂ its symmetry blocks are
/// multi-dimensional (σ_g and σ_u are 3-dimensional in STO-3G), so the
/// occupied subspace is *not* frozen by persymmetry and \f$T_e\f$ is strictly
/// positive on every production step — verified below, not assumed.
///
/// \f$E_{\mathrm{cons}}\f$ comes off `frame.meta[kConserved]` [eV], published
/// each step by `cpu::md::CpmdConserved`.
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
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/uhf.h"

using namespace ATV;

int main() {
    // Triplet O2 near equilibrium (r = 1.21 Å — Huber & Herzberg r_e = 1.2075 Å
    // for the X³Σ_g⁻ ground state), given a small COM-preserving stretch. The
    // molecule sits at the centre of a 20 Å cell: the SCF is non-periodic and
    // ignores the cell, but MD wraps coordinates into it, so a nucleus parked
    // on a face would teleport across the box (STO-3G overstretches this bond,
    // so the atoms do drift outward from their start).
    cpu::Frame frame(2);
    const double x0[2] = {9.395, 10.605};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8; // Z
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = 10.0;
        frame.span(kAtom_z)[i] = 10.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // COM-preserving stretch [Å/fs] — small KE, no net translation.
    frame.span(kAtom_vx)[0] = 0.001;
    frame.span(kAtom_vx)[1] = -0.001;
    frame.box = cpu::Box::cubic(20.0); // isolated molecule, centred

    // Nuclear timestep — pushed into the method by `MD` at build
    // (TimestepProvider → TimestepConsumer); there is no `cpmd_dt` member.
    constexpr double dt = 0.05; // fs

    // 16 electrons, high-spin triplet: 1σg²1σu²2σg²2σu²3σg²1πu⁴1πg²(↑↑).
    cpu::scf::UHF uhf_base{/*n_alpha=*/9, /*n_beta=*/7};
    uhf_base.basis = cpu::scf::BasisSet::STO3G;
    uhf_base.tol = 1e-9;
    // Scanned μ ∈ {400, 1000, 2000} a.u. × dt ∈ {0.05, 0.1} fs (REFERENCE.md):
    // the plane-wave textbook μ = 400 a.u. loses adiabaticity on the stiffer AO
    // residual (T_e → 25 eV inside 120 steps); μ = 1000 at dt = 0.05 fs is the
    // stable window and is the family-wide setting.
    constexpr double mu_au = 1000.0;
    cpu::electronic::CarParrinelloMethod<cpu::scf::UHF> uhf{std::move(uhf_base),
                                                            mu_au};

    constexpr int n_prod = 40;
    // Envelope: 18× the worst observed relative drift (2.81e-07) over these
    // 40 production steps — headroom for BLAS/threading variance, still five
    // orders of magnitude tighter than the GPW CP case (cpmd-water, 0.10).
    constexpr double drift_tol = 5.0e-6;
    constexpr double sum_tol = 1.0e-9; // eV, kConserved re-assembly residual
    constexpr double te_max_tol = 1.0; // eV — adiabaticity: electrons stay cold
    // High-spin UHF O2 is only mildly spin-contaminated; ⟨S²⟩ − S(S+1) well
    // below 0.5 rules out a collapsed / broken-symmetry seed.
    constexpr double s2_contam_tol = 0.5;
    constexpr double oo_min = 1.00; // Å
    constexpr double oo_max = 1.45; // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(uhf);
    // CpmdConserved sums E_UHF + T_nuc + T_e(α)+T_e(β) + E_therm^e onto
    // frame.meta[kConserved] [eV] at end_of_step @ -90.
    sim.set_observer(cpu::md::CpmdConserved{});
    sim.enable_timing(false);
    sim.build();

    // ── Step 1: the cold seed — one full converged UHF SCF + BO forces ────
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: seed step did not run\n");
        return 1;
    }
    if (!(uhf.iterations() > 0)) {
        std::printf("FAIL: seed step ran %d SCF iterations — the cold seed "
                    "must converge a real SCF\n",
                    uhf.iterations());
        return 1;
    }
    if (!uhf.converged()) {
        std::printf("FAIL: seed UHF SCF did not converge (%s)\n",
                    uhf.stop_reason().c_str());
        return 1;
    }

    // ── Spin sanity of the seed determinant (the state the electrons are
    //    launched from; `determinant()` intentionally stays the seed SCF's
    //    wavefunction on production steps — propagated orbitals have no
    //    canonical eigenvalues).
    const double s2_contam = uhf.determinant().spin_contamination();
    const int n_a =
        uhf.determinant().n_electrons(cpu::scf::SlaterDeterminant::Spin::alpha);
    const int n_b =
        uhf.determinant().n_electrons(cpu::scf::SlaterDeterminant::Spin::beta);
    if (n_a != 9 || n_b != 7) {
        std::printf("FAIL: seed determinant carries N_alpha=%d N_beta=%d, "
                    "expected 9 / 7 (triplet O2)\n",
                    n_a, n_b);
        return 1;
    }
    if (!std::isfinite(s2_contam) || !(std::fabs(s2_contam) < s2_contam_tol)) {
        std::printf("FAIL: seed spin contamination <S^2>-S(S+1) = %g exceeds "
                    "%.2f — not a clean high-spin triplet\n",
                    s2_contam, s2_contam_tol);
        return 1;
    }
    {
        const double pe_seed = frame.meta.get(kEnergy);
        // UHF/STO-3G triplet O2 is ≈ −147.6 Ha ≈ −4016 eV: a bound molecule.
        if (!std::isfinite(pe_seed) || !(pe_seed < -3800.0)
            || !(pe_seed > -4200.0)) {
            std::printf("FAIL: seed PE %g eV outside the bound-O2 window "
                        "(−4200, −3800)\n",
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
    if (uhf.iterations() != 0) {
        std::printf("FAIL: first CP step re-ran SCF (iters=%d) — CPMD must not "
                    "be a BOMD rename\n",
                    uhf.iterations());
        return 1;
    }
    const double e0 = frame.meta.get(kConserved);
    if (!std::isfinite(e0)) {
        std::printf("FAIL: non-finite E_cons after the first CP step: %g eV\n",
                    e0);
        return 1;
    }
    const double e_scale = std::max(1.0, std::fabs(e0));

    // ── Production: every post-seed step is a fixed-density UHF evaluation ─
    double worst_drift = 0.0;
    double worst_sum_residual = 0.0;
    double te_max = 0.0;
    double te_min = 1e300;
    for (int s = 0; s < n_prod; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: production step %d did not run\n", s + 1);
            return 1;
        }
        if (uhf.iterations() != 0) {
            std::printf("FAIL: production step %d re-ran SCF (iters=%d)\n",
                        s + 1, uhf.iterations());
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
    const double dx =
        static_cast<double>(frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0]);
    const double dy =
        static_cast<double>(frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0]);
    const double dz =
        static_cast<double>(frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0]);
    const double r_oo = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(r_oo > oo_min) || !(r_oo < oo_max)) {
        std::printf("FAIL: O–O distance %.4f Å left the physical window "
                    "(%.2f, %.2f)\n",
                    r_oo, oo_min, oo_max);
        return 1;
    }

    std::printf("CPMD O2 UHF triplet/STO-3G NVE (mu=%.0f a.u., dt=%.3g fs, "
                "N_alpha/N_beta=%d/%d): seed+1+%d steps  E0=%.9g eV  "
                "Ef=%.9g eV\n",
                mu_au, dt, n_a, n_b, n_prod, e0,
                frame.meta.get(kConserved));
    std::printf("  worst |dE_cons|/scale=%.3e (tol %.1e)  "
                "worst |Ec-(PE+KE+Te+Eth)|=%.3e eV (tol %.1e)\n",
                worst_drift, drift_tol, worst_sum_residual, sum_tol);
    std::printf("  PEf=%.9g eV  KEf=%.6g eV  Te in [%.3e, %.3e] eV  "
                "<S^2>-S(S+1)=%.4f  r(O–O)=%.4f Å  prod SCF iters=%d "
                "(expect 0)\n",
                frame.meta.get(kEnergy), frame.meta.get(kKinetic), te_min,
                te_max, s2_contam, r_oo, uhf.iterations());

    // Both fictitious channels must be alive: O2's multi-dimensional symmetry
    // blocks leave the occupied subspace free to move (unlike H2).
    if (!(te_min > 0.0)) {
        std::printf("FAIL: T_e stayed at %g eV — the fictitious electronic "
                    "DOF never moved, so this is BOMD in disguise\n",
                    te_min);
        return 1;
    }
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

    std::printf("PASS: cpmd-uhf-o2 (AO Car–Parrinello NVE on UHF triplet, "
                "dual-channel fictitious electronic DOF at occ = 1)\n");
    return 0;
}
