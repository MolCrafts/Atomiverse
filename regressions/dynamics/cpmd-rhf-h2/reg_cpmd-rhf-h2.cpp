/// @file regressions/dynamics/cpmd-rhf-h2/reg_cpmd-rhf-h2.cpp
/// @brief END-TO-END: Car–Parrinello MD on an AO-basis mean-field method —
///        RHF/STO-3G H₂ NVE, CP conserved-energy drift envelope pin.
///
/// The AO twin of `dynamics/cpmd-water` (which runs CP on GPW): the fictitious
/// electronic degrees of freedom are the *occupied MO coefficients* of a
/// Gaussian-basis `cpu::scf::RHF`, propagated on the nuclear clock instead of
/// re-converging the SCF at every step. This is the family's machinery case —
/// it pins the cold SCF seed, the zero-iteration production steps, the
/// `kConserved` assembly channel, and the dt-from-integrator protocol.
///
/// \f$E_{\mathrm{cons}}\f$ is **not** re-added here from method readbacks: the
/// engine publishes it on `frame.meta[kConserved]` [eV] every step through
/// `cpu::md::CpmdConserved`, and this case reads that one channel — the same
/// number a user sees in Thermo's `Conserved` column or from a drift
/// `Criterion`. The case does re-*assemble* the four terms once per step to
/// prove the published channel is the documented sum, not an independent
/// accumulator.
///
/// **Honest scope, read REFERENCE.md:** H₂ is homonuclear, so its AO Fock is
/// persymmetric at every bond length and the occupied σ_g vector is an exact
/// S-metric eigenvector throughout the trajectory. The CP band force therefore
/// vanishes identically and \f$T_e \equiv 0\f$ by symmetry — a physics fact,
/// not a broken propagator. This case is still the right machinery pin
/// (seed → zero-iteration production → conserved channel); the *live*
/// electronic DOF is exercised by `cpmd-rks-water` and `cpmd-uhf-o2`.
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
#include "atomiverse/cpu/scf/rhf.h"

using namespace ATV;

int main() {
    // H2 near equilibrium (r = 0.74 Å — Herzberg r_e = 0.7414 Å), stretched by
    // a small COM-preserving mode so the nuclei genuinely move. The molecule
    // sits at the centre of a 20 Å cell: the SCF is non-periodic and ignores
    // the cell, but MD wraps coordinates into it, so a nucleus parked on a
    // face would teleport across the box and wreck the bond-length gate.
    cpu::Frame frame(2);
    const double x0[2] = {9.63, 10.37};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1; // Z
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = 10.0;
        frame.span(kAtom_z)[i] = 10.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    // COM-preserving stretch [Å/fs] — small KE, no net translation.
    frame.span(kAtom_vx)[0] = 0.002;
    frame.span(kAtom_vx)[1] = -0.002;
    frame.box = cpu::Box::cubic(20.0); // isolated molecule, centred

    // Nuclear timestep. The electrons share it by construction: `MD` pushes it
    // from `VelocityVerlet` (TimestepProvider) into the method
    // (TimestepConsumer) at build — there is deliberately no `cpmd_dt` member.
    constexpr double dt = 0.05; // fs

    cpu::scf::RHF rhf_base;
    rhf_base.basis = cpu::scf::BasisSet::STO3G;
    rhf_base.tol = 1e-10;
    // Family-wide fictitious mass: the AO residual is stiffer than a
    // plane-wave one, and the textbook μ = 400 a.u. blows the sibling cases up
    // (see REFERENCE.md for the μ × dt scan). On H2 the drift is *identical*
    // to five digits for every scanned μ, because T_e ≡ 0 by persymmetry —
    // the value is carried here only to keep the family on one setting.
    constexpr double mu_au = 1000.0;
    cpu::electronic::CarParrinelloMethod<cpu::scf::RHF> rhf{std::move(rhf_base),
                                                            mu_au};

    constexpr int n_prod = 40;
    // Envelope: 12× the worst observed relative drift (4.20e-07) over these
    // 40 production steps — headroom for BLAS/threading variance, still five
    // orders of magnitude tighter than the GPW CP case (cpmd-water, 0.10).
    constexpr double drift_tol = 5.0e-6;
    constexpr double sum_tol = 1.0e-9; // eV, kConserved re-assembly residual
    // T_e is *identically* zero on H2 by persymmetry (see the file header);
    // observed max is 2.3e-32 eV, so 1e-20 eV is twelve orders of headroom and
    // still a real gate on the CP band-force projector (I − S C Cᵀ).
    constexpr double te_symmetry_tol = 1.0e-20; // eV
    constexpr double hh_min = 0.50;             // Å
    constexpr double hh_max = 1.20;             // Å

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(rhf);
    // E_cons is assembled by the engine: CpmdConserved sums
    // E_SCF + T_nuc + T_e + E_therm^e onto frame.meta[kConserved] [eV] at its
    // default slot (end_of_step @ -90 — after KineticEnergy @ -100, so the
    // nuclear term is this step's). Owning add() takes that default binding.
    sim.set_observer(cpu::md::CpmdConserved{});
    sim.enable_timing(false);
    sim.build();

    // ── Step 1: the cold seed — one full converged SCF + BO forces ────────
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: seed step did not run\n");
        return 1;
    }
    if (!(rhf.iterations() > 0)) {
        std::printf("FAIL: seed step ran %d SCF iterations — the cold seed "
                    "must converge a real SCF\n",
                    rhf.iterations());
        return 1;
    }
    if (!rhf.converged()) {
        std::printf("FAIL: seed SCF did not converge (%s)\n",
                    rhf.stop_reason().c_str());
        return 1;
    }
    {
        const double pe_seed = frame.meta.get(kEnergy);
        const double te_seed = frame.meta.get(kEnergyElectronKinetic);
        // RHF/STO-3G H2 at r_e is ≈ −1.1167 Ha ≈ −30.4 eV: a bound molecule.
        if (!std::isfinite(pe_seed) || !(pe_seed < -25.0)
            || !(pe_seed > -40.0)) {
            std::printf("FAIL: seed PE %g eV outside the bound-H2 window "
                        "(−40, −25)\n",
                        pe_seed);
            return 1;
        }
        // Nothing has been propagated yet, so the fictitious KE is exactly 0.
        if (!(te_seed == 0.0)) {
            std::printf("FAIL: seed step already carries T_e = %g eV\n",
                        te_seed);
            return 1;
        }
    }

    // ── Step 2: the first CP production step — baseline for the drift ─────
    if (sim.run(frame, 1) != 1) {
        std::printf("FAIL: first CP production step did not run\n");
        return 1;
    }
    if (rhf.iterations() != 0) {
        std::printf("FAIL: first CP step re-ran SCF (iters=%d) — CPMD must not "
                    "be a BOMD rename\n",
                    rhf.iterations());
        return 1;
    }
    const double e0 = frame.meta.get(kConserved);
    if (!std::isfinite(e0)) {
        std::printf("FAIL: non-finite E_cons after the first CP step: %g eV\n",
                    e0);
        return 1;
    }
    const double e_scale = std::max(1.0, std::fabs(e0));

    // ── Production: every post-seed step is a fixed-density evaluation ────
    double worst_drift = 0.0;
    double worst_sum_residual = 0.0;
    double te_max = 0.0;
    for (int s = 0; s < n_prod; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: production step %d did not run\n", s + 1);
            return 1;
        }
        if (rhf.iterations() != 0) {
            std::printf("FAIL: production step %d re-ran SCF (iters=%d)\n",
                        s + 1, rhf.iterations());
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
        // No electronic thermostat is wired, so its channel must be exactly 0.
        if (!(eth == 0.0)) {
            std::printf("FAIL: NVE electrons but E_therm = %g eV at step %d\n",
                        eth, s + 1);
            return 1;
        }
        // The published readback must be the published frame channel.
        if (!(te == rhf.last_electronic_ke_ev())) {
            std::printf(
                "FAIL: T_e channel %g eV != readback %g eV at step %d\n", te,
                rhf.last_electronic_ke_ev(), s + 1);
            return 1;
        }
        // kConserved is documented as E_SCF + T_nuc + T_e + E_therm^e —
        // re-assemble it and prove the channel is that sum.
        const double residual = std::fabs(ec - (pe + ke + te + eth));
        if (residual > worst_sum_residual)
            worst_sum_residual = residual;

        const double drift = std::fabs(ec - e0) / e_scale;
        if (drift > worst_drift)
            worst_drift = drift;
        if (te > te_max)
            te_max = te;
    }

    // ── Structural sanity: the molecule is still a molecule ───────────────
    const double dx =
        static_cast<double>(frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0]);
    const double dy =
        static_cast<double>(frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0]);
    const double dz =
        static_cast<double>(frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0]);
    const double r_hh = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(r_hh > hh_min) || !(r_hh < hh_max)) {
        std::printf("FAIL: H–H distance %.4f Å left the physical window "
                    "(%.2f, %.2f)\n",
                    r_hh, hh_min, hh_max);
        return 1;
    }

    std::printf("CPMD H2 RHF/STO-3G NVE (mu=%.0f a.u., dt=%.3g fs): "
                "seed+1+%d steps  E0=%.8g eV  Ef=%.8g eV\n",
                mu_au, dt, n_prod, e0, frame.meta.get(kConserved));
    std::printf("  worst |dE_cons|/scale=%.3e (tol %.1e)  "
                "worst |Ec-(PE+KE+Te+Eth)|=%.3e eV (tol %.1e)\n",
                worst_drift, drift_tol, worst_sum_residual, sum_tol);
    std::printf("  PEf=%.8g eV  KEf=%.6g eV  Te_max=%.3e eV (0 by H2 "
                "persymmetry)  r(H–H)=%.4f Å  prod SCF iters=%d (expect 0)\n",
                frame.meta.get(kEnergy), frame.meta.get(kKinetic), te_max, r_hh,
                rhf.iterations());

    // H2's occupied σ_g vector is an exact S-metric eigenvector at every bond
    // length, so the CP band force −(I − S C Cᵀ)(∂E/∂C) must vanish and the
    // fictitious electrons must never move. A nonzero T_e here means the
    // projector or the multiplier weight is wrong, not that the physics is
    // interesting.
    if (!(te_max < te_symmetry_tol)) {
        std::printf("FAIL: T_e reached %.3e eV on homonuclear H2 (limit "
                    "%.1e) — the CP band force must vanish by persymmetry\n",
                    te_max, te_symmetry_tol);
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

    std::printf("PASS: cpmd-rhf-h2 (AO Car–Parrinello NVE on RHF, "
                "zero-SCF production + conserved-energy envelope)\n");
    return 0;
}
