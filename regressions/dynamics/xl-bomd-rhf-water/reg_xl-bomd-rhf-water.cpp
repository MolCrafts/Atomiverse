/// @file regressions/dynamics/xl-bomd-rhf-water/reg_xl-bomd-rhf-water.cpp
/// @brief END-TO-END: extended-Lagrangian BOMD (Niklasson) as the initial
///        guess of a Gaussian-basis SCF — RHF/STO-3G H₂O NVE MD, warm-step
///        SCF-iteration reduction against **two** twins plus the drift
///        envelope.
///
/// The AO leg of the XL-BOMD family (`dynamics/xl-bomd-water` pins the same
/// propagator behind GPW forces). Same nuclei as the AO Car–Parrinello water
/// case (`dynamics/cpmd-rks-water`, CP2K getting-started geometry, centred in
/// the cell): three ways of answering "what is the density now" on one
/// trajectory — Car–Parrinello propagates the *coefficients*, XL-BOMD
/// propagates an *auxiliary density*, plain BOMD just reuses the last one.
///
/// ## What is pinned, and why it needs three legs
///
/// XL-BOMD's claim is that the auxiliary density — a dynamical variable with
/// its own (almost time-reversible) equation of motion — is a **better SCF
/// guess** than the previous step's converged density. "Better" is only
/// meaningful against a baseline, and there are two different baselines:
///
///   - **cold twin** — a brand-new `RHF` at every geometry the XL run visited
///     (no warm information of any kind). This is the acceptance-criterion
///     baseline: warm ≪ cold is what *any* working guess propagation must
///     deliver, and it is worth ~2.6 iterations here.
///   - **plain-BOMD twin** — a second `RHF` on the same trajectory with
///     `xl_bomd = false`, i.e. last-density reuse. This is the *specific*
///     XL claim (the auxiliary density beats naive reuse) and it is worth
///     ~0.9 iterations here — a real but modest margin, because this SCF
///     exits on |ΔE| < `tol` and that counter is coarse: see REFERENCE.md
///     §"Measured landscape" for the guess-error measurement (XL's guess is
///     **128–507× more accurate** than reuse at this timestep) and for the
///     measured dt dependence.
///
/// **Timestep.** dt = 0.25 fs, *not* the 0.5 fs of the CP twins. The auxiliary
/// density is a harmonic oscillator with κ = ω²Δt² (Table I: κ = 1.82), so it
/// only tracks the true density while the nuclear motion is slow compared with
/// that oscillator. Measured on this system (REFERENCE.md): the XL guess is
/// 128–507× more accurate than reuse at dt = 0.25 fs, 3.9–7.9× at dt = 0.5 fs,
/// and **worse than reuse** at dt = 1.0 fs, where ω_OH·Δt ≈ 0.7 breaks the
/// separation. 0.25 fs is ~36 steps per O–H period — the regime XL-BOMD is
/// designed for.
///
/// A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009).
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_HF + ATV_BUILD_MD.
/// Exit non-zero on failure.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/aimd_method.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

/// @brief Isolated water at the CP2K getting-started geometry [Å], centred in
///        a 10 Å cube — the nuclei of `dynamics/cpmd-rks-water`.
/// @return Three-atom Frame with a COM-preserving H-stretch kick [Å/fs].
cpu::Frame make_water() {
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
    frame.span(kAtom_vx)[1] = 0.0015;
    frame.span(kAtom_vx)[2] = -0.0015;
    frame.box = cpu::Box::cubic(10.0); // isolated molecule (SCF ignores PBC)
    return frame;
}

} // namespace

int main() {
    constexpr double dt = 0.25;      // fs — see the header on κ = ω²Δt²
    constexpr double scf_tol = 1e-6; // Ha, |ΔE| SCF exit
    constexpr int n_steps = 20;
    /// First warm step counted in the means (0-based). The dissipative history
    /// is K+1 = 6 deep, so by step 9 the stationary seed is fully flushed.
    constexpr int warm_from = 8;

    // ── Frozen margins (measured 2026-07-27, this exact configuration; the
    // full landscape is in REFERENCE.md). Iteration counts were bit-identical
    // at OMP_NUM_THREADS = 1 and 8.
    //   warm mean [steps 9..20]:  XL 5.667 | plain-BOMD 6.583 | cold 8.250
    constexpr double min_gain_vs_cold = 1.5;   // measured 2.583 (1.7× headroom)
    constexpr double min_gain_vs_reuse = 0.40; // measured 0.917 (2.3× headroom)
    /// Relative |E−E₀| envelope; measured worst 1.955e-06 (10× headroom). The
    /// budget is dominated by the Velocity-Verlet discretization of the O–H
    /// stretch, not by the SCF: the plain-BOMD twin lands on 1.954e-06.
    constexpr double drift_tol = 2.0e-5;
    /// The two MD legs must sample the SAME geometries for the iteration
    /// comparison to mean anything; measured max|Δx| = 7.6e-09 Å.
    constexpr double traj_match_tol = 1e-6; // Å
    constexpr double oh_max = 1.30;         // Å
    constexpr double oh_min = 0.70;         // Å

    // ── Leg A: the XL-BOMD run ────────────────────────────────────────────
    cpu::scf::RHF xl_base;
    xl_base.max_iter = 200;
    // Niklasson auxiliary-density propagation, Table I row K = 5
    cpu::electronic::XlBomdMethod<cpu::scf::RHF> xl{std::move(xl_base), 5};

    cpu::Frame frame = make_water();
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    // Borrow (add_ref) so the SCF diagnostics stay readable after the run.
    sim.set_evaluator_ref(xl);
    sim.enable_timing(false);
    sim.build();

    std::vector<int> xl_iters;
    std::vector<double> xl_etot;
    std::vector<double> geom; // 9 doubles per step: the visited trajectory
    for (int s = 0; s < n_steps; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: XL step %d did not run\n", s + 1);
            return 1;
        }
        if (!xl.converged()) {
            std::printf("FAIL: XL step %d did not converge (%s) after %d "
                        "iterations — a broken auxiliary guess poisons the "
                        "SCF, it does not merely slow it down\n",
                        s + 1, xl.stop_reason().c_str(), xl.iterations());
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite/negative energy at XL step %d "
                        "PE=%g KE=%g eV\n",
                        s + 1, pe, ke);
            return 1;
        }
        xl_iters.push_back(xl.iterations());
        xl_etot.push_back(pe + ke);
        for (int i = 0; i < 3; ++i) {
            const double x = static_cast<double>(frame.span(kAtom_x)[i]);
            const double y = static_cast<double>(frame.span(kAtom_y)[i]);
            const double z = static_cast<double>(frame.span(kAtom_z)[i]);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                std::printf("FAIL: non-finite coordinate at XL step %d atom "
                            "%d\n",
                            s + 1, i);
                return 1;
            }
            geom.push_back(x);
            geom.push_back(y);
            geom.push_back(z);
        }
    }

    // The witness that the propagator actually ran: one auxiliary advance per
    // converged nested step, no more (an advance on a standalone run or on an
    // unconverged step would integrate a different trajectory than the nuclei).
    if (xl.xl_advances() != n_steps) {
        std::printf("FAIL: %d auxiliary advances over %d nested steps — the "
                    "XL history is not on the nuclear clock\n",
                    xl.xl_advances(), n_steps);
        return 1;
    }

    // ── Leg B: the plain-BOMD twin (last-density reuse) ────────────────────
    cpu::scf::RHF reuse;
    reuse.max_iter = 200;
    // plain BOMD — the ONLY difference from leg A (no XL wrapper)

    cpu::Frame frame_b = make_water();
    cpu::md::MD sim_b;
    sim_b.set_integrator(cpu::md::VelocityVerlet{dt});
    sim_b.set_evaluator_ref(reuse);
    sim_b.enable_timing(false);
    sim_b.build();

    std::vector<int> reuse_iters;
    double traj_mismatch = 0.0;
    for (int s = 0; s < n_steps; ++s) {
        if (sim_b.run(frame_b, 1) != 1) {
            std::printf("FAIL: plain-BOMD step %d did not run\n", s + 1);
            return 1;
        }
        if (!reuse.converged()) {
            std::printf("FAIL: plain-BOMD step %d did not converge (%s)\n",
                        s + 1, reuse.stop_reason().c_str());
            return 1;
        }
        reuse_iters.push_back(reuse.iterations());
        for (int i = 0; i < 3; ++i) {
            const double dx =
                static_cast<double>(frame_b.span(kAtom_x)[i])
                - geom[static_cast<std::size_t>(s * 9 + i * 3 + 0)];
            const double dy =
                static_cast<double>(frame_b.span(kAtom_y)[i])
                - geom[static_cast<std::size_t>(s * 9 + i * 3 + 1)];
            const double dz =
                static_cast<double>(frame_b.span(kAtom_z)[i])
                - geom[static_cast<std::size_t>(s * 9 + i * 3 + 2)];
            traj_mismatch =
                std::max(traj_mismatch,
                         std::max(std::fabs(dx),
                                  std::max(std::fabs(dy), std::fabs(dz))));
        }
    }
    if (reuse.xl_advances() != 0) {
        std::printf("FAIL: the plain-BOMD twin advanced an auxiliary density "
                    "%d times with xl_bomd = false\n",
                    reuse.xl_advances());
        return 1;
    }
    if (!(traj_mismatch <= traj_match_tol)) {
        std::printf("FAIL: the two legs drifted apart by %.3e Å (limit %.1e) — "
                    "their iteration counts are no longer comparable\n",
                    traj_mismatch, traj_match_tol);
        return 1;
    }

    // ── Leg C: the cold-start twin — a fresh method at every geometry ─────
    std::vector<int> cold_iters;
    for (int s = 0; s < n_steps; ++s) {
        cpu::Frame f = make_water();
        for (int i = 0; i < 3; ++i) {
            f.span(kAtom_x)[i] = geom[static_cast<std::size_t>(s * 9 + i * 3)];
            f.span(kAtom_y)[i] =
                geom[static_cast<std::size_t>(s * 9 + i * 3 + 1)];
            f.span(kAtom_z)[i] =
                geom[static_cast<std::size_t>(s * 9 + i * 3 + 2)];
        }
        cpu::scf::RHF cold; // no warm state: constructed, used once, destroyed
        cold.max_iter = 200;
        cold.run(f);
        if (!cold.converged()) {
            std::printf("FAIL: cold twin did not converge at geometry %d\n",
                        s + 1);
            return 1;
        }
        cold_iters.push_back(cold.iterations());
    }
    // Step 1 of the XL leg IS a cold start (nothing has been propagated yet)
    // at exactly the geometry the cold twin replays, so the two counts are the
    // same number by construction — this is what proves the replay is faithful
    // and therefore that the cold baseline is measured, not assumed.
    if (cold_iters.front() != xl_iters.front()) {
        std::printf("FAIL: cold replay of step 1 took %d iterations but the "
                    "XL leg's own cold seed took %d — the replayed geometry is "
                    "not the geometry the XL leg solved at\n",
                    cold_iters.front(), xl_iters.front());
        return 1;
    }

    // ── The pins ──────────────────────────────────────────────────────────
    auto warm_mean = [&](const std::vector<int>& v) {
        double sum = 0.0;
        int n = 0;
        for (std::size_t i = static_cast<std::size_t>(warm_from); i < v.size();
             ++i) {
            sum += v[i];
            ++n;
        }
        return n ? sum / n : 0.0;
    };
    const double mean_xl = warm_mean(xl_iters);
    const double mean_reuse = warm_mean(reuse_iters);
    const double mean_cold = warm_mean(cold_iters);

    const double e0 = xl_etot.front();
    const double e_scale = std::max(1.0, std::fabs(e0));
    double worst_drift = 0.0;
    for (double e : xl_etot)
        worst_drift = std::max(worst_drift, std::fabs(e - e0) / e_scale);

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

    std::printf("XL-BOMD H2O RHF/STO-3G NVE (K=%d, dt=%.3g fs, tol=%.0e Ha): "
                "%d steps  E0=%.8g eV  Ef=%.8g eV\n",
                xl.xl_bomd_order(), dt, scf_tol, n_steps, e0, xl_etot.back());
    std::printf("  SCF iterations  XL:");
    for (int it : xl_iters)
        std::printf(" %d", it);
    std::printf("\n                  BO:");
    for (int it : reuse_iters)
        std::printf(" %d", it);
    std::printf("\n                cold:");
    for (int it : cold_iters)
        std::printf(" %d", it);
    std::printf("\n  warm mean [steps %d..%d]  XL=%.4f  plain-BOMD=%.4f  "
                "cold=%.4f\n",
                warm_from + 1, n_steps, mean_xl, mean_reuse, mean_cold);
    std::printf("  gain vs cold=%.4f (min %.2f)  vs reuse=%.4f (min %.2f)  "
                "xl_advances=%d\n",
                mean_cold - mean_xl, min_gain_vs_cold, mean_reuse - mean_xl,
                min_gain_vs_reuse, xl.xl_advances());
    std::printf("  worst |dE|/scale=%.3e (tol %.1e)  leg max|dx|=%.3e Å  "
                "PEf=%.8g eV  OH=%.4f/%.4f Å\n",
                worst_drift, drift_tol, traj_mismatch, frame.meta.get(kEnergy),
                oh1, oh2);

    // RHF/STO-3G water is ≈ −74.96 Ha ≈ −2039 eV: a bound molecule.
    const double pe_final = frame.meta.get(kEnergy);
    if (!(pe_final < -1900.0) || !(pe_final > -2200.0)) {
        std::printf("FAIL: final PE %g eV outside the bound-H2O window "
                    "(−2200, −1900)\n",
                    pe_final);
        return 1;
    }
    if (!(oh1 > oh_min) || !(oh1 < oh_max) || !(oh2 > oh_min)
        || !(oh2 < oh_max)) {
        std::printf("FAIL: O–H distances %.4f / %.4f Å left the physical "
                    "window (%.2f, %.2f)\n",
                    oh1, oh2, oh_min, oh_max);
        return 1;
    }
    if (!(mean_cold - mean_xl >= min_gain_vs_cold)) {
        std::printf("FAIL: warm XL mean %.4f vs cold-start twin %.4f — gain "
                    "%.4f below the frozen %.2f iterations: the propagated "
                    "auxiliary density is not warm-starting the SCF\n",
                    mean_xl, mean_cold, mean_cold - mean_xl, min_gain_vs_cold);
        return 1;
    }
    if (!(mean_reuse - mean_xl >= min_gain_vs_reuse)) {
        std::printf("FAIL: warm XL mean %.4f vs plain-BOMD reuse %.4f — gain "
                    "%.4f below the frozen %.2f iterations: the auxiliary "
                    "density is no better a guess than the last converged "
                    "one, which is the whole XL-BOMD claim\n",
                    mean_xl, mean_reuse, mean_reuse - mean_xl,
                    min_gain_vs_reuse);
        return 1;
    }
    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: NVE energy drift %.3e exceeds tol %.1e\n",
                    worst_drift, drift_tol);
        return 1;
    }

    std::printf("PASS: xl-bomd-rhf-water (XL-BOMD auxiliary-density guess on "
                "RHF/STO-3G: warm-step iteration reduction vs cold-start and "
                "plain-BOMD twins, NVE drift envelope)\n");
    return 0;
}
