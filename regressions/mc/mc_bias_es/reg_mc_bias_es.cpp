/// @file regressions/mc/mc_bias_es/reg_mc_bias_es.cpp
/// @brief END-TO-END REGRESSION: Metropolis Monte Carlo on an electronic
///        structure — H₂ scored by a NESTED RHF/STO-3G energy plus a harmonic
///        umbrella on the H–H bond — sampling the bond-length distribution and
///        pinning its variance against the Boltzmann stiff-spring limit
///        Var(r) = k_BT / k_eff, k_eff = k_bias + k_RHF.
///
/// The oracle is statistical mechanics, not a snapshot of our own output: the
/// canonical distribution of a harmonic coordinate has variance k_BT/k, and the
/// two springs the chain feels — the umbrella (an input) and the RHF bond
/// curvature at the umbrella centre (pinned offline, and re-measured here by
/// finite differences as a drift guard) — add. See REFERENCE.md for the full
/// derivation, the r² Jacobian assessment, the offline curvature measurement,
/// and the tolerance freeze.
///
/// This is also the first QM-in-MC case in the tree, and it is an end-to-end
/// witness for four separate pieces of driver contract:
///
///   1. **energy-only trials** (esmm-contract-04). `compute_force` is left at
///      its default `true`; the ONLY thing that suppresses the SCF gradient is
///      MC's `sampling::Phase::energy` slot being stamped
///      `forces_requested = false` and that veto riding down the nesting
///      channel into the method's own stages. The forces on the frame must
///      therefore stay exactly zero for all 125 000 sweeps — if the veto broke,
///      the RHF + bias gradient (~1 eV/Å at the sampled geometries) would land
///      there, and the chain would cost ~6× more CPU (measured; REFERENCE.md);
///   2. **node-aware MC validation** (esmm-contract-02): a nested method-driver
///      counts as MC's energy source, so `mc.build()` must accept a chain whose
///      only QM source is a `DriverNode`;
///   3. **per-sweep zeroing** (esmm-contract-01): 125 000 accumulating trial
///      evaluations onto a clean slate;
///   4. the **ASPC-under-MC guard** staying off-path: no `predictor()` here, so
///      warm starting is plain last-density reuse — legal under sampling, and
///      worth 3 SCF iterations per sweep.
///
/// This is an OUT-OF-LIBRARY consumer: public headers + link atomiverse,
/// nothing else. One flat main(); exits non-zero on failure.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    // 125 000 nested SCFs × ~5 info lines each is 600 000 lines of log. The
    // env var is read by the Backend's auto `log_init`, so no logging type
    // appears in this consumer's source.
    ::setenv("ATV_LOG_LEVEL", "warn", 1);

    // ── The system ───────────────────────────────────────────────────────
    // H₂ in a 20 Å vacuum box, molecule centred so the chain's centre-of-mass
    // random walk (~1 Å over 125 000 moves) never approaches a face.
    const double box_len = 20.0; // [Å]
    const double centre = 0.5 * box_len;
    // Umbrella centre = the RHF/STO-3G equilibrium bond length. NIST CCCBDB
    // (calculated bond lengths, H₂ ¹Σg⁺, HF/STO-3G) gives r_e = 0.712 Å
    // (0.7122 Å at table precision); the shipped machinery puts the stationary
    // point at 0.712230 Å (REFERENCE.md), i.e. 3 × 10⁻⁵ Å from this centre —
    // so the two curvatures add coherently about one common point.
    const double r0 = 0.7122; // [Å]

    // ── The two springs ──────────────────────────────────────────────────
    // Umbrella stiffness — an INPUT, chosen so that (a) the RHF curvature is a
    // 32% share of k_eff (removing it moves Var by +46%, nine tolerances), and
    // (b) the well is stiff enough that the anharmonic + Jacobian corrections
    // to k_BT/k_eff stay under 0.1% (REFERENCE.md "Systematics").
    const double k_bias = 120.0; // [eV/Å²]
    // OFFLINE PIN — RHF/STO-3G ∂²E/∂r² at r0, five-point stencil, h ≤ 0.02 Å,
    // SCF tol 1e-12 Ha; stable to 6 digits over h ∈ {0.005, 0.01, 0.02}.
    // Method, provenance and the literature cross-check (implied ν̃ = 5481
    // cm⁻¹ for a minimal-basis HF H₂ stretch) are in REFERENCE.md.
    // Re-measured below as a drift guard.
    const double k_rhf_pin = 55.684;  // [eV/Å²]
    const double k_rhf_tol = 0.005;   // 0.5% drift guard on the pin
    const double fd_h = 0.01;         // [Å] five-point stencil half-step

    // ── The chain ────────────────────────────────────────────────────────
    const double temperature = 300.0;   // [K]
    const unsigned seed = 20260727u;
    const double delta = 0.04;          // [Å] TranslateMove half-width
    const int n_burn = 5000;            // sweeps discarded
    const int n_sample = 120000;        // sweeps sampled (one r per sweep)
    // Frozen tolerance: 5% relative. Statistical 1σ on Var is 0.8% (blocking)
    // and the seed-to-seed spread over five chains was ±0.6%; systematics are
    // ≤ 0.1%. So 5% is ≈ 6σ, while the "no QM at all" failure mode is +46%.
    const double var_rel_tol = 0.05;

    const double kT = units::kB_ev_per_k * temperature; // [eV]

    // ── Drift guard: re-measure the RHF curvature at r0 ───────────────────
    // Five single points (energy only — this stencil wants no gradient). If
    // the Gaussian SCF surface ever moves, the analytic prediction below is
    // stale and this case must be re-derived, not re-toleranced.
    auto rhf_energy_at = [&](double r) {
        cpu::Frame f(2);
        for (int i = 0; i < 2; ++i) {
            f.span(kAtom_type)[i] = 1; // Z = 1
            f.span(kAtom_mass)[i] = 1.008;
            f.span(kAtom_tag)[i] = i + 1;
            f.span(kAtom_y)[i] = centre;
            f.span(kAtom_z)[i] = centre;
        }
        f.span(kAtom_x)[0] = centre - 0.5 * r;
        f.span(kAtom_x)[1] = centre + 0.5 * r;
        f.box = cpu::Box::cubic(box_len);
        cpu::scf::RHF probe;
        probe.basis = cpu::scf::BasisSet::STO3G;
        probe.tol = 1e-12; // [Hartree]
        probe.max_iter = 100;
        probe.compute_force = false; // this stencil wants energies only
        // `cpu::Evaluate` rather than the `cpu::single_point` sugar for one
        // reason: `set_threads(1)` — see the MC driver below.
        cpu::Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate).add_ref(probe);
        eval.set_threads(1);
        eval.build();
        eval.run(f);
        if (!probe.converged()) {
            std::printf("FAIL: reference SCF did not converge at r = %.4f A\n",
                        r);
            std::exit(1);
        }
        return static_cast<double>(f.meta.get(kEnergy)); // [eV]
    };
    const double k_rhf_fd =
        (-rhf_energy_at(r0 + 2 * fd_h) + 16.0 * rhf_energy_at(r0 + fd_h)
         - 30.0 * rhf_energy_at(r0) + 16.0 * rhf_energy_at(r0 - fd_h)
         - rhf_energy_at(r0 - 2 * fd_h))
        / (12.0 * fd_h * fd_h); // [eV/Å²]

    // ── The analytic pin ─────────────────────────────────────────────────
    const double k_eff = k_bias + k_rhf_pin;   // [eV/Å²]
    const double var_pred = kT / k_eff;        // [Å²]
    const double var_no_qm = kT / k_bias;      // the "QM went missing" value

    // ── Build the chain: nested RHF energy + umbrella, one mover ─────────
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1; // Z = 1 (hydrogen)
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1; // DistanceCV resolves these tags
        frame.span(kAtom_y)[i] = centre;
        frame.span(kAtom_z)[i] = centre;
    }
    frame.span(kAtom_x)[0] = centre - 0.5 * r0;
    frame.span(kAtom_x)[1] = centre + 0.5 * r0;
    frame.box = cpu::Box::cubic(box_len);

    // compute_force stays at its DEFAULT true — see the header comment: the
    // energy-only slot policy is the only thing that may suppress the gradient.
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-8; // [Hartree]; 3 × 10⁻⁷ eV, 10⁻⁵ of k_BT
    rhf.max_iter = 100;

    cpu::mc::MC mc{temperature, seed};
    mc.set_mover(cpu::mc::TranslateMove{delta});
    // Order 0: the nested method writes the total energy (it assigns);
    // order 10: the umbrella accumulates onto it.
    mc.on(sampling::Phase::energy).add_ref(rhf, /*order=*/0, /*budget=*/100);
    mc.on(sampling::Phase::energy)
        .add(cpu::bias::HarmonicBias{
                 std::make_unique<cpu::bias::DistanceCV>(1, 2), k_bias, r0},
             /*order=*/10);
    // Two atoms and two basis functions: at the node's default thread count
    // the OpenMP fork/join per sweep costs orders of magnitude more than the
    // physics (measured on this 48-core node: 6.8 s of CPU at 1 thread, 231 s
    // at 8, and no completion in 11 minutes at 48). The regressions target
    // runs the binary with no OMP_NUM_THREADS, so the case pins its own.
    mc.set_threads(1);
    mc.build();

    mc.run(frame, n_burn);

    std::vector<double> r_samples;
    r_samples.reserve(static_cast<std::size_t>(n_sample));
    long long scf_iter_sum = 0;
    int scf_iter_min = 1 << 30;
    int scf_iter_max = 0;
    double f_abs_max = 0.0; // must stay 0: the energy-only witness
    for (int s = 0; s < n_sample; ++s) {
        mc.run(frame, 1);
        double dx = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
        double dy = frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0];
        double dz = frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0];
        frame.box.min_image(dx, dy, dz);
        r_samples.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
        scf_iter_sum += rhf.iterations();
        scf_iter_min = std::min(scf_iter_min, rhf.iterations());
        scf_iter_max = std::max(scf_iter_max, rhf.iterations());
        for (int i = 0; i < 2; ++i) {
            f_abs_max = std::max(f_abs_max, std::fabs(frame.span(kAtom_fx)[i]));
            f_abs_max = std::max(f_abs_max, std::fabs(frame.span(kAtom_fy)[i]));
            f_abs_max = std::max(f_abs_max, std::fabs(frame.span(kAtom_fz)[i]));
        }
    }

    // ── Statistics ───────────────────────────────────────────────────────
    double sum_r = 0.0;
    for (double v : r_samples)
        sum_r += v;
    const double mean_r = sum_r / n_sample;
    double sum_sq = 0.0;
    for (double v : r_samples) {
        const double d = v - mean_r;
        sum_sq += d * d;
    }
    const double var_r = sum_sq / n_sample;

    // Blocking estimate of the correlated standard error on Var (the chain's
    // autocorrelation time is ~2 sweeps; blocks of 500 are 250× that).
    const int block = 500;
    const int n_block = n_sample / block;
    double b_sum = 0.0, b_sum_sq = 0.0;
    for (int b = 0; b < n_block; ++b) {
        double a = 0.0;
        for (int i = 0; i < block; ++i) {
            const double d = r_samples[static_cast<std::size_t>(b * block + i)]
                             - mean_r;
            a += d * d;
        }
        a /= block;
        b_sum += a;
        b_sum_sq += a * a;
    }
    const double b_mean = b_sum / n_block;
    const double b_var = std::max(b_sum_sq / n_block - b_mean * b_mean, 0.0);
    const double se_var = std::sqrt(b_var / n_block);

    const double acc = mc.acceptance_ratio();
    const double energy = static_cast<double>(frame.meta.get(kEnergy)); // [eV]
    const double var_tol = var_rel_tol * var_pred;
    const double var_diff = std::fabs(var_r - var_pred);

    // ── Report ───────────────────────────────────────────────────────────
    std::printf("MC(300 K) on H2: nested RHF/STO-3G energy + harmonic umbrella "
                "on the H-H bond\n");
    std::printf("  chain      : seed=%u  delta=%.3f A  burn=%d  sampled=%d "
                "sweeps\n",
                seed, delta, n_burn, n_sample);
    std::printf("  springs    : k_bias=%.3f  k_RHF(pin)=%.4f  "
                "k_RHF(FD here)=%.4f  k_eff=%.4f eV/A^2\n",
                k_bias, k_rhf_pin, k_rhf_fd, k_eff);
    std::printf("  Var(r)     : measured %.6e A^2   analytic kT/k_eff %.6e "
                "A^2   (tol %.1f%% = %.3e)\n",
                var_r, var_pred, 100.0 * var_rel_tol, var_tol);
    std::printf("               deviation %+.3f%%   blocking 1-sigma %.3f%%   "
                "no-QM value would be %+.1f%%\n",
                100.0 * (var_r - var_pred) / var_pred,
                100.0 * se_var / var_pred,
                100.0 * (var_no_qm - var_pred) / var_pred);
    std::printf("  <r>        : %.6f A   (umbrella centre %.4f A, expected "
                "shift +5.6e-4 A)\n",
                mean_r, r0);
    std::printf("  acceptance : %.3f     E_last = %.6f eV\n", acc, energy);
    std::printf("  SCF/sweep  : mean %.3f  min %d  max %d   (warm start, no "
                "predictor)\n",
                static_cast<double>(scf_iter_sum) / n_sample, scf_iter_min,
                scf_iter_max);
    std::printf("  |f|_max over the whole chain: %.3e eV/A   (energy-only "
                "trials: must be 0)\n",
                f_abs_max);

    bool ok = true;
    if (!(std::fabs(k_rhf_fd - k_rhf_pin) <= k_rhf_tol * k_rhf_pin)) {
        std::printf("FAIL: the RHF/STO-3G bond curvature at r0 moved: FD "
                    "%.6f vs pinned %.6f eV/A^2 (tol %.1f%%). The analytic "
                    "prediction is derived from the pin — re-derive it "
                    "offline, do not re-tolerance this case.\n",
                    k_rhf_fd, k_rhf_pin, 100.0 * k_rhf_tol);
        ok = false;
    }
    if (!(f_abs_max <= 1e-8)) {
        std::printf("FAIL: a force of %.3e eV/A survived the trial energy "
                    "phase — MC scores energies, so sampling::Phase::energy "
                    "must run energy-only (esmm-contract-04). Either the slot "
                    "flag is not stamped or it no longer reaches the nested "
                    "method.\n",
                    f_abs_max);
        ok = false;
    }
    if (!(scf_iter_min >= 1)) {
        std::printf("FAIL: a sweep ran zero SCF iterations — the nested method "
                    "did not actually score that trial.\n");
        ok = false;
    }
    if (!(static_cast<double>(scf_iter_sum) / n_sample <= 4.0)) {
        std::printf("FAIL: %.2f SCF iterations per sweep (frozen bound 4.0, "
                    "measured 3.00 with last-density warm start) — the warm "
                    "start regressed and the case is no longer cheap.\n",
                    static_cast<double>(scf_iter_sum) / n_sample);
        ok = false;
    }
    if (!(acc > 0.35 && acc < 0.65)) {
        std::printf("FAIL: acceptance %.3f outside the sane band [0.35, 0.65] "
                    "(measured 0.484-0.491 over five seeds). A frozen or a "
                    "free chain does not sample this distribution.\n",
                    acc);
        ok = false;
    }
    if (!std::isfinite(energy) || !(energy > -31.0 && energy < -29.5)) {
        std::printf("FAIL: last-accepted energy %.6f eV is not a bound H2 "
                    "(RHF/STO-3G minimum is -30.408884 eV, the umbrella adds "
                    "at most ~0.2 eV over the sampled window).\n",
                    energy);
        ok = false;
    }
    if (!(std::fabs(mean_r - r0) <= 2.0e-3)) {
        std::printf("FAIL: <r> = %.6f A is %.1e A off the umbrella centre "
                    "%.4f A; the Jacobian + anharmonic shift is only "
                    "+5.6e-4 A, so the chain is not in the right well.\n",
                    mean_r, std::fabs(mean_r - r0), r0);
        ok = false;
    }
    if (!(var_diff <= var_tol)) {
        std::printf("FAIL: Var(r) off the Boltzmann stiff-spring limit: got "
                    "%.6e A^2, want %.6e A^2, |diff| %.3e > tol %.3e A^2\n",
                    var_r, var_pred, var_diff, var_tol);
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS: Var(r) matches k_BT/(k_bias + k_RHF) within %.0f%%, "
                "sampled by 125000 energy-only nested-RHF Metropolis trials\n",
                100.0 * var_rel_tol);
    return 0;
}
