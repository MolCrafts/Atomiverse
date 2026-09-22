/// @file regressions/mc/npt_lj/reg_npt_lj.cpp
/// @brief END-TO-END REGRESSION: isothermal–isobaric (NPT) Monte Carlo of the
///        truncated-and-shifted Lennard-Jones vapor against the exact
///        finite-N ensemble average ⟨V⟩ = (N+1)k_BT/P plus the second-virial
///        correction N·B₂ — and against the exact ensemble WIDTH
///        σ_V = √(N+1)·k_BT/P.
///
/// See REFERENCE.md for the literature source of every relation below and the
/// physical justification of the tolerance. The (N+1) — not N — is the exact
/// consequence of the ∫V^N e^{−βPV} dV weight that uniform-in-V volume
/// sampling targets: any error in the P·ΔV term, the N k_BT ln(V'/V) term, or
/// the box/coordinate rollback shifts ⟨V⟩ off this value.
///
/// This is an OUT-OF-LIBRARY consumer: public headers + link atomiverse,
/// nothing else. One flat main(); exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/npt_mc.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/montecarlo/volume_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Argon-flavoured LJ, truncated AND shifted at 2.5σ ────────────────
    const double eps = 120.0 * units::kB_ev_per_k;      // ε/k_B = 120 K [eV]
    const double sigma = 3.4;                           // [Å]
    const double cutoff = 2.5 * sigma;                  // [Å]
    const double t_red = 1.5;                           // supercritical vapor
    const double temperature = t_red * 120.0;           // 180 K
    const double kT = units::kB_ev_per_k * temperature; // [eV]
    const double mass = 39.948;                         // argon [amu]
    const int n_atoms = 125;                            // 5³ — clean SC lattice

    // Dilute state point: P* = Pσ³/ε = 0.0297 targets ρσ³ ≈ 0.02, where the
    // virial series is converged at second order (see REFERENCE.md).
    const double p_red = 0.0297;
    const double pressure = p_red * eps / (sigma * sigma * sigma); // [eV/Å³]

    // ── B₂ of the cut-shifted potential by quadrature (exact reference) ──
    // B₂(T) = −2π ∫₀^rc [exp(−βu_s(r)) − 1] r² dr (McQuarrie eq. 12-30).
    const double beta = 1.0 / kT;
    const double u_at_rc =
        4.0 * eps
        * (std::pow(sigma / cutoff, 12) - std::pow(sigma / cutoff, 6));
    double b2 = 0.0; // [Å³]
    {
        const int n_quad = 200000;
        const double h = cutoff / n_quad;
        double acc = 0.0;
        for (int k = 0; k <= n_quad; ++k) {
            const double r = k * h;
            double f;
            if (r < 1e-12) {
                f = -1.0;
            } else {
                const double sr6 = std::pow(sigma / r, 6);
                const double u = 4.0 * eps * (sr6 * sr6 - sr6) - u_at_rc;
                f = std::exp(-beta * u) - 1.0;
            }
            const double w =
                (k == 0 || k == n_quad) ? 1.0 : (k % 2 ? 4.0 : 2.0);
            acc += w * f * r * r;
        }
        b2 = -2.0 * units::pi * acc * h / 3.0;
    }

    // Analytic prediction: ⟨V⟩ = (N+1)k_BT/P + N·B₂ + O(N B₂² ρ).
    const double v_ideal = (n_atoms + 1) * kT / pressure; // [Å³]
    const double v_pred = v_ideal + n_atoms * b2;         // [Å³]
    // Exact ideal-gas NPT width: Var V = (N+1)(k_BT/P)².
    const double sd_pred = std::sqrt(double(n_atoms + 1)) * kT / pressure;
    // The fingerprint of the V^N weight: getting the exponent wrong by one
    // (N instead of N+1) moves ⟨V⟩ by exactly this much.
    const double n_plus_one_signature = kT / pressure; // [Å³]
    std::printf("B2 = %.4f sigma^3, <V> predicted = %.1f A^3 (ideal part %.1f, "
                "virial shift %.1f)\n",
                b2 / (sigma * sigma * sigma), v_pred, v_ideal, n_atoms * b2);

    // ── Start 12% BELOW the predicted volume, on a simple-cubic lattice ──
    // Off-prediction start: the chain must expand to the answer, so a frozen
    // chain cannot fake a pass by never leaving its initial box. Lattice
    // start: random placement can overlap two atoms, and an infinite initial
    // energy freezes the Markov chain with NaN ΔE forever.
    const double box_len = std::cbrt(0.88 * v_pred);
    const int side = 5; // 5³ = 125 = n_atoms
    const double spacing = box_len / side;
    cpu::Frame frame(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        frame.span(ATV::kAtom_x)[i] = (i % side + 0.5) * spacing;
        frame.span(ATV::kAtom_y)[i] = ((i / side) % side + 0.5) * spacing;
        frame.span(ATV::kAtom_z)[i] = (i / (side * side) + 0.5) * spacing;
        frame.span(ATV::kAtom_mass)[i] = mass;
        frame.span(ATV::kAtom_type)[i] = 1;
    }
    frame.box = ATV::cpu::Box::cubic(box_len);

    // δV ≈ σ_V/√N keeps volume moves near 50% acceptance. Volume is the
    // observable, so it gets half the trials: the volume autocorrelation time
    // is what sets the error bar, and translations do not shorten it.
    const double dv_max = v_pred / std::sqrt(static_cast<double>(n_atoms));
    cpu::mc::NptMc mc{temperature, pressure, /*seed=*/1337u};
    mc.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    mc.set_neighbor(cpu::md::NeighborList<>{cutoff, /*skin=*/1.0, 1, 0, true});
    mc.set_mover(cpu::mc::MoveSelector{}
                     .add(cpu::mc::TranslateMove{0.4 * sigma}, 0.5)
                     .add(cpu::mc::VolumeMove{dv_max}, 0.5));
    mc.build();

    mc.run(frame, 20000); // equilibrate

    // ── Blocking (batch means) — Flyvbjerg & Petersen 1989 ───────────────
    // Consecutive MC samples are correlated, so the naive √(Var/n) understates
    // the error by the square root of the autocorrelation time. Batch means
    // over blocks far longer than that time ARE independent, so their spread
    // is the honest standard error. It is also the numerically stable route:
    // ⟨V²⟩ − ⟨V⟩² at V ≈ 2.4e5 subtracts two ~5.9e10 numbers and returns a
    // NEGATIVE variance in double precision.
    const int n_batches = 40;
    const int samples_per_batch = 1000;
    const int trials_per_sample = 10;
    std::vector<double> batch_mean(n_batches, 0.0);
    // Shifted-data accumulation about the prediction: the deviations are
    // O(σ_V) rather than O(⟨V⟩), so nothing large is ever subtracted.
    const double shift = v_pred;
    double s1 = 0.0, s2 = 0.0;
    long n_samples = 0;
    for (int b = 0; b < n_batches; ++b) {
        double acc = 0.0;
        for (int s = 0; s < samples_per_batch; ++s) {
            mc.run(frame, trials_per_sample);
            const double d = frame.box.volume() - shift;
            acc += d;
            s1 += d;
            s2 += d * d;
            ++n_samples;
        }
        batch_mean[b] = shift + acc / samples_per_batch;
    }
    const double v_avg = shift + s1 / double(n_samples);
    double batch_var = 0.0;
    for (double m : batch_mean)
        batch_var += (m - v_avg) * (m - v_avg);
    batch_var /= double(n_batches - 1);
    const double se = std::sqrt(batch_var / double(n_batches));

    // Ensemble WIDTH of the samples themselves (not of the batch means).
    const double d_mean = s1 / double(n_samples);
    const double sd_measured =
        std::sqrt(s2 / double(n_samples) - d_mean * d_mean);

    const double acc_disp =
        mc.acceptance_ratio(cpu::mc::Transaction::Verb::Displace);
    const double acc_vol = mc.acceptance_ratio(cpu::mc::Transaction::Verb::Box);

    // Tolerance: 3× the BLOCKED standard error + 0.2% of ⟨V⟩ for the
    // O(B₃ρ²) truncation.
    const double tol = 3.0 * se + 0.002 * v_pred;
    const double diff = std::abs(v_avg - v_pred);
    // Model discrimination, with no tolerance at all: the sampled mean must
    // sit closer to the (N+1) prediction than to the N one.
    const double miss_n_plus_1 = std::abs(v_avg - v_pred);
    const double miss_n = std::abs(v_avg - (v_pred - n_plus_one_signature));
    std::printf("<V> got %.1f  want %.1f  (diff %.1f = %.2f SE, blocked SE "
                "%.1f, tol %.1f)\n",
                v_avg, v_pred, v_avg - v_pred, (v_avg - v_pred) / se, se, tol);
    std::printf("sigma_V got %.1f  want %.1f (exact ideal-gas NPT width)  "
                "ratio %.4f\n",
                sd_measured, sd_pred, sd_measured / sd_pred);
    std::printf("acc[disp] %.3f  acc[vol] %.3f\n", acc_disp, acc_vol);
    std::printf("V^N weight: |<V> - (N+1)kT/P - N B2| = %.1f  vs  "
                "|<V> - N kT/P - N B2| = %.1f  (signature %.1f)\n",
                miss_n_plus_1, miss_n, n_plus_one_signature);

    // A frozen chain (e.g. a NaN energy, or a rollback that corrupted the
    // configuration into an overlap no trial can escape) must FAIL, never
    // coast through on its starting volume.
    if (!(acc_disp > 0.01) || !(acc_vol > 0.01)) {
        std::printf(
            "FAIL: frozen chain (displacement/volume acceptance ~ 0)\n");
        return 1;
    }
    // A frozen chain has sigma_V = 0 exactly; a broken volume Jacobian
    // reshapes the distribution. The width is an ensemble statement in its
    // own right, independent of the mean.
    if (!(std::fabs(sd_measured / sd_pred - 1.0) <= 0.15)) {
        std::printf("FAIL: sigma_V is %.1f%% off the exact NPT ensemble "
                    "width — the sampled distribution is not P(V) ∝ "
                    "V^N e^{-beta P V}\n",
                    100.0 * (sd_measured / sd_pred - 1.0));
        return 1;
    }
    if (!(diff <= tol)) {
        std::printf("FAIL: <V> off the NPT ensemble prediction\n");
        return 1;
    }
    // The (N+1) is the fingerprint of the V^N weight that uniform-in-V
    // sampling targets. Comparing the two candidate predictions needs no
    // tolerance: one of them is right and the other is 1/(N+1) away.
    if (!(miss_n_plus_1 < miss_n)) {
        std::printf("FAIL: <V> sits closer to N*kT/P than to (N+1)*kT/P — the "
                    "volume weight is not V^N (check the N k_BT ln(V'/V) "
                    "Jacobian term)\n");
        return 1;
    }

    // ── The sampler's reference energy must still describe the frame ─────
    // `NptMetropolis` carries the last accepted energy forward instead of
    // recomputing it; if anything mutates the configuration behind the
    // sampler's back (a rollback splicing a saved row over the wrong atom,
    // say) the two silently part company and every later trial is judged
    // against a state that no longer exists.
    cpu::Frame check = frame;
    cpu::Evaluate eval;
    eval.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, /*skin=*/1.0});
    eval.build();
    eval.run(check);
    const double e_true = check.meta.get(ATV::kEnergy);
    const double e_book = static_cast<double>(mc.current_energy());
    std::printf("bookkeeping E %.9f eV   recomputed E %.9f eV   |dE| %.3e\n",
                e_book, e_true, std::fabs(e_book - e_true));
    if (!(std::fabs(e_book - e_true) <= 1e-9 * std::fabs(e_true) + 1e-12)) {
        std::printf("FAIL: the chain's reference energy no longer describes "
                    "the configuration it holds\n");
        return 1;
    }

    std::printf("PASS: NPT <V> matches (N+1)kT/P + N*B2\n");
    return 0;
}
