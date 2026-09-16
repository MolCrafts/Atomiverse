/// @file regressions/mc/gcmc_lj/reg_gcmc_lj.cpp
/// @brief END-TO-END REGRESSION: grand-canonical (μVT) Monte Carlo of the
///        truncated-and-shifted Lennard-Jones vapor against the exact Mayer
///        activity expansion — ⟨N⟩/V = z − 2B₂z² + O(z³) — at TWO activities.
///
/// See REFERENCE.md for the literature source of every relation below and the
/// physical justification of the tolerance. Matching the analytic μVT
/// activity response at two state points is precisely the "detailed balance
/// against the analytic μVT relation" check the spec demands: any error in
/// the insertion/deletion prefactor V/(Λ³(N+1)), in Λ, in the μ convention,
/// or in the accept/rollback bookkeeping shifts ⟨N⟩ off this curve.
///
/// This is an OUT-OF-LIBRARY consumer: public headers + link atomiverse,
/// nothing else. One flat main(); exits non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/gcmc.h"
#include "atomiverse/cpu/montecarlo/insert_delete_move.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // ── Argon-flavoured LJ, truncated AND shifted at 2.5σ ────────────────
    // The engine's PairLJCut is cut-and-shifted, so B₂ below is computed for
    // exactly that potential — convention-exact, no long-range correction.
    const double eps = 120.0 * units::kB_ev_per_k;      // ε/k_B = 120 K [eV]
    const double sigma = 3.4;                           // [Å]
    const double cutoff = 2.5 * sigma;                  // [Å]
    const double t_red = 1.5;                           // supercritical vapor
    const double temperature = t_red * 120.0;           // 180 K
    const double kT = units::kB_ev_per_k * temperature; // [eV]
    const double mass = 39.948;                         // argon [amu]
    const double box_len = 12.0 * sigma;                // 40.8 Å; > 2 rc
    const double volume = box_len * box_len * box_len;  // [Å³]

    // ── B₂ of the cut-shifted potential by quadrature (exact reference) ──
    // B₂(T) = −2π ∫₀^rc [exp(−βu_s(r)) − 1] r² dr;  u_s ≡ 0 beyond rc.
    // (McQuarrie, Statistical Mechanics, eq. 12-30; see REFERENCE.md.)
    const double beta = 1.0 / kT;
    const double u_at_rc =
        4.0 * eps
        * (std::pow(sigma / cutoff, 12) - std::pow(sigma / cutoff, 6));
    double b2 = 0.0; // [Å³]
    {
        const int n_quad = 200000; // Simpson; integrand is smooth and bounded
        const double h = cutoff / n_quad;
        double acc = 0.0;
        for (int k = 0; k <= n_quad; ++k) {
            const double r = k * h;
            double f; // exp(−βu_s) − 1
            if (r < 1e-12) {
                f = -1.0; // core: u → ∞
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
    std::printf("B2(T* = %.2f, cut-shifted 2.5 sigma) = %.4f sigma^3\n", t_red,
                b2 / (sigma * sigma * sigma));

    // ── Two activities on the dilute vapor branch ────────────────────────
    const double z_red[2] = {0.010, 0.030};                         // z σ³
    const double lambda = units::lambda_thermal(mass, temperature); // [Å]
    const double lambda3 = lambda * lambda * lambda;                // [Å³]

    int failures = 0;
    for (int p = 0; p < 2; ++p) {
        const double z = z_red[p] / (sigma * sigma * sigma); // [Å⁻³]
        const double mu = kT * std::log(z * lambda3);        // [eV], total μ

        // Analytic prediction: ρ = z − 2 B₂ z² + O(z³).
        const double rho_pred = z - 2.0 * b2 * z * z; // [Å⁻³]
        const double n_pred = rho_pred * volume;

        // Start BELOW the predicted density (the chain must climb to the
        // answer, so a frozen chain cannot fake a pass), on a simple-cubic
        // lattice — random placement can overlap two atoms, and an infinite
        // initial energy freezes the Markov chain with NaN ΔE forever.
        const int n0 = static_cast<int>(0.8 * n_pred);
        const int side = static_cast<int>(std::ceil(std::cbrt(double(n0))));
        const double spacing = box_len / side;
        cpu::Frame frame(n0);
        for (int i = 0; i < n0; ++i) {
            frame.span(ATV::kAtom_x)[i] = (i % side + 0.5) * spacing;
            frame.span(ATV::kAtom_y)[i] = ((i / side) % side + 0.5) * spacing;
            frame.span(ATV::kAtom_z)[i] = (i / (side * side) + 0.5) * spacing;
            frame.span(ATV::kAtom_mass)[i] = mass;
            frame.span(ATV::kAtom_type)[i] = 1;
        }
        frame.box = ATV::cpu::Box::cubic(box_len);

        cpu::mc::Gcmc mc{temperature, mu, mass, /*seed=*/97u + p};
        mc.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
        mc.set_neighbor(
            cpu::md::NeighborList<>{cutoff, /*skin=*/1.0, 1, 0, true});
        mc.set_mover(cpu::mc::MoveSelector{}
                         .add(cpu::mc::TranslateMove{0.4 * sigma}, 0.5)
                         .add(cpu::mc::InsertDeleteMove{1, mass}, 0.5));
        mc.build();

        mc.run(frame, 50000); // equilibrate
        const int n_blocks = 40000;
        double n_sum = 0.0, n_sq_sum = 0.0;
        for (int b = 0; b < n_blocks; ++b) {
            mc.run(frame, 10); // decorrelate between samples
            const double n_now = static_cast<double>(frame.nlocal());
            n_sum += n_now;
            n_sq_sum += n_now * n_now;
        }
        const double n_avg = n_sum / n_blocks;
        const double n_var = n_sq_sum / n_blocks - n_avg * n_avg;
        // ⟨N⟩ fluctuations are ~Poissonian; block samples 10 sweeps apart are
        // correlated over ~⟨N⟩ sweeps, so inflate the naive SE accordingly.
        const double se =
            std::sqrt(n_var / n_blocks) * std::sqrt(1.0 + n_avg / 10.0);

        // Tolerance: 5σ statistical + the magnitude of the first NEGLECTED
        // term of the activity expansion, 3·b₃·z³·V, bounded via
        // |b₃| = |2b₂² − B₃/2| ≤ 2.5·b₂² (B₃ > 0 at T* = 1.5 and partially
        // cancels; see REFERENCE.md).
        const double tol = 5.0 * se + 3.0 * 2.5 * b2 * b2 * z * z * z * volume;
        const double diff = std::abs(n_avg - n_pred);
        const double acc_ins =
            mc.acceptance_ratio(cpu::mc::Transaction::Verb::Insert);
        const double acc_del =
            mc.acceptance_ratio(cpu::mc::Transaction::Verb::Delete);
        std::printf("z* = %.3f: <N> got %.3f  want %.3f  (se %.3f, tol %.3f)  "
                    "acc[ins] %.2f  acc[del] %.2f\n",
                    z_red[p], n_avg, n_pred, se, tol, acc_ins, acc_del);
        // A frozen chain (e.g. a NaN energy rejecting every trial) must FAIL,
        // never coast through on its starting configuration.
        if (!(acc_ins > 0.01) || !(acc_del > 0.01)) {
            std::printf("FAIL: frozen chain at z* = %.3f (insert/delete "
                        "acceptance ~ 0)\n",
                        z_red[p]);
            ++failures;
        } else if (!(diff <= tol)) {
            std::printf("FAIL: <N> off the muVT activity expansion at z* = "
                        "%.3f\n",
                        z_red[p]);
            ++failures;
        }
    }

    if (failures == 0)
        std::printf("PASS: muVT <N>(z) matches the Mayer activity expansion "
                    "at both activities\n");
    return failures == 0 ? 0 : 1;
}
