/// @file regressions/mc/npt_lj/reg_npt_lj.cpp
/// @brief END-TO-END REGRESSION: isothermal–isobaric (NPT) Monte Carlo of the
///        truncated-and-shifted Lennard-Jones vapor against the exact
///        finite-N ensemble average ⟨V⟩ = (N+1)k_BT/P plus the second-virial
///        correction N·B₂.
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

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/mc/move_selector.h"
#include "atomiverse/cpu/mc/npt_mc.h"
#include "atomiverse/cpu/mc/translate_move.h"
#include "atomiverse/cpu/mc/volume_move.h"
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
    const double v_pred = (n_atoms + 1) * kT / pressure + n_atoms * b2; // [Å³]
    std::printf("B2 = %.4f sigma^3, <V> predicted = %.1f A^3 (ideal part %.1f, "
                "virial shift %.1f)\n",
                b2 / (sigma * sigma * sigma), v_pred,
                (n_atoms + 1) * kT / pressure, n_atoms * b2);

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
    frame.simbox = ATV::cpu::Box::cubic(box_len);

    // δV ≈ σ_V = ⟨V⟩/√N keeps volume moves near 50% acceptance.
    const double dv_max = v_pred / std::sqrt(static_cast<double>(n_atoms));
    cpu::mc::NptMc mc{temperature, pressure, /*seed=*/1337u};
    mc.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
    mc.set_neighbor(cpu::md::NeighborList<>{cutoff, /*skin=*/1.0, 1, 0, true});
    mc.set_mover(cpu::mc::MoveSelector{}
                     .add(cpu::mc::TranslateMove{0.4 * sigma}, 0.8)
                     .add(cpu::mc::VolumeMove{dv_max}, 0.2));
    mc.build();

    mc.run(frame, 40000); // equilibrate
    const int n_blocks = 30000;
    double v_sum = 0.0, v_sq_sum = 0.0;
    for (int b = 0; b < n_blocks; ++b) {
        mc.run(frame, 10);
        const double v_now = frame.simbox.volume();
        v_sum += v_now;
        v_sq_sum += v_now * v_now;
    }
    const double v_avg = v_sum / n_blocks;
    const double v_var = v_sq_sum / n_blocks - v_avg * v_avg;
    // Volume decorrelates over ~1/(volume-move fraction) sweeps; samples sit
    // 10 sweeps apart, so inflate the naive SE by that correlation factor.
    const double se = std::sqrt(v_var / n_blocks) * std::sqrt(1.0 + 5.0 / 10.0);

    // Tolerance: 5σ statistical + 0.2% for the O(B₃ρ²) truncation.
    const double tol = 5.0 * se + 0.002 * v_pred;
    const double diff = std::abs(v_avg - v_pred);
    const double acc_disp =
        mc.acceptance_ratio(cpu::mc::Transaction::Verb::Displace);
    const double acc_vol = mc.acceptance_ratio(cpu::mc::Transaction::Verb::Box);
    std::printf("<V> got %.1f  want %.1f  (se %.1f, tol %.1f)  acc[disp] %.2f  "
                "acc[vol] %.2f\n",
                v_avg, v_pred, se, tol, acc_disp, acc_vol);
    // A frozen chain (e.g. a NaN energy rejecting every trial) must FAIL,
    // never coast through on its starting volume.
    if (!(acc_disp > 0.01) || !(acc_vol > 0.01)) {
        std::printf(
            "FAIL: frozen chain (displacement/volume acceptance ~ 0)\n");
        return 1;
    }
    if (!(diff <= tol)) {
        std::printf("FAIL: <V> off the NPT ensemble prediction\n");
        return 1;
    }
    std::printf("PASS: NPT <V> matches (N+1)kT/P + N*B2\n");
    return 0;
}
