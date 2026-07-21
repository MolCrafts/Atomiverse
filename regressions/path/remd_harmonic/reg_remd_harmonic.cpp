/// @file regressions/path/remd_harmonic/reg_remd_harmonic.cpp
/// @brief REMD on an LJ dimer with four temperatures — exchange stats.
///
/// Literature / textbook check (Sugita–Okamoto REMD Metropolis):
///   P_acc = min(1, exp[(β_i − β_j)(U_i − U_j)])
/// with adjacent temperature swaps and velocity rescaling √(T_i/T_j).
///
/// Full analytic temperature-marginal histograms for a free harmonic
/// oscillator require a 1-particle external spring evaluator and longer
/// sampling; this regression pins the public Remd API, every-edge attempts,
/// nonzero accept ratio, collapse to nlocal, and deterministic seed.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/path/remd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

static void build_lj(cpu::md::MD& md, double, void*) {
    md.add_potential(cpu::PairLJCut{0.0104, 3.4, 8.5});
    md.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
}

int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 4.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 39.948;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.simbox = cpu::Box::cubic(30.0);

    const std::vector<double> T = {100.0, 150.0, 200.0, 250.0};
    cpu::path::Remd remd(
        {T, /*dt=*/1.0, /*steps_per_exchange=*/20, /*seed=*/7});
    remd.set_program_builder(&build_lj, nullptr);
    constexpr int n_ex = 40;
    remd.run(f, n_ex);

    // Even/odd alternating edges: per round ceil((N-1)/2) or floor attempts.
    // N=4 → 2 attempts on even rounds, 1 on odd → average 1.5 * n_ex.
    const int attempts = remd.exchange_attempts();
    const int accepts = remd.exchange_accepts();
    std::printf("REMD attempts=%d accepts=%d ratio=%.3f nlocal=%d\n", attempts,
                accepts, attempts ? double(accepts) / attempts : 0.0,
                f.nlocal());

    if (attempts < n_ex) {
        std::printf("FAIL: too few exchange attempts (%d)\n", attempts);
        return 1;
    }
    // Every temperature edge is exercised over even/odd schedule.
    // With n_ex=40, even rounds contribute 20*2, odd 20*1 = 60 attempts.
    if (attempts < 50) {
        std::printf("FAIL: expected ≥50 attempts for 4-replica ladder\n");
        return 1;
    }
    if (accepts < 1) {
        std::printf("FAIL: zero accepts — ladder or Metropolis broken\n");
        return 1;
    }
    const double ratio = double(accepts) / double(attempts);
    if (!(ratio > 0.01 && ratio < 0.99)) {
        std::printf("FAIL: accept ratio out of (0.01,0.99): %.3f\n", ratio);
        return 1;
    }
    if (f.nlocal() != 2) {
        std::printf("FAIL: collapse nlocal=%d\n", f.nlocal());
        return 1;
    }
    // Collapse writes the lowest-T slot; positions must remain finite.
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(f.span(kAtom_x)[i])
            || !std::isfinite(f.span(kAtom_y)[i])
            || !std::isfinite(f.span(kAtom_z)[i])) {
            std::printf("FAIL: non-finite coordinates after collapse\n");
            return 1;
        }
    }

    // Determinism: same seed → same accept count.
    cpu::Frame f2(2);
    f2.span(kAtom_x)[0] = 0;
    f2.span(kAtom_y)[0] = 0;
    f2.span(kAtom_z)[0] = 0;
    f2.span(kAtom_x)[1] = 4.0;
    f2.span(kAtom_y)[1] = 0;
    f2.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f2.span(kAtom_mass)[i] = 39.948;
        f2.span(kAtom_type)[i] = 0;
        f2.span(kAtom_tag)[i] = i + 1;
    }
    f2.simbox = cpu::Box::cubic(30.0);
    cpu::path::Remd remd2(
        {T, /*dt=*/1.0, /*steps_per_exchange=*/20, /*seed=*/7});
    remd2.set_program_builder(&build_lj, nullptr);
    remd2.run(f2, n_ex);
    if (remd2.exchange_accepts() != accepts
        || remd2.exchange_attempts() != attempts) {
        std::printf("FAIL: seed-7 not deterministic (%d/%d vs %d/%d)\n",
                    remd2.exchange_accepts(), remd2.exchange_attempts(),
                    accepts, attempts);
        return 1;
    }

    std::printf("PASS: REMD Metropolis ladder + exchange stats\n");
    return 0;
}
