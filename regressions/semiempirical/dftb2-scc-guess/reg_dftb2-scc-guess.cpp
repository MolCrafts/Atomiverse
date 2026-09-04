/// @file regressions/semiempirical/dftb2-scc-guess/reg_dftb2-scc-guess.cpp
/// @brief ASPC extrapolation of the SCC charges across launches.
///
/// Domain: Kolafa's always-stable predictor–corrector (Kolafa, J. Comput.
/// Chem. 25, 335 (2004)) applied to the Mulliken Δq of SCC-DFTB2 (Elstner et
/// al., Phys. Rev. B 58, 7260 (1998)).
///
/// Two claims, and the second is the one that matters:
///   1. Along a smooth trajectory the extrapolated guess cuts SCC iterations.
///   2. It does **not** move where the SCC converges — the guess changes the
///      starting point, the residual gate is untouched, so the energy is the
///      same to well inside `scc_tol`.
///
/// A trajectory is scripted here rather than driven by `md::MD` so the case
/// runs in the semiempirical-only build (ATV_BUILD_MD=OFF); the predictor's
/// contract is per-launch, which is exactly what this exercises.
/// Public API only. <= 60 s.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// Same 8-water cell as dftb2-ewald-gamma: polar, so Δq is large enough that
// extrapolating it is worth anything at all.
constexpr int kNat = 24;
constexpr int kZnum[] = {8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
                         8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};
constexpr double kPx[] = {
    -0.0732797495, 0.1809613674,  0.6824432840, -0.0514297404, 0.6141779072,
    0.0535641618,  -0.2407676239, 0.1094438666, -1.1414820823, -0.1626781756,
    -0.5247682562, -0.7327417158, 3.3762864232, 4.0533709385,  3.7169811408,
    3.3358584678,  2.6141476801,  3.2571129125, 3.5344097740,  2.7489353490,
    3.7406766111,  3.4372518187,  3.8256912542, 2.5020007581};
constexpr double kPy[] = {
    -0.0007076845, -0.6347547813, 0.0378056567,  -0.2982686596, -0.4893001502,
    -0.9947695860, 3.6210048464,  2.7588128450,  3.4456516298,  3.6013820956,
    3.9190698903,  2.8693140347,  -0.1267883946, 0.3623014016,  -0.2051387448,
    -0.0462642146, 0.3986494519,  0.2334386697,  3.6528025669,  3.1825548232,
    3.2723932997,  3.3480300544,  2.4794891522,  3.1760506787};
constexpr double kPz[] = {
    0.1101771075,  -0.5643195445, 0.7009388355,  3.2426048348,  2.5777188077,
    3.8948798842,  -0.0838408823, -0.3195883419, 0.1982312961,  3.5350728103,
    2.7046721608,  3.7814544541,  0.1282566371,  -0.3449745226, 1.0223418299,
    3.5186942410,  3.9690112052,  2.6037267913,  -0.1183447434, 0.1706137091,
    -0.9752827699, 3.6877641784,  3.8156371865,  3.5561262356};

constexpr int kSteps = 16;
/// Per-step displacement amplitude [Å] — the scale a 0.2 fs MD step moves a
/// hydrogen at 300 K, so the charge history is as smooth as a real trajectory.
constexpr double kAmp = 0.01;

/// Smooth, deterministic displacement of atom @p i at step @p s [Å].
double offset(int i, int s, int axis) {
    const double phase = 0.37 * i + 1.13 * axis;
    return kAmp * static_cast<double>(s) * std::sin(phase);
}

cpu::Frame make_cell(int step) {
    cpu::Frame f(kNat);
    for (int i = 0; i < kNat; ++i) {
        f.span(kAtom_type)[i] = kZnum[i];
        f.span(kAtom_x)[i] = kPx[i] + offset(i, step, 0);
        f.span(kAtom_y)[i] = kPy[i] + offset(i, step, 1);
        f.span(kAtom_z)[i] = kPz[i] + offset(i, step, 2);
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (kZnum[i] == 8) ? 15.999 : 1.008;
    }
    f.box = cpu::Box::cubic(7.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Walk the trajectory, returning total SCC iterations; @p last_energy gets
/// the final total energy [eV].
long walk(bool use_predictor, double& last_energy) {
    cpu::semiempirical::DFTB2 dftb;
    dftb.boundary = cpu::semiempirical::DftbBoundary::lattice;
    dftb.scc_tol = 1e-8;
    dftb.max_scc = 200;
    dftb.with_forces = false;
    if (use_predictor)
        dftb.predictor(3);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(dftb);
    eval.build();

    long total = 0;
    for (int s = 0; s < kSteps; ++s) {
        cpu::Frame f = make_cell(s);
        eval.run(f);
        total += dftb.last_scc_iterations();
        last_energy = dftb.last_energy();
    }
    return total;
}

} // namespace

int main() {
    double e_plain = 0, e_pred = 0;
    const long it_plain = walk(false, e_plain);
    const long it_pred = walk(true, e_pred);

    std::printf("dftb2-scc-guess: 8 H2O cell, %d launches, scc_tol 1e-8\n",
                kSteps);
    std::printf("  SCC iterations  neutral start = %ld   ASPC guess = %ld"
                "  (%.2fx)\n",
                it_plain, it_pred,
                static_cast<double>(it_plain) / static_cast<double>(it_pred));
    std::printf("  final energy    neutral start = %.10f eV\n", e_plain);
    std::printf("                  ASPC guess    = %.10f eV   |dE| = %.3e\n",
                e_pred, std::fabs(e_pred - e_plain));

    // The guess must not move the converged answer. This is the load-bearing
    // gate: a predictor that changed the energy would be a bug, not a speedup.
    constexpr double kEnergyTolEV = 1e-6;
    if (std::fabs(e_pred - e_plain) > kEnergyTolEV) {
        std::printf("FAIL predictor moved the converged energy (tol %.1e eV)\n",
                    kEnergyTolEV);
        return 1;
    }

    // Speedup gate. Measured ~1.48x over 16 launches at order 3; the first K+1
    // launches are cold, and the floor past that is the mixer's own
    // convergence rate, not the quality of the guess. Gate well below the
    // observed value so this tracks a regression, not machine noise.
    constexpr double kMinSpeedup = 1.25;
    const double speedup =
        static_cast<double>(it_plain) / static_cast<double>(it_pred);
    if (!(speedup >= kMinSpeedup)) {
        std::printf("FAIL ASPC guess saved too little (%.2fx < %.2fx)\n",
                    speedup, kMinSpeedup);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
