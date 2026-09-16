/// @file regressions/minimizer/fire_lj_dimer/reg_fire_lj_dimer.cpp
/// @brief END-TO-END: FIRE-minimize a displaced LJ dimer to r*=2^(1/6)σ.
/// Public headers only. See REFERENCE.md.
///
/// Oracle (analytic 12-6 LJ minimum; PairLJCut is energy-shifted at cutoff):
///   r* = 2^{1/6} σ
///   U_shift(r*) in (-1.1ε, -0.9ε) for this ε/σ/rc setup
/// Algorithm: Bitzek et al., PRL 97, 170201 (2006) FIRE 1.0

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    const double eps = 0.0104; // eV
    const double sigma = 3.4;  // Å
    const double cutoff = 2.5 * sigma;
    const double r_star = sigma * std::pow(2.0, 1.0 / 6.0);
    // Start farther than the LBFGS dimer case so |F| is above a tight ftol
    // until the basin is reached (FIRE takes smaller early steps than L-BFGS).
    const double r0 = 1.5 * r_star;
    const double L = 40.0;

    cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = r0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_mass)[i] = 39.948;
        frame.span(ATV::kAtom_type)[i] = 0;
        frame.span(ATV::kAtom_tag)[i] = i + 1;
    }
    frame.box = cpu::Box::cubic(L);

    cpu::minimizer::Minimize opt;
    cpu::minimizer::FIRE fire;
    // Pseudo-time scale: ASE defaults (dt0=0.1) assume O(1) forces; LJ dimer
    // residual at start is ≪1 eV/Å, so grow dt_max so steps remain useful.
    fire.dt0 = 1.0;
    fire.dt_max = 20.0;
    opt.set_optimizer_ref(fire);
    opt.add_criterion(cpu::ForceConverged{1e-3, 5e-4, 0.05, 0.02, 1e-6});
    opt.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    opt.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    opt.build();
    opt.run(frame, 2000);

    const double dx = frame.span(ATV::kAtom_x)[1] - frame.span(ATV::kAtom_x)[0];
    const double dy = frame.span(ATV::kAtom_y)[1] - frame.span(ATV::kAtom_y)[0];
    const double dz = frame.span(ATV::kAtom_z)[1] - frame.span(ATV::kAtom_z)[0];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double U = frame.meta.get(ATV::kEnergy);
    const double fmax = std::max(
        std::hypot(frame.span(ATV::kAtom_fx)[0], frame.span(ATV::kAtom_fy)[0],
                   frame.span(ATV::kAtom_fz)[0]),
        std::hypot(frame.span(ATV::kAtom_fx)[1], frame.span(ATV::kAtom_fy)[1],
                   frame.span(ATV::kAtom_fz)[1]));
    const std::string reason = opt.stop_reason();

    std::printf("FIRE LJ dimer: r=%.6f (r*=%.6f)  U=%.8f (−ε=%.8f)  "
                "fmax=%.3e  converged=%d  reason=%s\n",
                r, r_star, U, -eps, fmax, (int)opt.converged(), reason.c_str());

    bool ok = true;
    if (!(std::fabs(r - r_star) < 0.01)) {
        std::printf("FAIL: distance off analytic minimum\n");
        ok = false;
    }
    // PairLJCut is energy-shifted at the cutoff: U_shift(r*) = −ε − U(rc) > −ε.
    if (!(U < -0.9 * eps) || !(U > -1.1 * eps)) {
        std::printf("FAIL: energy not in the shifted-LJ well near −ε "
                    "(U=%.8f, −ε=%.8f)\n",
                    U, -eps);
        ok = false;
    }
    if (!(fmax < 0.05)) {
        std::printf("FAIL: residual force too large\n");
        ok = false;
    }
    if (!opt.converged()) {
        std::printf("FAIL: not converged\n");
        ok = false;
    }
    if (reason.find("force_converged") == std::string::npos) {
        std::printf("FAIL: stop_reason expected force_converged, got %s\n",
                    reason.c_str());
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}
