/// @file regressions/minimizer/lj_dimer/reg_lj_dimer.cpp
/// @brief END-TO-END: minimize a displaced LJ dimer to r*=2^(1/6)σ, U=−ε.
/// Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    const double eps = 0.0104; // eV
    const double sigma = 3.4;  // Å
    const double cutoff = 2.5 * sigma;
    const double r_star = sigma * std::pow(2.0, 1.0 / 6.0);
    const double r0 = 1.2 * r_star; // displaced start
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
    opt.set_optimizer(cpu::minimizer::LBFGS{10, 0.05, 20});
    opt.add_criterion(cpu::ForceConverged{0.01, 0.005, 0.05, 0.02, 1e-6});
    opt.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    opt.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    opt.build();
    opt.run(frame, 200);

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

    std::printf("LJ dimer: r=%.6f (r*=%.6f)  U=%.8f (−ε=%.8f)  "
                "fmax=%.3e  converged=%d  reason=%s\n",
                r, r_star, U, -eps, fmax, (int)opt.converged(), reason.c_str());

    bool ok = true;
    if (!(std::fabs(r - r_star) < 0.01)) {
        std::printf("FAIL: distance off analytic minimum\n");
        ok = false;
    }
    // PairLJCut is energy-shifted at the cutoff: U_shift(r*) = −ε − U(rc) > −ε.
    // Oracle: force vanishes and r→r*; energy must be below −0.9ε (bound state).
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
    if (!opt.converged()
        || reason.find("force_converged") == std::string::npos) {
        std::printf("FAIL: expected ForceConverged stop reason\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS: LJ dimer minimized to analytic r* and −ε\n");
    return 0;
}
