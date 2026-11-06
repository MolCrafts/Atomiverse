/// @file cpu/potentials/pair_morse.cpp
/// @brief PairMorse — live "one new pair style = two files" demonstration.

#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

#include <cmath>

namespace ATV {
namespace cpu {

PairMorse::PairMorse(double D, double alpha, double r0, double cutoff)
    : Pair(cutoff), D_(D), alpha_(alpha), r0_(r0) {
    cutoff_sq_ = cutoff * cutoff;
    // Shift the potential so V(cutoff) = 0 — no jump at the cutoff,
    // matching LAMMPS `pair_modify shift yes`.
    const double xc = std::exp(-alpha * (cutoff - r0));
    v_shift_ = D * (1.0 - xc) * (1.0 - xc);
}

void PairMorse::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;

    const auto kernel = md::RadialKernel{
        cutoff_sq_, [&](double r2) {
            const double r = std::sqrt(r2);
            const double xexp = std::exp(-alpha_ * (r - r0_));
            const double one_minus_x = 1.0 - xexp;
            const double v = D_ * one_minus_x * one_minus_x - v_shift_;
            // dV/dr = 2 D α (1 − x) x
            const double dvdr = 2.0 * D_ * alpha_ * one_minus_x * xexp;
            const double fpair = -dvdr / r; // F_ij = fpair · r_ij
            return md::PairTerm{fpair, v};
        }};
    const md::PairTotals sums =
        md::PairLoop<decltype(kernel)>::run(ctx, kernel, "PairMorse");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
