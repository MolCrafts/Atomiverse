/// @file cpu/potentials/pair_yukawa.cpp
#include "atomiverse/cpu/potentials/pair_yukawa.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

PairYukawa::PairYukawa(double A, double kappa, double cutoff)
    : Pair(cutoff), A_(A), kappa_(kappa) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairYukawa: cutoff must be > 0");
    if (!(kappa >= 0.0))
        throw std::invalid_argument("PairYukawa: kappa must be ≥ 0");
    cutoff_sq_ = cutoff * cutoff;
    const double urc = A * std::exp(-kappa * cutoff) / cutoff;
    v_shift_ = urc;
}

void PairYukawa::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;

    const auto kernel = md::RadialKernel{
        cutoff_sq_, [&](double r2) {
            const double r = std::sqrt(r2);
            const double e = std::exp(-kappa_ * r);
            const double inv_r = 1.0 / r;
            const double v = A_ * e * inv_r - v_shift_;
            // The shift is a constant, so it drops out of the derivative:
            // U = A e^{−κr}/r ⇒ dU/dr = −A e^{−κr}(κ/r + 1/r²).
            const double dUdr = -A_ * e * (kappa_ * inv_r + inv_r * inv_r);
            const double fpair =
                -dUdr / r; // F on i from j: fpair * r_ij with r_ij = r_j-r_i
            return md::PairTerm{fpair, v};
        }};
    const md::PairTotals sums = loop_.run(ctx, kernel, "PairYukawa");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
