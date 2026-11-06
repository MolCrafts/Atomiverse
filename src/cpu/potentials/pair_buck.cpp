/// @file cpu/potentials/pair_buck.cpp
#include "atomiverse/cpu/potentials/pair_buck.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

PairBuck::PairBuck(double A, double rho, double C, double cutoff)
    : Pair(cutoff), A_(A), rho_(rho), C_(C) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairBuck: cutoff must be > 0");
    if (!(rho > 0.0))
        throw std::invalid_argument("PairBuck: rho must be > 0");
    if (!(A >= 0.0))
        throw std::invalid_argument("PairBuck: A must be ≥ 0");
    if (!(C >= 0.0))
        throw std::invalid_argument("PairBuck: C must be ≥ 0");
    cutoff_sq_ = cutoff * cutoff;
    // U(rc) = A exp(-rc/rho) - C/rc^6  (shift so U_shifted(rc)=0)
    const double rc6 = cutoff_sq_ * cutoff_sq_ * cutoff_sq_;
    v_shift_ = A * std::exp(-cutoff / rho) - C / rc6;
}

void PairBuck::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const double inv_rho = 1.0 / rho_;
    const auto kernel =
        md::RadialKernel{cutoff_sq_, [&](double r2) {
                             const double r = std::sqrt(r2);
                             const double r6 = r2 * r2 * r2;
                             const double expterm = std::exp(-r * inv_rho);
                             const double v = A_ * expterm - C_ / r6 - v_shift_;
                             // dU/dr (unshifted) = -(A/rho) exp(-r/rho) + 6 C / r^7
                             const double dUdr =
                                 -A_ * inv_rho * expterm + 6.0 * C_ / (r6 * r);
                             // F on i from j: F_i = -∇_i U; r = |r_j - r_i|, ∇_i r = -r̂
                             // → F_i = dUdr * r̂ with r̂ = (r_j-r_i)/r → F_i = (dUdr/r) * (r_j-r_i)
                             // Convention in this codebase (Yukawa/Morse): fpair = -dUdr/r and
                             // F_i -= fpair * (r_j-r_i)  ⇔  F_i = (dUdr/r) * (r_j-r_i).
                             const double fpair = -dUdr / r;
                             return md::PairTerm{fpair, v};
                         }};
    const md::PairTotals sums =
        md::PairLoop<decltype(kernel)>::run(ctx, kernel, "PairBuck");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
