/// @file cpu/potentials/pair_wca.cpp
#include "atomiverse/cpu/potentials/pair_wca.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/pair_kernel.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

PairWCA::PairWCA(double epsilon, double sigma)
    : Pair(std::pow(2.0, 1.0 / 6.0) * sigma), epsilon_(epsilon), sigma_(sigma) {
    if (!(epsilon > 0.0) || !(sigma > 0.0))
        throw std::invalid_argument("PairWCA: epsilon and sigma must be > 0");
    const double rc = pair_cutoff();
    cutoff_sq_ = rc * rc;
    // At rc = 2^{1/6} σ: (σ/rc)^6 = 1/2, (σ/rc)^12 = 1/4 → 4ε(1/4−1/2)= −ε
    v_shift_ = -epsilon_;
}

void PairWCA::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const double sig2 = sigma_ * sigma_;
    // The 12-6 math is the backend-neutral potentials::LJKernel, cut and
    // shifted at the WCA minimum.
    const potentials::LJKernel lj{epsilon_, sig2, cutoff_sq_, v_shift_};
    const auto kernel =
        md::RadialKernel{cutoff_sq_, [&](double r2) {
                             const potentials::PairContribution<double> c =
                                 lj.raw(r2, 0.0, 0.0);
                             return md::PairTerm{c.fpair, c.energy};
                         }};
    const md::PairTotals sums = loop_.run(ctx, kernel, "PairWCA");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
