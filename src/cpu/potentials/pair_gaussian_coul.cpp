/// @file src/cpu/potentials/pair_gaussian_coul.cpp
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

PairGaussianCoul::PairGaussianCoul(double cutoff,
                                   std::vector<double> eta_by_type,
                                   bool include_self)
    : Pair(cutoff), eta_by_type_(std::move(eta_by_type)),
      include_self_(include_self) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairGaussianCoul: cutoff must be > 0 [Å]");
    cutoff_sq_ = cutoff * cutoff;
}

void PairGaussianCoul::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;

    const StorageReal* __restrict q = f.span(kAtom_charge);
    const LocalInt* __restrict type = f.span(kAtom_type);
    const double K = ctx.runtime.units.qqrd2e();
    double energy = 0.0;

    // This style is *not* a `LongRangeParticipant`: it is a self-contained
    // real-space Coulomb with a hard cutoff (Reed / MetalWalls form), so no
    // reciprocal solver carries any part of this energy and there is nothing
    // to double count — the cutoff pair sum is the whole stress contribution.
    // The Gaussian self-energy is deliberately absent from it: it depends only
    // on η and q, is independent of every coordinate and of the cell, and so
    // contributes nothing to dE/dV.
    struct Kernel {
        const PairGaussianCoul* self;
        const StorageReal* q;
        const LocalInt* type;
        double K, rc_sq;
        struct Row {
            double qi, ei;
        };
        double cutoff_sq() const { return rc_sq; }
        Row row(int i) const {
            return {static_cast<double>(q[i]),
                    self->eta_of(static_cast<int>(type[i]))};
        }
        md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                                const md::PairChunk&) const {
            const int j = p.j;
            const double r = std::sqrt(p.r2);
            const double ej = self->eta_of(static_cast<int>(type[j]));
            const double pref = K * ri.qi * static_cast<double>(q[j]);
            // F_i = −pref·G'(r)/r · (r_i − r_j).
            return {-pref * GaussianChargeKernel::dgreens_dr(r, ri.ei, ej) / r,
                    pref * GaussianChargeKernel::greens(r, ri.ei, ej)};
        }
    };
    // eta_of throws on a type outside its table; surface that here, serially,
    // rather than inside the threaded walk.
    for (int i = 0; i < f.n_total(); ++i)
        (void)eta_of(static_cast<int>(type[i]));
    const Kernel kernel{this, q, type, K, cutoff_sq_};
    const md::PairTotals sums = loop_.run(ctx, kernel, "PairGaussianCoul");
    energy += sums.energy;

    if (include_self_) {
        for (int i = 0; i < f.nlocal(); ++i) {
            const double qi = static_cast<double>(q[i]);
            const double ei = eta_of(static_cast<int>(type[i]));
            if (GaussianChargeKernel::is_gaussian(ei))
                energy += GaussianChargeKernel::self_energy(K, qi, ei);
        }
    }

    f.meta.get(kEnergy) += energy;
    f.meta.get(kEnergyPair) += energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
