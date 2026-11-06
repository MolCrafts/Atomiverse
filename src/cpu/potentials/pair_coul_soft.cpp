/// @file src/cpu/potentials/pair_coul_soft.cpp
/// @brief Beutler/GROMACS one-leg soft-core Coulomb (neighbor-list walk).

#include "atomiverse/cpu/potentials/pair_coul_soft.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

PairCoulSoft::PairCoulSoft(double cutoff, double sigma_c, double alpha,
                           double p)
    : Pair(cutoff), sigma_c_(sigma_c), alpha_(alpha), p_(p) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairCoulSoft: cutoff must be > 0");
    if (!(sigma_c > 0.0))
        throw std::invalid_argument("PairCoulSoft: sigma_c must be > 0");
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairCoulSoft: alpha must be > 0");
    if (!(p > 0.0))
        throw std::invalid_argument("PairCoulSoft: p must be > 0");

    sigma6_ = sigma_c * sigma_c * sigma_c * sigma_c * sigma_c * sigma_c;
    cutoff_sq_ = cutoff * cutoff;
    inv_rc_ = 1.0 / cutoff;
}

void PairCoulSoft::build(const BuildContext& ctx) {
    loop_.reset();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());
    dU_dlambda_ = 0;
}

void PairCoulSoft::destroy(const DestroyContext&) {}

namespace {

/// Beutler one-leg soft-core Coulomb pair, with its dU/dλ as the extra.
struct CoulSoftKernel {
    const StorageReal* q = nullptr;
    double rc_sq = 0, inv_rc = 0, qqr2e = 1, lam = 1, soft0 = 0,
           dsoft0_dlam = 0;

    struct Row {
        double qi;
    };

    double cutoff_sq() const { return rc_sq; }
    Row row(int i) const { return {static_cast<double>(q[i])}; }
    bool skip(const Row& ri) const { return ri.qi == 0.0; }

    md::PairTermX operator()(const Row& ri, const md::PairSite& p,
                             const md::PairChunk&) const {
        const double qj = q[p.j];
        if (qj == 0.0)
            return {};
        const double r2 = p.r2;
        const double r6 = r2 * r2 * r2;
        const double r_lam = std::pow(soft0 + r6, 1.0 / 6.0);
        const double inv_rl = 1.0 / r_lam;
        const double qq = qqr2e * ri.qi * qj;

        // V = λ qq (1/r_λ − 1/r_c)
        const double vpair = qq * (inv_rl - inv_rc);
        md::PairTermX t;
        t.energy = lam * vpair;

        // ∂V/∂λ = vpair + λ qq (−1/r_λ²) ∂r_λ/∂λ
        // ∂r_λ/∂λ = (1/6)(soft0+r6)^{−5/6} ∂soft0/∂λ
        //         = dsoft0_dlam / (6 r_λ⁵)
        const double r_lam5 = r_lam * r_lam * r_lam * r_lam * r_lam;
        const double drlam_dlam = dsoft0_dlam / (6.0 * r_lam5);
        t.extra = vpair + lam * qq * (-inv_rl * inv_rl) * drlam_dlam;

        // fpair such that F_i = fpair * dx:
        // F_r = λ qq r⁵ / r_λ⁷  ⇒ fpair = λ qq r⁴ / r_λ⁷
        if (lam > 0.0) {
            const double r4 = r2 * r2;
            const double r_lam7 = r_lam5 * r_lam * r_lam;
            t.fpair = lam * qq * r4 / r_lam7;
        }
        t.fvirial = t.fpair;
        return t;
    }
};

} // namespace

void PairCoulSoft::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairCoulSoft::compute: runtime.md.pair_list is null");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    const double sig6 = sigma6_;
    const double alpha = alpha_;
    const double p = p_;
    const double lam = std::min(1.0, std::max(0.0, lambda));
    const double one_m_lam = 1.0 - lam;
    const double soft0 = alpha * sig6 * std::pow(one_m_lam, p);
    // d soft0 / d λ = α σ⁶ p (1−λ)^{p−1} * (−1)
    const double dsoft0_dlam =
        (p == 1.0) ? (-alpha * sig6)
                   : (-alpha * sig6 * p * std::pow(one_m_lam, p - 1.0));

    if (!(lam > 0.0) && soft0 == 0.0) {
        dU_dlambda_ = 0;
        return;
    }

    CoulSoftKernel kernel;
    kernel.q = f.span(kAtom_charge);
    kernel.rc_sq = cutoff_sq_;
    kernel.inv_rc = inv_rc_;
    kernel.qqr2e = units::md_qqr2e;
    kernel.lam = lam;
    kernel.soft0 = soft0;
    kernel.dsoft0_dlam = dsoft0_dlam;
    const md::PairTotals sums = loop_.run(ctx, md::PairRows::of(pairs), kernel);
    const double energy_total = sums.energy;
    const double dU_total = sums.extra;
    const md::Voigt6& vir = sums.virial;

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (ctx.stress_requested)
        vir.flush_to(f.meta.get(kStress));
    dU_dlambda_ = dU_total;
}

} // namespace cpu
} // namespace ATV
