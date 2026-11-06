/// @file src/cpu/potentials/pair_lj_soft.cpp
/// @brief Beutler/GROMACS one-leg soft-core LJ (neighbor-list walk).

#include "atomiverse/cpu/potentials/pair_lj_soft.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

PairLJSoft::PairLJSoft(double epsilon, double sigma, double cutoff,
                       double alpha, double p)
    : Pair(cutoff), epsilon_(epsilon), sigma_(sigma), alpha_(alpha), p_(p) {
    if (!(epsilon > 0.0))
        throw std::invalid_argument("PairLJSoft: epsilon must be > 0");
    if (!(sigma > 0.0))
        throw std::invalid_argument("PairLJSoft: sigma must be > 0");
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairLJSoft: cutoff must be > 0");
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairLJSoft: alpha must be > 0");
    if (!(p > 0.0))
        throw std::invalid_argument("PairLJSoft: p must be > 0");

    sigma6_ = sigma * sigma * sigma * sigma * sigma * sigma;
    cutoff_sq_ = cutoff * cutoff;
    const double r2inv = 1.0 / cutoff_sq_;
    const double r6inv = r2inv * r2inv * r2inv * sigma6_;
    v_shift_ = 4.0 * epsilon_ * (r6inv * r6inv - r6inv);
}

void PairLJSoft::build(const BuildContext& ctx) {
    loop_.reset();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());
    dU_dlambda_ = 0;
}

void PairLJSoft::destroy(const DestroyContext&) {}

namespace {

/// Beutler one-leg soft-core LJ pair, with its dU/dλ as the extra.
struct LJSoftKernel {
    double rc_sq = 0, sig6 = 0, eps4 = 0, eps24 = 0, vshift = 0, lam = 1,
           soft0 = 0, dsoft0_dlam = 0;

    double cutoff_sq() const { return rc_sq; }
    int row(int) const { return 0; }

    md::PairTermX operator()(int, const md::PairSite& p,
                             const md::PairChunk&) const {
        const double r2 = p.r2;
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double r_lam = std::pow(soft0 + r6, 1.0 / 6.0);
        const double r_lam2 = r_lam * r_lam;
        const double r2inv_l = 1.0 / r_lam2;
        const double r6inv_l = r2inv_l * r2inv_l * r2inv_l * sig6;

        // Energy: λ (V_LJ(r_λ) − V_shift)
        const double vlj = eps4 * (r6inv_l * r6inv_l - r6inv_l);
        const double vpair = vlj - vshift;
        md::PairTermX t;
        t.energy = lam * vpair;

        // ∂U/∂λ = vpair + λ (dV_LJ/dr_λ)(dr_λ/dλ)
        // dV/dr = −fpair_h · r_λ  (fpair definition F = fpair·dx)
        const double fpair_h =
            eps24 * r2inv_l * (2.0 * r6inv_l * r6inv_l - r6inv_l);
        const double dVdr = -fpair_h * r_lam;
        const double r_lam5 = r_lam2 * r_lam2 * r_lam; // r_λ^4 * r_λ
        const double drlam_dlam = dsoft0_dlam / (6.0 * r_lam5);
        t.extra = vpair + lam * dVdr * drlam_dlam;

        // Hard fpair at r_λ, then soft map: fpair_s = λ fpair_h (r/r_λ)^4
        if (lam > 0.0) {
            const double ratio = r / r_lam;
            const double ratio2 = ratio * ratio;
            t.fpair = lam * fpair_h * ratio2 * ratio2;
        }
        t.fvirial = t.fpair;
        return t;
    }
};

} // namespace

void PairLJSoft::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJSoft::compute: runtime.md.pair_list is null. Did you add a "
            "NeighborList ahead of the soft LJ potential?");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    const double rc_sq = cutoff_sq_;
    const double sig6 = sigma6_;
    const double eps4 = 4.0 * epsilon_;
    const double eps24 = 24.0 * epsilon_;
    const double vshift = v_shift_;
    const double alpha = alpha_;
    const double p = p_;
    // Clamp λ to a sane range for stability; user still owns the member.
    const double lam = std::min(1.0, std::max(0.0, lambda));
    const double one_m_lam = 1.0 - lam;
    const double soft0 = alpha * sig6 * std::pow(one_m_lam, p);
    const double dsoft0_dlam =
        (p == 1.0) ? (-alpha * sig6)
                   : (-alpha * sig6 * p * std::pow(one_m_lam, p - 1.0));

    if (!(lam > 0.0) && soft0 == 0.0) {
        dU_dlambda_ = 0;
        return;
    }

    LJSoftKernel kernel;
    kernel.rc_sq = rc_sq;
    kernel.sig6 = sig6;
    kernel.eps4 = eps4;
    kernel.eps24 = eps24;
    kernel.vshift = vshift;
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
