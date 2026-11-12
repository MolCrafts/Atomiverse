/// @file src/cpu/potentials/coul_long.cpp
/// @brief Real-space screened Coulomb implementation (CPU).
#include "atomiverse/cpu/potentials/coul_long.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/ewald_erfc.h"
#include "atomiverse/cpu/potentials/pair_curvature.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/domain.h"
#endif

namespace ATV {
namespace cpu {

static constexpr double kEwaldF = 1.1283791670955126;

CoulLong::CoulLong(double alpha, double cutoff)
    : alpha_(alpha), cutoff_(cutoff), cutoff_sq_(cutoff * cutoff) {
    if (alpha <= 0.0)
        throw std::invalid_argument("CoulLong: alpha must be > 0");
    if (cutoff <= 0.0)
        throw std::invalid_argument("CoulLong: cutoff must be > 0");
}

CoulLong& CoulLong::set_special_bonds(double s12, double s13, double s14) & {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return *this;
}

CoulLong CoulLong::set_special_bonds(double s12, double s13, double s14) && {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return std::move(*this);
}

void CoulLong::build(const BuildContext& ctx) {
    loop_.reset();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads, ctx.frame.capacity());

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    special_.clear();
    special_version_ = 0;
    if (!any_special_)
        return;
    const int n_owners = ctx.frame.nlocal();
    special_.rebuild(ctx.frame, n_owners);
}

namespace {

/// Real-space Ewald pair, LAMMPS `coul/long` with the special convention of
/// a kspace partner: the erfc part is unscaled; scaled / excluded bonded
/// pairs subtract the (1-f)·qq/r plain-Coulomb share the reciprocal sum
/// added for them, so zero-scale (1-2/1-3) pairs stay in the walk.
struct CoulLongKernel {
    const StorageReal* q = nullptr;
    const TagInt* tag = nullptr; ///< non-null with special bonds
    const md::SpecialScaleTable* special = nullptr;
    double rc_sq = 0, alpha = 0, qqr2e = 1;

    struct Row {
        int i;
        double qi;
    };

    double cutoff_sq() const { return rc_sq; }
    Row row(int i) const { return {i, static_cast<double>(q[i])}; }
    bool skip(const Row& ri) const { return ri.qi == 0.0; }

    /// erfc(αr)/r and its force factor for every survivor, one xsimd batch
    /// at a time (`ewald_erfc.h`); the walk pads `r2` past `n`.
    void prepare(const Row&, const double* r2, int n, md::PairChunk& c) const {
        md::ewald_pair_factors(r2, n, alpha, c.a, c.b);
    }

    md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                            const md::PairChunk& c) const {
        const double qj = q[p.j];
        if (qj == 0.0)
            return {};
        const double qq = qqr2e * ri.qi * qj;
        double energy = qq * c.a[p.k];
        double fpair = qq * c.b[p.k];
        const double sc = special ? special->scale_owner(ri.i, tag[p.j]) : 1.0;
        if (sc != 1.0) {
            const double r2inv = 1.0 / p.r2;
            const double comp = (1.0 - sc) * qq * std::sqrt(r2inv);
            energy -= comp;
            fpair -= comp * r2inv;
        }
        return {fpair, energy};
    }
};

} // namespace

void CoulLong::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error("CoulLong::launch: runtime.md.pair_list is "
                                 "null. Add a NeighborList "
                                 "to the force phase ahead of CoulLong.");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    if (any_special_ && pairs.version != special_version_) {
        special_.s12 = s12_;
        special_.s13 = s13_;
        special_.s14 = s14_;
        special_.rebuild(f, pairs.inum);
        special_version_ = pairs.version;
    }

    CoulLongKernel kernel;
    kernel.q = f.span(kAtom_charge);
    kernel.tag = any_special_ ? f.span(kAtom_tag).data() : nullptr;
    kernel.special = any_special_ ? &special_ : nullptr;
    kernel.rc_sq = cutoff_sq_;
    kernel.alpha = alpha_;
    kernel.qqr2e = ctx.runtime.units.qqrd2e();
    const md::PairTotals sums = loop_.run(ctx, md::PairRows::of(pairs), kernel);

    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

// The pairs of launch(): the published half list, special scales included.
void CoulLong::add_curvature(const LaunchContext& ctx,
                             const CurvatureRows& rows) {
    const Frame& f = ctx.frame;
    const StorageReal* q = f.span(kAtom_charge).data();
    const TagInt* tag = any_special_ ? f.span(kAtom_tag).data() : nullptr;
    const double qqr2e = ctx.runtime.units.qqrd2e();
    const double alpha = alpha_;
    const double c = kEwaldF * alpha; // 2α/√π
    md::for_each_curvature_pair(
        ctx, cutoff_sq_, /*use_list=*/true, "CoulLong",
        [&](int i, int j, int, double dx, double dy, double dz, double r2) {
            const double qq = qqr2e * q[i] * q[j];
            if (qq == 0.0)
                return;
            const double sc = tag ? special_.scale_owner(i, tag[j]) : 1.0;
            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double inv_r2 = inv_r * inv_r;
            const double expm2 = std::exp(-alpha * alpha * r2);
            const double erfcv = std::erfc(alpha * r);
            // erfc(αr)/r and the plain-Coulomb remainder −(1−s)/r.
            const double g1 = -erfcv * inv_r2 - c * expm2 * inv_r;
            const double g2 = 2.0 * erfcv * inv_r2 * inv_r
                              + 2.0 * c * expm2 * inv_r2
                              + 2.0 * alpha * alpha * c * expm2;
            const double rem = 1.0 - sc;
            md::RadialCurvature k;
            k.d1 = qq * (g1 + rem * inv_r2);
            k.d2 = qq * (g2 - 2.0 * rem * inv_r2 * inv_r);
            k.lap = qq * 2.0 * alpha * alpha * c * expm2;
            k.dlap = -qq * 4.0 * alpha * alpha * alpha * alpha * c * r * expm2;
            md::add_pair_curvature(rows, i, j, dx, dy, dz, r, k);
        });
}

void CoulLong::destroy(const DestroyContext&) {
    special_.clear();
    any_special_ = false;
    special_version_ = 0;
}

} // namespace cpu
} // namespace ATV
