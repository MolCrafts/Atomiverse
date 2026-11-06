/// @file src/cpu/potentials/coul_long.cpp
/// @brief Real-space screened Coulomb implementation (CPU).
#include "atomiverse/cpu/potentials/coul_long.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
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

    md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                            const md::PairChunk&) const {
        const double qj = q[p.j];
        if (qj == 0.0)
            return {};
        const double sc = special ? special->scale_owner(ri.i, tag[p.j]) : 1.0;
        const double r = std::sqrt(p.r2);
        const double r2inv = 1.0 / p.r2;
        const double grij = alpha * r;
        const double expm2 = std::exp(-grij * grij);
        const double erfcv = std::erfc(grij);
        const double prefactor = qqr2e * ri.qi * qj / r;
        double forcecoul = prefactor * (erfcv + kEwaldF * grij * expm2);
        if (sc != 1.0)
            forcecoul -= (1.0 - sc) * prefactor;
        const double energy = sc != 1.0
                                  ? prefactor * erfcv - (1.0 - sc) * prefactor
                                  : prefactor * erfcv;
        return {forcecoul * r2inv, energy};
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
    kernel.qqr2e = ATV::units::md_qqr2e;
    const md::PairTotals sums = loop_.run(ctx, md::PairRows::of(pairs), kernel);

    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

void CoulLong::destroy(const DestroyContext&) {
    special_.clear();
    any_special_ = false;
    special_version_ = 0;
}

} // namespace cpu
} // namespace ATV
