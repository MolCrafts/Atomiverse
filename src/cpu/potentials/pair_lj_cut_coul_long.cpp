/// @file src/cpu/potentials/pair_lj_cut_coul_long.cpp
/// @brief Fused multi-type LJ + analytic coul/long, one neighbor walk.
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/ewald_erfc.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/domain.h"
#endif

namespace ATV {
namespace cpu {

namespace {

/// Multi-type LJ (+ optional r⁻⁴ attraction) and the real-space Ewald term
/// of one pair, LAMMPS `lj/cut/coul/long` with special-bond tiers.
struct LJCoulLongKernel {
    const StorageReal* q = nullptr;
    const int* type = nullptr;
    const LJPairCoeff* tab = nullptr;
    const std::uint8_t* tier = nullptr; ///< per CSR slot; null: no specials
    double t_lj[4] = {1, 1, 1, 1};
    double t_coul[4] = {1, 1, 1, 1};
    int ntypes = 0;
    double rc_sq = 0, coul_sq = 0, qqr2e = 1, alpha = 0;
    bool has_r4 = false;

    struct Row {
        double qi_scale; ///< qqr2e · q_i
        std::size_t ti_base;
    };

    double cutoff_sq() const { return rc_sq; }

    Row row(int i) const {
        return {qqr2e * q[i], static_cast<std::size_t>(type[i])
                                  * static_cast<std::size_t>(ntypes)};
    }

    /// erfc(αr)/r and its force factor for every survivor, one xsimd batch
    /// at a time (`ewald_erfc.h`); the walk pads `r2` past `n`.
    void prepare(const Row& ri, const double* r2, int n,
                 md::PairChunk& c) const {
        if (ri.qi_scale != 0.0)
            md::ewald_pair_factors(r2, n, alpha, c.a, c.b);
    }

    md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                            const md::PairChunk& c) const {
        const int j = p.j, entry = p.entry, k = p.k;
        const double r2 = p.r2;
        double fpair = 0.0;
        double epair = 0.0;
        const double sl = tier ? t_lj[tier[entry]] : 1.0;
        if (sl != 0.0) {
            // LAMMPS: factor_lj * (forcelj, evdwl - offset) inside the
            // pair's own cut_ljsq.
            const LJPairCoeff& cf =
                tab[ri.ti_base + static_cast<std::size_t>(type[j])];
            if (r2 < cf.cut_ljsq) {
                const double r2inv = 1.0 / r2;
                const double r6inv = r2inv * r2inv * r2inv;
                fpair += sl * r2inv * (cf.lj1 * r6inv * r6inv - cf.lj2 * r6inv);
                epair +=
                    sl * (cf.lj3 * r6inv * r6inv - cf.lj4 * r6inv - cf.offset);
            }
            if (has_r4 && r2 < cf.r4_cutsq) {
                const R4Term<double> t =
                    r4_attraction(r2, cf.r4_a, cf.r4_rc5inv, cf.r4_e0);
                fpair += sl * t.fpair;
                epair += sl * t.energy;
            }
        }
        // LAMMPS special convention with a kspace partner: the erfc part is
        // UNscaled; scaled/excluded bonded pairs subtract (1-f)*qq/r — the
        // plain-Coulomb share the reciprocal sum added for them.
        if (ri.qi_scale != 0.0 && r2 < coul_sq) {
            const double qj = q[j];
            if (qj != 0.0) {
                const double qq = ri.qi_scale * qj;
                double ec = qq * c.a[k];
                double fc = qq * c.b[k];
                const double sc = tier ? t_coul[tier[entry]] : 1.0;
                if (sc != 1.0) {
                    const double r2i = 1.0 / r2;
                    const double comp = (1.0 - sc) * qq * std::sqrt(r2i);
                    ec -= comp;
                    fc -= comp * r2i;
                }
                fpair += fc;
                epair += ec;
            }
        }
        return {fpair, epair};
    }
};

} // namespace

PairLJCutCoulLong::PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs,
                                     double alpha, double cutoff)
    : Pair(cutoff), alpha_(alpha), cut_coul_(cutoff),
      cut_coulsq_(cutoff * cutoff),
      // Refuses an empty table and a non-positive cutoff.
      table_(std::move(coeffs), cutoff) {
    if (!(alpha > 0.0))
        throw std::invalid_argument("PairLJCutCoulLong: alpha must be > 0");
    refresh_cutoff_();
    set_special_bonds_amber();
}

void PairLJCutCoulLong::refresh_cutoff_() {
    // LAMMPS init_one returns max(cut_lj[i][j], cut_coul); cutforce is the
    // largest of those, and the neighbor list must cover it.
    const double rc = std::max(cut_coul_, table_.max_cutoff());
    cutoff_ = static_cast<AccReal>(rc);
    cut_sq_ = rc * rc;
}

PairLJCutCoulLong& PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) & {
    table_.set_coeff(row);
    refresh_cutoff_();
    return *this;
}

PairLJCutCoulLong PairLJCutCoulLong::set_type_coeff(const LJTypeCoeff& row) && {
    set_type_coeff(row);
    return std::move(*this);
}

PairLJCutCoulLong&
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) & {
    table_.set_r4(std::move(rows));
    refresh_cutoff_();
    return *this;
}

PairLJCutCoulLong
PairLJCutCoulLong::set_r4_attraction(std::vector<R4TypeCoeff> rows) && {
    set_r4_attraction(std::move(rows));
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_special_bonds_amber() & {
    // LAMMPS `special_bonds amber`
    return set_special_bonds(0.0, 0.0, 0.5, 0.0, 0.0, 5.0 / 6.0);
}

PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds_amber() && {
    set_special_bonds_amber();
    return std::move(*this);
}

PairLJCutCoulLong&
PairLJCutCoulLong::set_special_bonds(double lj12, double lj13, double lj14,
                                     double c12, double c13, double c14) & {
    lj12_ = lj12;
    lj13_ = lj13;
    lj14_ = lj14;
    c12_ = c12;
    c13_ = c13;
    c14_ = c14;
    any_special_ = (lj12 != 1.0 || lj13 != 1.0 || lj14 != 1.0 || c12 != 1.0
                    || c13 != 1.0 || c14 != 1.0);
    return *this;
}

PairLJCutCoulLong PairLJCutCoulLong::set_special_bonds(double lj12, double lj13,
                                                       double lj14, double c12,
                                                       double c13,
                                                       double c14) && {
    set_special_bonds(lj12, lj13, lj14, c12, c13, c14);
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_compute_stress(bool on) & {
    compute_stress_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_compute_stress(bool on) && {
    compute_stress_ = on;
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_energy_shift(bool on) & {
    energy_shift_ = on;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_energy_shift(bool on) && {
    energy_shift_ = on;
    return std::move(*this);
}

PairLJCutCoulLong& PairLJCutCoulLong::set_tail_correction(bool on) & {
    tail_ = on;
    tail_ready_ = false;
    return *this;
}
PairLJCutCoulLong PairLJCutCoulLong::set_tail_correction(bool on) && {
    set_tail_correction(on);
    return std::move(*this);
}

void PairLJCutCoulLong::build(const BuildContext& ctx) {
    scale_source_ = nullptr;
    loop_.reset();
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    {
        // The table must span every type the Frame carries; a type no row
        // names has no LJ site.
        const int* type = ctx.frame.span(kAtom_type);
        int max_type = -1;
        for (int i = 0; i < ctx.frame.nlocal(); ++i) {
            if (type[i] < 0)
                throw std::runtime_error(
                    "PairLJCutCoulLong: atom types must be >= 0");
            max_type = std::max(max_type, type[i]);
        }
        table_.build(max_type + 1, energy_shift_);
        n_types_ = table_.n_types();
        types_source_ = nullptr;
        // The type counts belong to this build's Frame.
        tail_ready_ = false;
    }
    if (md::ChunkForces* arena = ctx.runtime.md.chunk_forces)
        arena->reserve(n_threads_, ctx.frame.capacity());

    tier_lj_[1] = lj12_;
    tier_lj_[2] = lj13_;
    tier_lj_[3] = lj14_;
    tier_coul_[1] = c12_;
    tier_coul_[2] = c13_;
    tier_coul_[3] = c14_;
    scale_version_ = 0;
}

void PairLJCutCoulLong::rebuild_filtered_list(const Frame& f,
                                              const md::PairList& pairs) {
    // One byte per pair, parallel to pairs.neighbors[] (no compact copy):
    // the special tier of (i, j), 0 for an ordinary pair. Rebuilt with the
    // list because atom sort and migration reorder owner rows.
    const int n = pairs.inum;
    const auto nspecial = f.span(kAtom_nspecial);
    const auto special = f.span(kAtom_special);
    const TagInt* __restrict tag = f.span(kAtom_tag);

    // Owner → (partner tag, tier) CSR. kAtom_special lists 1-2, then 1-3,
    // then 1-4 partners, so a ring atom seen at two tiers keeps the nearer
    // one (first match below), as LAMMPS does. Tiers whose LJ and Coulomb
    // weights are both 1 are not listed at all.
    special_off_.assign(static_cast<std::size_t>(n) + 1, 0);
    special_tag_.clear();
    special_tier_.clear();
    for (int i = 0; i < n; ++i) {
        const int end[3] = {nspecial(i, 0), nspecial(i, 1), nspecial(i, 2)};
        int s = 0;
        for (int tier = 1; tier <= 3; ++tier)
            for (; s < end[tier - 1]; ++s) {
                if (tier_lj_[tier] == 1.0 && tier_coul_[tier] == 1.0)
                    continue;
                const TagInt tj = special(i, s);
                if (tj == tag[i])
                    continue;
                special_tag_.push_back(tj);
                special_tier_.push_back(static_cast<std::uint8_t>(tier));
            }
        special_off_[static_cast<std::size_t>(i) + 1] =
            static_cast<int>(special_tag_.size());
    }

    pair_tier_.assign(static_cast<std::size_t>(pairs.total_pairs), 0);
    const int* __restrict numneigh = pairs.numneigh.data();
    const int* __restrict firstneigh = pairs.firstneigh.data();
    const int* __restrict neighbors = pairs.neighbors.data();
    std::uint8_t* __restrict out = pair_tier_.data();
    const int* __restrict soff = special_off_.data();
    const TagInt* __restrict stag = special_tag_.data();
    const std::uint8_t* __restrict stier = special_tier_.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int s0 = soff[i], s1 = soff[i + 1];
        if (s0 == s1)
            continue; // already 0
        const int off = firstneigh[i];
        for (int jj = 0; jj < numneigh[i]; ++jj) {
            const TagInt tj = tag[neighbors[off + jj]];
            for (int s = s0; s < s1; ++s)
                if (stag[s] == tj) {
                    out[off + jj] = stier[s];
                    break;
                }
        }
    }
}

void PairLJCutCoulLong::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJCutCoulLong: runtime.md.pair_list is null");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;

    const int n_force = cpu::force_count(ctx);
#ifdef ATV_HAS_MPI
    const auto* dd = ctx.runtime.md.dd;
    const bool mpi = cpu::multi_rank_dd(ctx.runtime);
#endif

    // Rows (and with them types) change only at a neighbor rebuild. A type
    // outside the table would index past it; refuse it at that rebuild.
    if (types_source_ != &pairs || pairs.version != types_version_) {
        const int* type = f.span(kAtom_type);
        for (int i = 0; i < n_force; ++i)
            if (type[i] < 0 || type[i] >= n_types_)
                throw std::runtime_error(
                    "PairLJCutCoulLong: atom type " + std::to_string(type[i])
                    + " is outside the coefficient table (types 0.."
                    + std::to_string(n_types_ - 1)
                    + "); name it in a coefficient before build()");
        types_source_ = &pairs;
        types_version_ = pairs.version;
    }

    if (any_special_
        && (scale_source_ != &pairs || pairs.version != scale_version_)) {
        rebuild_filtered_list(f, pairs);
        scale_version_ = pairs.version;
        scale_source_ = &pairs;
    }

    LJCoulLongKernel kernel;
    kernel.q = f.span(kAtom_charge);
    kernel.type = f.span(kAtom_type);
    kernel.tab = table_.pairs().data();
    kernel.tier = any_special_ ? pair_tier_.data() : nullptr;
    for (int t = 0; t < 4; ++t) {
        kernel.t_lj[t] = tier_lj_[t];
        kernel.t_coul[t] = tier_coul_[t];
    }
    kernel.ntypes = n_types_;
    kernel.rc_sq = cut_sq_;
    kernel.coul_sq = cut_coulsq_;
    kernel.qqr2e = ctx.runtime.units.qqrd2e();
    kernel.alpha = alpha_;
    kernel.has_r4 = table_.has_r4();
    const int* __restrict type_ptr = kernel.type;
    const int ntypes = n_types_;

    // The style's own stress switch narrows the launch's request.
    const bool do_virial = compute_stress_ && ctx.stress_requested;
    LaunchContext walk = ctx;
    walk.stress_requested = do_virial;
    const md::PairTotals sums = loop_.run(walk, kernel, "PairLJCutCoulLong");

    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (do_virial)
        sums.virial.flush_to(f.meta.get(kStress));

    // ── LJ tail correction (LAMMPS pair_modify tail yes) ─────────────────
    // E_tail = (2π/V)·Σ_i Σ_j N_i N_j ∫_rc^∞ u_ij r² dr = etail_pref_/V,
    // P_tail = ptail_pref_/V² — a global constant over the fixed type
    // counts, so it lands on rank 0 only (the energy/virial reduction sums
    // ranks). Volume is re-read every step (barostats move the cell).
    if (tail_) {
        if (!tail_ready_) {
            // LAMMPS init_one: global per-type counts (Allreduce over the
            // owned atoms), each pair's tail from its own cut_lj.
            std::vector<double> cnt(static_cast<std::size_t>(ntypes), 0.0);
            const int n_owned = f.nlocal();
            for (int i = 0; i < n_owned; ++i) {
                const int t = type_ptr[i];
                if (t >= 0 && t < ntypes)
                    cnt[static_cast<std::size_t>(t)] += 1.0;
            }
#ifdef ATV_HAS_MPI
            if (mpi)
                MPI_Allreduce(MPI_IN_PLACE, cnt.data(), ntypes, MPI_DOUBLE,
                              MPI_SUM, dd->cart_comm);
#endif
            const LJTail t = table_.tail(cnt);
            etail_pref_ = t.energy;
            ptail_pref_ = t.virial;
            tail_ready_ = true;
        }
#ifdef ATV_HAS_MPI
        const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;
#else
        const bool first_rank = true;
#endif
        if (first_rank) {
            const double vol = f.box.volume();
            f.meta.get(kEnergy) += etail_pref_ / vol;
            f.meta.get(kEnergyPair) += etail_pref_ / vol;
            if (do_virial) {
                // Each diagonal += ptail_pref_/V so tr(W)/(3V) = P_tail.
                f.meta.get(kStress)[0] += ptail_pref_ / vol;
                f.meta.get(kStress)[1] += ptail_pref_ / vol;
                f.meta.get(kStress)[2] += ptail_pref_ / vol;
            }
        }
    }
}

void PairLJCutCoulLong::destroy(const DestroyContext&) {
    special_off_.clear();
    special_tag_.clear();
    special_tier_.clear();
    pair_tier_.clear();
    scale_version_ = 0;
    scale_source_ = nullptr;
    // any_special_ is configuration (set_special_bonds), not build state: a
    // destroy → build cycle must keep the exclusions.
}

std::size_t PairLJCutCoulLong::memory_bytes() const {
    std::size_t b = 0;
    b += pair_tier_.capacity();
    b += special_tag_.capacity() * sizeof(TagInt);
    return b;
}

} // namespace cpu
} // namespace ATV
