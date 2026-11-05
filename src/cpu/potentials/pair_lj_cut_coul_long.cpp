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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/domain.h"
#endif

namespace ATV {
namespace cpu {

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
    const int n = ctx.frame.capacity();
    if (n_threads_ > 1) {
        fx_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
        fy_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
        fz_thr_.assign(n_threads_, std::vector<double>(n, 0.0));
    } else {
        fx_thr_.clear();
        fy_thr_.clear();
        fz_thr_.clear();
    }

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

    if (n_threads_ > 1
        && static_cast<int>(fx_thr_.empty() ? 0 : fx_thr_[0].size())
               < n_force) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n_force, 0.0);
            fy_thr_[t].resize(n_force, 0.0);
            fz_thr_[t].resize(n_force, 0.0);
        }
    }

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

    const int k_inum = pairs.inum;
    const int* __restrict k_ilist = pairs.ilist.data();
    const int* __restrict k_numneigh = pairs.numneigh.data();
    const int* __restrict k_firstneigh = pairs.firstneigh.data();
    const int* __restrict k_neighbors = pairs.neighbors.data();
    const std::uint8_t* __restrict k_tier =
        any_special_ ? pair_tier_.data() : nullptr;
    const double t_lj[4] = {tier_lj_[0], tier_lj_[1], tier_lj_[2], tier_lj_[3]};
    const double t_coul[4] = {tier_coul_[0], tier_coul_[1], tier_coul_[2],
                              tier_coul_[3]};

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const StorageReal* __restrict q_ptr = f.span(kAtom_charge);
    const int* __restrict type_ptr = f.span(kAtom_type);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

    const double rc_sq = cut_sq_;
    const double coul_sq = cut_coulsq_;
    const double qqr2e = ATV::units::md_qqr2e;
    const int ntypes = n_types_;
    const LJPairCoeff* __restrict tab = table_.pairs().data();
    const bool has_r4 = table_.has_r4();
    const double alpha = alpha_;
    const bool do_virial = compute_stress_ && ctx.stress_requested;

    double energy_total = 0;
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

// if(n_threads_>1): pure serial path skips OpenMP fork (OMP=1 fair vs LAMMPS).
#pragma omp parallel num_threads(n_threads_) if (n_threads_ > 1)               \
    reduction(+ : energy_total) reduction(+ : v_xx) reduction(+ : v_yy)        \
    reduction(+ : v_zz) reduction(+ : v_xy) reduction(+ : v_xz)                \
    reduction(+ : v_yz)
    {
        int tid = 0, team_size = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        const bool direct = (n_threads_ == 1);
        double* __restrict fxt;
        double* __restrict fyt;
        double* __restrict fzt;
        if (direct) {
            fxt = fx_ptr;
            fyt = fy_ptr;
            fzt = fz_ptr;
        } else {
            std::fill(fx_thr_[tid].begin(), fx_thr_[tid].begin() + n_force,
                      0.0);
            std::fill(fy_thr_[tid].begin(), fy_thr_[tid].begin() + n_force,
                      0.0);
            std::fill(fz_thr_[tid].begin(), fz_thr_[tid].begin() + n_force,
                      0.0);
            fxt = fx_thr_[tid].data();
            fyt = fy_thr_[tid].data();
            fzt = fz_thr_[tid].data();
        }

        // Survivor-compaction staging, one chunk of a neighbor list at a
        // time. 7 x 512 doubles + the index list is ~30 KB, so a chunk and
        // its survivor list stay L1-resident (48 KB/core) alongside the
        // random position gathers of pass 1. The survivor r² and its two
        // Ewald factors are padded to the xsimd batch width.
        constexpr int kChunk = 512;
        constexpr int kLanes = md::ewald_batch_width();
        alignas(64) double dxb[kChunk];
        alignas(64) double dyb[kChunk];
        alignas(64) double dzb[kChunk];
        alignas(64) double r2b[kChunk];
        alignas(64) int surv[kChunk];
        alignas(64) double r2s[kChunk + kLanes];
        alignas(64) double ecb[kChunk + kLanes];
        alignas(64) double fcb[kChunk + kLanes];

#pragma omp for schedule(static)
        for (int ii = 0; ii < k_inum; ++ii) {
            const int i = k_ilist[ii];
            const double xi = x_ptr[i], yi = y_ptr[i], zi = z_ptr[i];
            const double qi_scale = qqr2e * q_ptr[i]; // hoist coulomb*qi
            const int ti = type_ptr[i];
            const int jnum = k_numneigh[i];
            const int off = k_firstneigh[i];
            double fi_x = 0, fi_y = 0, fi_z = 0;
            const std::size_t ti_base =
                static_cast<std::size_t>(ti) * static_cast<std::size_t>(ntypes);

            // ~45% of a 2 A-skin list is out of cutoff. Rather than pay a
            // hard-to-predict early exit inside the force loop, geometry is
            // computed unconditionally (pass 1, branch-free and
            // vectorizable), survivors are compacted, and only they reach
            // the LJ/coulomb body.
            for (int base = 0; base < jnum; base += kChunk) {
                const int nc = std::min(kChunk, jnum - base);

                // Halo-resolved axes retain their selected image.
                for (int t = 0; t < nc; ++t) {
                    const int j = k_neighbors[off + base + t];
                    double dx = xi - x_ptr[j];
                    double dy = yi - y_ptr[j];
                    double dz = zi - z_ptr[j];
                    dxb[t] = dx;
                    dyb[t] = dy;
                    dzb[t] = dz;
                    r2b[t] = dx * dx + dy * dy + dz * dz;
                }

                // Branchless append; ascending order is mandatory so pass 2
                // accumulates energy/force/virial in exactly the order the
                // single-pass loop did (bitwise-identical totals).
                int ns = 0;
                for (int t = 0; t < nc; ++t) {
                    const double r2 = r2b[t];
                    surv[ns] = t;
                    r2s[ns] = r2;
                    ns += (r2 >= rc_sq || r2 == 0.0) ? 0 : 1;
                }

                // Pass 1.5: erfc(αr)/r and its force factor for every
                // survivor, analytically, one xsimd batch at a time
                // (cpu/potentials/ewald_erfc.h). Padding lanes get a
                // harmless r² and are never read.
                if (qi_scale != 0.0 && ns > 0) {
                    const int padded = (ns + kLanes - 1) / kLanes * kLanes;
                    for (int k = ns; k < padded; ++k)
                        r2s[k] = 1.0;
                    md::ewald_pair_factors(r2s, ns, alpha, ecb, fcb);
                }

                for (int k = 0; k < ns; ++k) {
                    const int slot = surv[k];
                    const int idx = off + base + slot;
                    const int j = k_neighbors[idx];
                    const double dx = dxb[slot];
                    const double dy = dyb[slot];
                    const double dz = dzb[slot];
                    const double r2 = r2b[slot];

                    double fpair = 0.0;
                    double epair = 0.0;

                    const double sl = k_tier ? t_lj[k_tier[idx]] : 1.0;
                    if (sl != 0.0) {
                        // LAMMPS: factor_lj * (forcelj, evdwl - offset)
                        // inside the pair's own cut_ljsq.
                        const LJPairCoeff& c =
                            tab[ti_base
                                + static_cast<std::size_t>(type_ptr[j])];
                        if (r2 < c.cut_ljsq) {
                            const double r2inv = 1.0 / r2;
                            const double r6inv = r2inv * r2inv * r2inv;
                            fpair += sl * r2inv
                                     * (c.lj1 * r6inv * r6inv - c.lj2 * r6inv);
                            epair += sl
                                     * (c.lj3 * r6inv * r6inv - c.lj4 * r6inv
                                        - c.offset);
                        }
                        if (has_r4 && r2 < c.r4_cutsq) {
                            const R4Term<double> t =
                                r4_attraction(r2, c.r4_a, c.r4_rc5inv, c.r4_e0);
                            fpair += sl * t.fpair;
                            epair += sl * t.energy;
                        }
                    }
                    // LAMMPS special convention with a kspace partner: the
                    // erfc part is UNscaled; scaled/excluded bonded pairs
                    // subtract (1-f)*qq/r — the plain-Coulomb share the
                    // reciprocal sum added for them.
                    if (qi_scale != 0.0 && r2 < coul_sq) {
                        const double qj = q_ptr[j];
                        if (qj != 0.0) {
                            const double qq = qi_scale * qj;
                            double ec = qq * ecb[k];
                            double fc = qq * fcb[k];
                            const double sc =
                                k_tier ? t_coul[k_tier[idx]] : 1.0;
                            if (sc != 1.0) {
                                const double r2i = 1.0 / r2;
                                const double comp =
                                    (1.0 - sc) * qq * std::sqrt(r2i);
                                ec -= comp;
                                fc -= comp * r2i;
                            }
                            fpair += fc;
                            epair += ec;
                        }
                    }
                    if (fpair == 0.0 && epair == 0.0)
                        continue;

                    const double fxp = dx * fpair;
                    const double fyp = dy * fpair;
                    const double fzp = dz * fpair;
                    fi_x += fxp;
                    fi_y += fyp;
                    fi_z += fzp;
                    if (ctx.forces_requested) {
                        fxt[j] -= fxp;
                        fyt[j] -= fyp;
                        fzt[j] -= fzp;
                    }
                    energy_total += epair;
                    if (do_virial) {
                        v_xx += dx * fxp;
                        v_yy += dy * fyp;
                        v_zz += dz * fzp;
                        v_xy += dx * fyp;
                        v_xz += dx * fzp;
                        v_yz += dy * fzp;
                    }
                }
            }
            if (ctx.forces_requested) {
                fxt[i] += fi_x;
                fyt[i] += fi_y;
                fzt[i] += fi_z;
            }
        }

        if (!direct) {
#pragma omp for schedule(static)
            for (int i = 0; i < n_force; ++i) {
                for (int t = 0; t < team_size; ++t) {
                    if (ctx.forces_requested) {
                        fx_ptr[i] += fx_thr_[t][i];
                        fy_ptr[i] += fy_thr_[t][i];
                        fz_ptr[i] += fz_thr_[t][i];
                    }
                }
            }
        }
    }

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (do_virial) {
        if (ctx.stress_requested) {
            f.meta.get(kStress)[0] += v_xx;
            f.meta.get(kStress)[1] += v_yy;
            f.meta.get(kStress)[2] += v_zz;
            f.meta.get(kStress)[3] += v_xy;
            f.meta.get(kStress)[4] += v_xz;
            f.meta.get(kStress)[5] += v_yz;
        }
    }

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
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
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
    for (const auto& v : fx_thr_)
        b += v.capacity() * sizeof(double);
    for (const auto& v : fy_thr_)
        b += v.capacity() * sizeof(double);
    for (const auto& v : fz_thr_)
        b += v.capacity() * sizeof(double);
    b += pair_tier_.capacity();
    b += special_tag_.capacity() * sizeof(TagInt);
    return b;
}

} // namespace cpu
} // namespace ATV
