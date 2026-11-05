/// @file cpu/potentials/pair_loop.h
/// @brief `PairLoop` — the one CPU half-list pair traversal.
///
/// Every CPU pair style used to carry its own walk: CSR traversal, cutoff
/// test, Newton-third-law scatter, threaded force buffers, virial and
/// energy sums — and several were serial, with an O(N²) fallback when no
/// list was published. This header holds that walk once; a style owns a
/// `PairLoop` (it caches the row partition of the list it walks) and supplies
/// only the per-pair physics as a kernel:
///
/// @code{.cpp}
/// struct Kernel {
///     double cutoff_sq() const;                 // outer cutoff² [length²]
///     Row row(int i) const;                     // per-i state hoisted once
///     md::PairTerm operator()(const Row& ri, const md::PairSite& p,
///                             const md::PairChunk& chunk) const;
///     // optional, once per survivor chunk of row i:
///     void prepare(const Row& ri, const double* r2, int n,
///                  md::PairChunk& chunk) const;
///     // optional: rows that contribute nothing (e.g. q_i = 0)
///     bool skip(const Row& ri) const;
///     // optional: walk rows [ii_begin, ii_end) of `rows` itself (a SIMD
///     // ket loop) instead of the generic survivor walk; adds their forces
///     // to `f` (i and every partner) and energy / virial to `acc`.
///     void walk(const md::PairRows& rows, int ii_begin, int ii_end,
///               md::ForceRows f, md::PairAcc& acc, bool do_force,
///               bool do_virial) const;
/// };
/// @endcode
///
/// A kernel returns `PairTerm`, or `PairTermX` when it also carries a style
/// extra (summed into `PairTotals::extra`, e.g. a soft-core dU/dλ) and a
/// virial force factor that differs from the force (DPD's thermostat forces
/// are not part of its virial).
///
/// `PairSite` carries the partner row `j`, the CSR slot `entry` of the pair
/// (for per-pair side tables such as special-bond tiers; −1 in the all-pairs
/// walk), the survivor index `k` within the current chunk (so `prepare` can
/// fill `chunk.a[k]` / `chunk.b[k]` with batched factors, e.g. the Ewald
/// erfc pair, `ewald_erfc.h`), `r2` (strictly inside `cutoff_sq()`, non-zero)
/// and the separation `r_i − r_j`.
///
/// The walk (LAMMPS `PairLJCutCoulLongOMP::eval`, Newton on): geometry for a
/// chunk of 512 neighbours is computed branch-free, survivors are compacted
/// in ascending order (totals accumulate in list order), only they reach the
/// kernel. The i-force is stored once per row; the virial is summed only when
/// stress is requested. Without a published half list (single process only)
/// the same kernel runs over all owned pairs with minimum-image separations.
///
/// **Threads and determinism.** The rows are cut into chunks of balanced pair
/// count (`PairRowPartition`, rebuilt when the walked list changes); the team
/// follows the pair count, so a small system runs the single-thread direct
/// path. Chunks are handed out dynamically, but every chunk scatters into its
/// own `md::ForcePass` slot and stores its own energy / virial partials, all
/// combined in chunk order: for a given thread count the result is
/// bit-identical run to run, whichever thread ran which chunk.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_partition.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief One pair's contribution: force on i is `fpair·(r_i − r_j)`.
struct PairTerm {
    double fpair = 0.0;
    double energy = 0.0;
};

/// @brief `PairTerm` plus a style extra and the force factor the virial
///        takes (`fvirial`, in place of `fpair`).
struct PairTermX {
    double fpair = 0.0;
    double energy = 0.0;
    double extra = 0.0;
    double fvirial = 0.0;
};

/// @brief One surviving pair as the kernel sees it.
struct PairSite {
    int j;     ///< partner row
    int entry; ///< CSR slot of the pair in the walked list (−1: all pairs)
    int k;     ///< survivor index within the current chunk
    double r2; ///< |r_i − r_j|², inside the cutoff and non-zero
    double dx, dy, dz; ///< r_i − r_j
};

/// @brief Survivors per chunk; a chunk and its scratch stay L1-resident.
inline constexpr int kPairChunk = 512;

/// @brief Per-chunk batch scratch a kernel's `prepare` may fill; padded so
///        SIMD batches may run past the last survivor.
struct PairChunk {
    alignas(64) double a[kPairChunk + 16];
    alignas(64) double b[kPairChunk + 16];
};

/// @brief Energy and virial one `PairLoop::run` accumulated (forces went
///        to the frame / the shared chunk buffers).
struct PairTotals {
    double energy = 0.0;
    Voigt6 virial;
    double extra = 0.0;
};

/// @brief A chunk's running energy and virial, for kernels that `walk`
///        their rows themselves.
struct PairAcc {
    double energy = 0.0;
    Voigt6 virial;
};

/// @brief A half list as `PairLoop` walks it: the published `PairList` or a
///        style's filtered copy of it. `source` + `version` identify it for
///        the cached row partition.
struct PairRows {
    int inum;
    const int* ilist;
    const int* numneigh;
    const int* firstneigh;
    const int* neighbors;
    const void* source;
    unsigned long long version;

    static PairRows of(const PairList& pl) {
        return {pl.inum,
                pl.ilist.data(),
                pl.numneigh.data(),
                pl.firstneigh.data(),
                pl.neighbors.data(),
                &pl,
                pl.version};
    }
};

/// @brief Kernel for a radial pair potential that depends on r² alone:
///        `term(r2)` returns the `PairTerm` (use with CTAD,
///        `md::RadialKernel{cutoff_sq, [&](double r2) { ... }}`).
template <class Term>
struct RadialKernel {
    double rc_sq;
    Term term;

    double cutoff_sq() const { return rc_sq; }
    int row(int) const { return 0; }
    PairTerm operator()(int, const PairSite& p, const PairChunk&) const {
        return term(p.r2);
    }
};

class PairLoop {
  public:
    /// @brief Walk every pair of the published half list once (or, without
    ///        one, every owned pair) and apply `kernel`.
    /// @param ctx Launch context; `forces_requested` / `stress_requested`
    ///        gate the scatter and the virial.
    /// @param kernel Per-pair physics.
    /// @param style Style name for error messages.
    template <class Kernel>
    PairTotals run(const LaunchContext& ctx, const Kernel& kernel,
                   const char* style) {
        if (const PairList* pairs = ctx.runtime.md.pair_list)
            return run(ctx, PairRows::of(*pairs), kernel);
        if (multi_rank_dd(ctx.runtime))
            throw std::runtime_error(std::string(style)
                                     + ": spatial decomposition requires a "
                                       "half neighbor list");
        return all_pairs_(ctx, kernel);
    }

    /// @brief Walk every owned pair once with minimum-image separations,
    ///        whether or not a list is published (single process only).
    template <class Kernel>
    PairTotals all_pairs(const LaunchContext& ctx, const Kernel& kernel,
                         const char* style) {
        if (multi_rank_dd(ctx.runtime))
            throw std::runtime_error(std::string(style)
                                     + ": spatial decomposition requires a "
                                       "half neighbor list");
        return all_pairs_(ctx, kernel);
    }

    /// @brief Walk `rows` (the published list or a style's filtered copy).
    template <class Kernel>
    PairTotals run(const LaunchContext& ctx, const PairRows& rows,
                   const Kernel& kernel);

    /// @brief Forget the cached partition (a style's `build` / `destroy`).
    void reset() noexcept { source_ = nullptr; }

  private:
    template <class K>
    static constexpr bool has_prepare_ =
        requires(const K& k, const decltype(k.row(0))& r, const double* p,
                 PairChunk& c) { k.prepare(r, p, 0, c); };
    template <class K>
    static constexpr bool has_skip_ =
        requires(const K& k, const decltype(k.row(0))& r) { bool(k.skip(r)); };
    template <class K>
    static constexpr bool has_walk_ =
        requires(const K& k, const PairRows& L, ForceRows f, PairAcc& a) {
            k.walk(L, 0, 0, f, a, true, true);
        };
    template <class K>
    static constexpr bool extended_ = std::is_same_v<
        decltype(std::declval<const K&>()(
            std::declval<const decltype(std::declval<const K&>().row(0))&>(),
            std::declval<const PairSite&>(), std::declval<const PairChunk&>())),
        PairTermX>;

    template <class Kernel>
    PairTotals all_pairs_(const LaunchContext& ctx, const Kernel& kernel);

    PairRowPartition partition_;
    const void* source_ = nullptr;
    unsigned long long version_ = 0;
};

template <class Kernel>
PairTotals PairLoop::run(const LaunchContext& ctx, const PairRows& L,
                         const Kernel& kernel_in) {
    Frame& f = ctx.frame;
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const int* __restrict ilist = L.ilist;
    const int* __restrict numneigh = L.numneigh;
    const int* __restrict firstneigh = L.firstneigh;
    const int* __restrict neighbors = L.neighbors;
    const double rc_sq = kernel_in.cutoff_sq();
    const bool do_force = ctx.forces_requested;
    const bool do_virial = ctx.stress_requested;

    if (source_ != L.source || version_ != L.version) {
        partition_.rebuild(L.inum, ilist, numneigh, firstneigh, neighbors);
        source_ = L.source;
        version_ = L.version;
    }
    ForcePass forces(ctx);
    const int team = partition_.team(forces.team());
    const int n_chunks = partition_.chunks(team);
    // Rows past the list's highest reference receive nothing.
    forces.open(team, n_chunks,
                std::min(partition_.n_touch(), force_count(ctx)));
    const PairRowPartition& part = partition_;

#pragma omp parallel num_threads(team) if (team > 1)
    {
        // A private copy: force stores cannot alias its coefficients, so
        // they stay in registers across the scatter.
        const Kernel kernel = kernel_in;
        alignas(64) double dxb[kPairChunk];
        alignas(64) double dyb[kPairChunk];
        alignas(64) double dzb[kPairChunk];
        alignas(64) double r2b[kPairChunk];
        alignas(64) int surv[kPairChunk];
        alignas(64) double r2s[kPairChunk + 16];
        PairChunk chunk;

#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_chunks; ++c) {
            const ForceRows tf = forces.rows(c);
            double energy = 0.0, extra = 0.0;
            double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
            const auto [ii_begin, ii_end] = part.range(c, n_chunks);
            if constexpr (has_walk_<Kernel>) {
                PairAcc acc;
                kernel.walk(L, ii_begin, ii_end, tf, acc, do_force, do_virial);
                forces.store(c, acc.energy, acc.virial);
                continue;
            }
            for (int ii = ii_begin; ii < ii_end; ++ii) {
                const int i = ilist[ii];
                const auto ri = kernel.row(i);
                if constexpr (has_skip_<Kernel>)
                    if (kernel.skip(ri))
                        continue;
                const int jnum = numneigh[i];
                const int off = firstneigh[i];
                const double xi = x[i], yi = y[i], zi = z[i];
                double fi_x = 0, fi_y = 0, fi_z = 0;
                for (int base = 0; base < jnum; base += kPairChunk) {
                    const int nc = std::min(kPairChunk, jnum - base);
                    // Halo-resolved rows carry their own image: plain
                    // differences, no minimum image.
                    for (int t = 0; t < nc; ++t) {
                        const int j = neighbors[off + base + t];
                        const double dx = xi - x[j];
                        const double dy = yi - y[j];
                        const double dz = zi - z[j];
                        dxb[t] = dx;
                        dyb[t] = dy;
                        dzb[t] = dz;
                        r2b[t] = dx * dx + dy * dy + dz * dz;
                    }
                    // Branchless ascending compaction keeps list order.
                    int ns = 0;
                    for (int t = 0; t < nc; ++t) {
                        const double r2 = r2b[t];
                        surv[ns] = t;
                        r2s[ns] = r2;
                        ns += (r2 >= rc_sq || r2 == 0.0) ? 0 : 1;
                    }
                    if constexpr (has_prepare_<Kernel>) {
                        // Harmless padding for batch tails; never read back.
                        for (int p = ns; p < ns + 16; ++p)
                            r2s[p] = 1.0;
                        if (ns > 0)
                            kernel.prepare(ri, r2s, ns, chunk);
                    }
                    for (int k = 0; k < ns; ++k) {
                        const int slot = surv[k];
                        const int entry = off + base + slot;
                        const int j = neighbors[entry];
                        const double dx = dxb[slot];
                        const double dy = dyb[slot];
                        const double dz = dzb[slot];
                        const auto term = kernel(
                            ri, PairSite{j, entry, k, r2b[slot], dx, dy, dz},
                            chunk);
                        if constexpr (extended_<Kernel>) {
                            if (term.fpair == 0.0 && term.energy == 0.0
                                && term.extra == 0.0 && term.fvirial == 0.0)
                                continue;
                            extra += term.extra;
                        } else {
                            if (term.fpair == 0.0 && term.energy == 0.0)
                                continue;
                        }
                        const double fxp = dx * term.fpair;
                        const double fyp = dy * term.fpair;
                        const double fzp = dz * term.fpair;
                        fi_x += fxp;
                        fi_y += fyp;
                        fi_z += fzp;
                        if (do_force) {
                            tf.fx[j] -= fxp;
                            tf.fy[j] -= fyp;
                            tf.fz[j] -= fzp;
                        }
                        energy += term.energy;
                        if (do_virial) {
                            double vx = fxp, vy = fyp, vz = fzp;
                            if constexpr (extended_<Kernel>) {
                                vx = dx * term.fvirial;
                                vy = dy * term.fvirial;
                                vz = dz * term.fvirial;
                            }
                            v_xx += dx * vx;
                            v_yy += dy * vy;
                            v_zz += dz * vz;
                            v_xy += dx * vy;
                            v_xz += dx * vz;
                            v_yz += dy * vz;
                        }
                    }
                }
                if (do_force) {
                    tf.fx[i] += fi_x;
                    tf.fy[i] += fi_y;
                    tf.fz[i] += fi_z;
                }
            }
            forces.store(c, energy,
                         Voigt6{{v_xx, v_yy, v_zz, v_xy, v_xz, v_yz}}, extra);
        }
    }
    return {forces.energy(), forces.virial(), forces.extra()};
}

/// No list (single process): every owned pair once, minimum image. Rows are
/// cut so every chunk holds about the same number of the n(n−1)/2 pairs.
template <class Kernel>
PairTotals PairLoop::all_pairs_(const LaunchContext& ctx,
                                const Kernel& kernel_in) {
    Frame& f = ctx.frame;
    const int n = f.nlocal();
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const double rc_sq = kernel_in.cutoff_sq();
    const bool do_force = ctx.forces_requested;
    const bool do_virial = ctx.stress_requested;

    // Pairs of rows [0, i): i·n − i(i+1)/2.
    const auto before = [n](std::int64_t i) { return i * n - i * (i + 1) / 2; };
    const std::int64_t total = before(n);
    ForcePass forces(ctx);
    const int team = static_cast<int>(std::clamp<std::int64_t>(
        total / PairRowPartition::kMinPairsPerThread, 1, forces.team()));
    const int n_chunks =
        team > 1 ? team * PairRowPartition::kChunksPerThread : 1;
    forces.open(team, n_chunks, n);
    const auto cut = [&](int c) {
        if (c <= 0)
            return 0;
        if (c >= n_chunks)
            return n;
        const std::int64_t target = total * c / n_chunks;
        int lo = 0, hi = n;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (before(mid) < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    };

#pragma omp parallel num_threads(team) if (team > 1)
    {
        const Kernel kernel = kernel_in; // private copy, as in run()
        PairChunk chunk;
        double r2one[16];
        std::fill(r2one, r2one + 16, 1.0);
#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_chunks; ++c) {
            const ForceRows tf = forces.rows(c);
            double energy = 0.0, extra = 0.0;
            Voigt6 vir;
            const int i_end = cut(c + 1);
            for (int i = cut(c); i < i_end; ++i) {
                const auto ri = kernel.row(i);
                if constexpr (has_skip_<Kernel>)
                    if (kernel.skip(ri))
                        continue;
                for (int j = i + 1; j < n; ++j) {
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double dz = z[i] - z[j];
                    f.box.min_image(dx, dy, dz);
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc_sq || r2 == 0.0)
                        continue;
                    if constexpr (has_prepare_<Kernel>) {
                        r2one[0] = r2;
                        kernel.prepare(ri, r2one, 1, chunk);
                    }
                    const auto term =
                        kernel(ri, PairSite{j, -1, 0, r2, dx, dy, dz}, chunk);
                    const double fxp = dx * term.fpair;
                    const double fyp = dy * term.fpair;
                    const double fzp = dz * term.fpair;
                    if (do_force) {
                        tf.fx[i] += fxp;
                        tf.fy[i] += fyp;
                        tf.fz[i] += fzp;
                        tf.fx[j] -= fxp;
                        tf.fy[j] -= fyp;
                        tf.fz[j] -= fzp;
                    }
                    energy += term.energy;
                    if constexpr (extended_<Kernel>) {
                        extra += term.extra;
                        if (do_virial)
                            vir.add_pair(dx, dy, dz, dx * term.fvirial,
                                         dy * term.fvirial, dz * term.fvirial);
                    } else {
                        if (do_virial)
                            vir.add_pair(dx, dy, dz, fxp, fyp, fzp);
                    }
                }
            }
            forces.store(c, energy, vir, extra);
        }
    }
    return {forces.energy(), forces.virial(), forces.extra()};
}

} // namespace md
} // namespace cpu
} // namespace ATV
