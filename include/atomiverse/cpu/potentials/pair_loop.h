/// @file cpu/potentials/pair_loop.h
/// @brief `PairLoop<Kernel>` — the one CPU half-list pair traversal.
///
/// Every CPU pair style used to carry its own walk: CSR traversal, cutoff
/// test, Newton-third-law scatter, per-thread force buffers, virial and
/// energy sums — and several were serial, with an O(N²) fallback when no
/// list was published. This header holds that walk once; a style supplies
/// only the per-pair physics as a kernel:
///
/// @code{.cpp}
/// struct Kernel {
///     double cutoff_sq() const;                 // outer cutoff² [length²]
///     Row row(int i) const;                     // per-i state hoisted once
///     md::PairTerm operator()(const Row& ri, int j, int entry, double r2,
///                             const md::PairChunk& chunk, int k) const;
///     // optional, once per survivor chunk of row i:
///     void prepare(const Row& ri, const double* r2, int n,
///                  md::PairChunk& chunk) const;
/// };
/// @endcode
///
/// `entry` is the CSR slot of the pair (for per-pair side tables such as
/// special-bond tiers); `k` indexes the survivor within the current chunk,
/// so `prepare` can fill `chunk.a[k]` / `chunk.b[k]` with batched factors
/// (e.g. the Ewald erfc pair, `ewald_erfc.h`). `r2` is strictly inside
/// `cutoff_sq()` and non-zero.
///
/// The walk (LAMMPS `PairLJCutCoulLongOMP::eval`, Newton on): geometry for a
/// chunk of 512 neighbours is computed branch-free, survivors are compacted
/// in ascending order (totals accumulate in list order), only they reach the
/// kernel. Forces scatter through `md::ForcePass` (the Runtime's shared
/// per-thread buffers), the i-force is stored once per row, the virial is
/// summed only when stress is requested. Rows are split statically across
/// the team. Without a published half list (single process only) the same
/// kernel runs over all owned pairs with minimum-image displacements.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

/// @brief One pair's contribution: force on i is `fpair·(r_i − r_j)`.
struct PairTerm {
    double fpair = 0.0;
    double energy = 0.0;
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
///        to the frame / the shared thread buffers).
struct PairTotals {
    double energy = 0.0;
    Voigt6 virial;
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
    PairTerm operator()(int, int, int, double r2, const PairChunk&, int) const {
        return term(r2);
    }
};

template <class Kernel>
class PairLoop {
  public:
    /// @brief Walk every pair once and apply `kernel`.
    /// @param ctx Launch context; `forces_requested` / `stress_requested`
    ///        gate the scatter and the virial.
    /// @param kernel Per-pair physics.
    /// @param style Style name for error messages.
    static PairTotals run(const LaunchContext& ctx, const Kernel& kernel,
                          const char* style) {
        if (const PairList* pairs = ctx.runtime.md.pair_list)
            return listed_(ctx, *pairs, kernel);
        if (multi_rank_dd(ctx.runtime))
            throw std::runtime_error(std::string(style)
                                     + ": spatial decomposition requires a "
                                       "half neighbor list");
        return all_pairs_(ctx, kernel);
    }

  private:
    template <class K>
    static constexpr bool has_prepare_ =
        requires(const K& k, const decltype(k.row(0))& r, const double* p,
                 PairChunk& c) { k.prepare(r, p, 0, c); };

    static PairTotals listed_(const LaunchContext& ctx, const PairList& pl,
                              const Kernel& kernel) {
        Frame& f = ctx.frame;
        const StorageReal* __restrict x = f.span(kAtom_x);
        const StorageReal* __restrict y = f.span(kAtom_y);
        const StorageReal* __restrict z = f.span(kAtom_z);
        const int inum = pl.inum;
        const int* __restrict ilist = pl.ilist.data();
        const int* __restrict numneigh = pl.numneigh.data();
        const int* __restrict firstneigh = pl.firstneigh.data();
        const int* __restrict neighbors = pl.neighbors.data();
        const double rc_sq = kernel.cutoff_sq();
        const bool do_force = ctx.forces_requested;
        const bool do_virial = ctx.stress_requested;

        const ForcePass forces(ctx);
        double energy = 0.0;
        double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
#pragma omp parallel num_threads(forces.team()) if (forces.team() > 1)         \
    reduction(+ : energy, v_xx, v_yy, v_zz, v_xy, v_xz, v_yz)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            const ThreadForce tf = forces.thread(tid);
            alignas(64) double dxb[kPairChunk];
            alignas(64) double dyb[kPairChunk];
            alignas(64) double dzb[kPairChunk];
            alignas(64) double r2b[kPairChunk];
            alignas(64) int surv[kPairChunk];
            alignas(64) double r2s[kPairChunk + 16];
            PairChunk chunk;

#pragma omp for schedule(static)
            for (int ii = 0; ii < inum; ++ii) {
                const int i = ilist[ii];
                const auto ri = kernel.row(i);
                const double xi = x[i], yi = y[i], zi = z[i];
                const int jnum = numneigh[i];
                const int off = firstneigh[i];
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
                        const PairTerm term =
                            kernel(ri, j, entry, r2b[slot], chunk, k);
                        if (term.fpair == 0.0 && term.energy == 0.0)
                            continue;
                        const double dx = dxb[slot];
                        const double dy = dyb[slot];
                        const double dz = dzb[slot];
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
                            v_xx += dx * fxp;
                            v_yy += dy * fyp;
                            v_zz += dz * fzp;
                            v_xy += dx * fyp;
                            v_xz += dx * fzp;
                            v_yz += dy * fzp;
                        }
                    }
                }
                if (do_force) {
                    tf.fx[i] += fi_x;
                    tf.fy[i] += fi_y;
                    tf.fz[i] += fi_z;
                }
            }
        }
        return totals_(energy, v_xx, v_yy, v_zz, v_xy, v_xz, v_yz);
    }

    /// No list (single process): every owned pair once, minimum image.
    static PairTotals all_pairs_(const LaunchContext& ctx,
                                 const Kernel& kernel) {
        Frame& f = ctx.frame;
        const int n = f.nlocal();
        const StorageReal* __restrict x = f.span(kAtom_x);
        const StorageReal* __restrict y = f.span(kAtom_y);
        const StorageReal* __restrict z = f.span(kAtom_z);
        const double rc_sq = kernel.cutoff_sq();
        const bool do_force = ctx.forces_requested;
        const bool do_virial = ctx.stress_requested;
        const ForcePass forces(ctx);
        double energy = 0.0;
        double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
#pragma omp parallel num_threads(forces.team()) if (forces.team() > 1)         \
    reduction(+ : energy, v_xx, v_yy, v_zz, v_xy, v_xz, v_yz)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            const ThreadForce tf = forces.thread(tid);
            PairChunk chunk;
            double r2one[16];
            std::fill(r2one, r2one + 16, 1.0);
#pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < n; ++i) {
                const auto ri = kernel.row(i);
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
                    const PairTerm term = kernel(ri, j, -1, r2, chunk, 0);
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
        }
        return totals_(energy, v_xx, v_yy, v_zz, v_xy, v_xz, v_yz);
    }

    static PairTotals totals_(double e, double xx, double yy, double zz,
                              double xy, double xz, double yz) {
        PairTotals t;
        t.energy = e;
        t.virial.v[0] = xx;
        t.virial.v[1] = yy;
        t.virial.v[2] = zz;
        t.virial.v[3] = xy;
        t.virial.v[4] = xz;
        t.virial.v[5] = yz;
        return t;
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
