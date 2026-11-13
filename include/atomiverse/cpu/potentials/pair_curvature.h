/// @file cpu/potentials/pair_curvature.h
/// @brief Second derivatives of a radial pair energy for
///        `CurvatureSource` (configurational temperature), and the pair walk
///        that matches `PairLoop`'s force walk (`CurvatureLoop`).
///
/// For one pair term \(\varphi(r)\), \(r = |r_i - r_j|\),
/// \(\hat r = (r_i - r_j)/r\):
///  - Hessian block \(h = \partial^2\varphi/\partial r_i^2
///    = \tfrac{\varphi'}{r} I + (\varphi'' - \tfrac{\varphi'}{r})\hat r\hat r\),
///    with \(\partial^2\varphi/\partial r_i\partial r_j = -h\), so
///    \((Hu)_i \mathrel{+}= h(u_i - u_j)\), \((Hu)_j \mathrel{-}= h(u_i - u_j)\);
///  - Laplacian \(L = \nabla_i^2\varphi = \nabla_j^2\varphi
///    = \varphi'' + 2\varphi'/r\) on both atoms;
///  - the pair adds \(2L(r)\) to \(\sum_k \nabla_k^2 U\), whose gradient is
///    \(\pm 2L'(r)\hat r\) on i and j.
///
/// A truncated pair potential is treated as zero beyond its cutoff: the
/// \(\delta\)-shell that a force discontinuity at \(r_c\) puts into the
/// Laplacian is not included (the same truncation the forces make).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xsimd/xsimd.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/curvature_protocol.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/device/team.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_loop.h"
#include "atomiverse/cpu/potentials/pair_partition.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace md {

/// @brief Radial derivatives of one pair term at its separation.
struct RadialCurvature {
    double d1 = 0.0;   ///< \(\varphi'(r)\)
    double d2 = 0.0;   ///< \(\varphi''(r)\)
    double lap = 0.0;  ///< \(L = \varphi'' + 2\varphi'/r\)
    double dlap = 0.0; ///< \(L'(r)\)
};

/// @brief Batch scratch of one survivor chunk of a curvature walk. The walk
///        fills `r` and `inv_r` for every survivor; a kernel's `prepare` may
///        fill `a` / `b` (e.g. its exponentials, one SIMD batch at a time).
///        Padded so batches may run past the last survivor (`r` = 1 there).
struct CurvatureChunk {
    alignas(64) double r[kPairChunk + 16];     ///< \(r\) [length]
    alignas(64) double inv_r[kPairChunk + 16]; ///< \(1/r\)
    alignas(64) double a[kPairChunk + 16];
    alignas(64) double b[kPairChunk + 16];
};

/// @brief The pair walk of a radial term's curvature: the pairs
///        `PairLoop::run` walks, accumulated into `CurvatureRows`.
///
/// A style owns one (it caches the row partition of the list it walks and
/// the per-thread accumulators) and supplies the radial derivatives as a
/// kernel, the shape of a `PairLoop` kernel:
///
/// @code{.cpp}
/// struct Kernel {
///     double cutoff_sq() const;                 // outer cutoff² [length²]
///     Row row(int i) const;                     // per-i state hoisted once
///     md::RadialCurvature operator()(const Row& ri, const md::PairSite& p,
///                                    const md::CurvatureChunk& c) const;
///     // optional, once per survivor chunk of row i: partners `j[0..n)`,
///     // `c.r` / `c.inv_r` filled; may fill `c.a` / `c.b`
///     void prepare(const Row& ri, const int* j, int n,
///                  md::CurvatureChunk& c) const;
///     // optional: rows that contribute nothing (e.g. q_i = 0)
///     bool skip(const Row& ri) const;
/// };
/// @endcode
///
/// A pair that does not interact returns a default `RadialCurvature` (all
/// zero). `PairSite::k` indexes the chunk arrays.
///
/// **Threads and determinism.** As the force walk: the rows are cut into one
/// chunk of balanced pair count per thread and the whole team opens once the
/// pairs repay it (`PairRowPartition::team_for`), otherwise the calling
/// thread walks alone. Thread c walks chunk c into its own accumulator (one
/// cache line per row: \(Hu\), the Laplacian, its gradient); the same region
/// then adds the accumulators to `CurvatureRows` row by row in chunk order,
/// so a given thread count gives the same bits every run, and different
/// thread counts differ by the rounding of that sum.
///
/// **Ghost partners.** Under a single-process halo (`Context::md
/// .image_owner`) a ghost partner is an image of an owned row and `u` is the
/// same on both, so the walk reads the owner's `u` and accumulates into the
/// owner: the accumulators span the owned rows only. Without the map they
/// span every row the list names, and the consumer folds the ghost rows.
class CurvatureLoop {
  public:
    /// @brief Add the curvature of every pair to `out`.
    /// @param ctx      Launch context (frame, published list, halo map).
    /// @param out      Accumulators and the vector field `u`.
    /// @param use_list Walk the published half list; otherwise every owned
    ///                 pair at minimum image (single process only).
    /// @param style    Style name for error messages.
    /// @param kernel   Radial derivatives (class comment).
    template <class Kernel>
    void run(const LaunchContext& ctx, const CurvatureRows& out, bool use_list,
             const char* style, const Kernel& kernel);

    /// @brief Forget the cached partition (a style's `build` / `destroy`).
    void reset() noexcept { source_ = nullptr; }

  private:
    /// Doubles per accumulator row: hx hy hz lap gx gy gz and one pad, a
    /// cache line.
    static constexpr int kRow = 8;

    template <class K>
    using row_t_ = decltype(std::declval<const K&>().row(0));
    template <class K>
    static constexpr bool has_prepare_ =
        requires(const K& k, const row_t_<K>& r, const int* j,
                 CurvatureChunk& c) { k.prepare(r, j, 0, c); };
    template <class K>
    static constexpr bool has_skip_ =
        requires(const K& k, const row_t_<K>& r) { bool(k.skip(r)); };

    PairRowPartition partition_;
    const void* source_ = nullptr;
    unsigned long long version_ = 0;
    std::vector<double> acc_; ///< team × (rows × kRow + pad) accumulators
};

template <class Kernel>
void CurvatureLoop::run(const LaunchContext& ctx, const CurvatureRows& out,
                        bool use_list, const char* style,
                        const Kernel& kernel_in) {
    const Frame& f = ctx.frame;
    const PairList* list = nullptr;
    if (use_list) {
        list = ctx.runtime.md.pair_list;
        if (list == nullptr)
            throw std::runtime_error(std::string(style)
                                     + ": curvature needs the published "
                                       "neighbor list its forces walked");
    } else if (multi_rank_dd(ctx.runtime)) {
        throw std::runtime_error(std::string(style)
                                 + ": spatial decomposition requires a "
                                   "half neighbor list");
    }
    const int n = f.nlocal();
    const StorageReal* __restrict x = f.span(kAtom_x).data();
    const StorageReal* __restrict y = f.span(kAtom_y).data();
    const StorageReal* __restrict z = f.span(kAtom_z).data();
    const int* __restrict ilist = list ? list->ilist.data() : nullptr;
    const int* __restrict numneigh = list ? list->numneigh.data() : nullptr;
    const int* __restrict firstneigh = list ? list->firstneigh.data() : nullptr;
    const int* __restrict neighbors = list ? list->neighbors.data() : nullptr;
    const double rc_sq = kernel_in.cutoff_sq();

    // Walked rows and their pair counts: the list's, or rows [0, n) with
    // the n − 1 − i later rows each.
    if (list != nullptr && (source_ != list || version_ != list->version)) {
        partition_.rebuild(list->inum, ilist, numneigh, firstneigh, neighbors);
        source_ = list;
        version_ = list->version;
    }
    const auto before = [n](std::int64_t i) { return i * n - i * (i + 1) / 2; };
    const std::int64_t total = list ? partition_.total() : before(n);
    const int* __restrict owner = nullptr;
    if (list != nullptr && ctx.runtime.md.image_owner != nullptr
        && ctx.runtime.md.image_owner_rows == f.n_total())
        owner = ctx.runtime.md.image_owner;
    const int n_rows = (list == nullptr || owner != nullptr)
                           ? n
                           : std::min(partition_.n_touch(), out.n_total);
    const int team =
        PairRowPartition::team_for(total, n_rows, device::max_team());
    // One accumulator per thread, a cache line apart.
    const std::size_t stride = static_cast<std::size_t>(n_rows) * kRow + kRow;
    if (acc_.size() < stride * static_cast<std::size_t>(team))
        acc_.resize(stride * static_cast<std::size_t>(team));
    double* const acc_all = acc_.data();
    const PairRowPartition& part = partition_;
    // Rows [begin, end) of chunk c out of `chunks`.
    const auto range = [&](int c, int chunks) -> std::pair<int, int> {
        if (list != nullptr)
            return part.range(c, chunks);
        const auto cut = [&](int k) {
            if (k <= 0)
                return 0;
            if (k >= chunks)
                return n;
            const std::int64_t target = total * k / chunks;
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
        return {cut(c), cut(c + 1)};
    };

#pragma omp parallel num_threads(team) if (team > 1)
    {
        // A private copy: the accumulator stores cannot alias its
        // coefficients (as in PairLoop::run).
        const Kernel kernel = kernel_in;
        alignas(64) double dxb[kPairChunk];
        alignas(64) double dyb[kPairChunk];
        alignas(64) double dzb[kPairChunk];
        alignas(64) double r2b[kPairChunk];
        alignas(64) int surv[kPairChunk];
        alignas(64) double r2s[kPairChunk + 16];
        alignas(64) int js[kPairChunk];
        CurvatureChunk chunk;

        int rank = 0, size = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num();
        size = omp_get_num_threads();
#endif
        double* __restrict acc = acc_all + stride * rank;
        std::fill_n(acc, static_cast<std::size_t>(n_rows) * kRow, 0.0);
        const auto [ii_begin, ii_end] = range(rank, size);
        for (int ii = ii_begin; ii < ii_end; ++ii) {
            const int i = list ? ilist[ii] : ii;
            const auto ri = kernel.row(i);
            if constexpr (has_skip_<Kernel>)
                if (kernel.skip(ri))
                    continue;
            const int jnum = list ? numneigh[i] : n - 1 - i;
            const int off = list ? firstneigh[i] : i + 1;
            const double xi = x[i], yi = y[i], zi = z[i];
            const double uix = out.ux[i], uiy = out.uy[i], uiz = out.uz[i];
            double hix = 0, hiy = 0, hiz = 0, lapi = 0;
            double gix = 0, giy = 0, giz = 0;
            for (int base = 0; base < jnum; base += kPairChunk) {
                const int nc = std::min(kPairChunk, jnum - base);
                if (list != nullptr) {
                    // Halo-resolved rows carry their own image.
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
                } else {
                    for (int t = 0; t < nc; ++t) {
                        const int j = off + base + t;
                        double dx = xi - x[j];
                        double dy = yi - y[j];
                        double dz = zi - z[j];
                        f.box.min_image(dx, dy, dz);
                        dxb[t] = dx;
                        dyb[t] = dy;
                        dzb[t] = dz;
                        r2b[t] = dx * dx + dy * dy + dz * dz;
                    }
                }
                // Branchless ascending compaction keeps list order.
                int ns = 0;
                for (int t = 0; t < nc; ++t) {
                    const double r2 = r2b[t];
                    surv[ns] = t;
                    r2s[ns] = r2;
                    ns += (r2 >= rc_sq || r2 == 0.0) ? 0 : 1;
                }
                if (ns == 0)
                    continue;
                // Harmless padding for batch tails; never read back.
                for (int p = ns; p < ns + 16; ++p)
                    r2s[p] = 1.0;
                {
                    using B = xsimd::batch<double>;
                    constexpr int W = static_cast<int>(B::size);
                    for (int k = 0; k < ns; k += W) {
                        const B r = xsimd::sqrt(B::load_unaligned(r2s + k));
                        r.store_aligned(chunk.r + k);
                        (B(1.0) / r).store_aligned(chunk.inv_r + k);
                    }
                }
                for (int k = 0; k < ns; ++k)
                    js[k] = list ? neighbors[off + base + surv[k]]
                                 : off + base + surv[k];
                if constexpr (has_prepare_<Kernel>)
                    kernel.prepare(ri, js, ns, chunk);
                for (int k = 0; k < ns; ++k) {
                    const int slot = surv[k];
                    const int j = js[k];
                    const double dx = dxb[slot];
                    const double dy = dyb[slot];
                    const double dz = dzb[slot];
                    const RadialCurvature c =
                        kernel(ri,
                               PairSite{j, list ? off + base + slot : -1, k,
                                        r2b[slot], dx, dy, dz},
                               chunk);
                    // u is the same on a ghost and its owner.
                    const int jo = owner != nullptr ? owner[j] : j;
                    const double inv_r = chunk.inv_r[k];
                    const double ex = dx * inv_r, ey = dy * inv_r,
                                 ez = dz * inv_r;
                    const double ux = uix - out.ux[jo];
                    const double uy = uiy - out.uy[jo];
                    const double uz = uiz - out.uz[jo];
                    const double radial = c.d1 * inv_r;
                    const double along =
                        (c.d2 - radial) * (ex * ux + ey * uy + ez * uz);
                    const double hx = radial * ux + along * ex;
                    const double hy = radial * uy + along * ey;
                    const double hz = radial * uz + along * ez;
                    const double g = 2.0 * c.dlap;
                    const double gx = g * ex, gy = g * ey, gz = g * ez;
                    double* __restrict aj =
                        acc + static_cast<std::size_t>(jo) * kRow;
                    aj[0] -= hx;
                    aj[1] -= hy;
                    aj[2] -= hz;
                    aj[3] += c.lap;
                    aj[4] -= gx;
                    aj[5] -= gy;
                    aj[6] -= gz;
                    hix += hx;
                    hiy += hy;
                    hiz += hz;
                    lapi += c.lap;
                    gix += gx;
                    giy += gy;
                    giz += gz;
                }
            }
            double* __restrict ai = acc + static_cast<std::size_t>(i) * kRow;
            ai[0] += hix;
            ai[1] += hiy;
            ai[2] += hiz;
            ai[3] += lapi;
            ai[4] += gix;
            ai[5] += giy;
            ai[6] += giz;
        }
        // Every accumulator is complete; each row sums them in chunk order.
#pragma omp barrier
#pragma omp for schedule(static)
        for (int row = 0; row < n_rows; ++row) {
            double s[kRow - 1] = {0, 0, 0, 0, 0, 0, 0};
            for (int c = 0; c < size; ++c) {
                const double* __restrict a =
                    acc_all + stride * c + static_cast<std::size_t>(row) * kRow;
                for (int q = 0; q < kRow - 1; ++q)
                    s[q] += a[q];
            }
            out.hx[row] += s[0];
            out.hy[row] += s[1];
            out.hz[row] += s[2];
            out.lap[row] += s[3];
            out.gx[row] += s[4];
            out.gy[row] += s[5];
            out.gz[row] += s[6];
        }
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
