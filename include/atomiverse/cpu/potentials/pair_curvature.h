/// @file cpu/potentials/pair_curvature.h
/// @brief Second derivatives of a radial pair energy for
///        `CurvatureSource` (configurational temperature), and the pair walk
///        that matches `PairLoop`'s force walk.
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

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/curvature_protocol.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

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

/// @brief Accumulate one pair's Hessian product, Laplacian and Laplacian
///        gradient (file comment) into `rows`.
/// @param dx,dy,dz \(r_i - r_j\); `r` its length.
inline void add_pair_curvature(const CurvatureRows& rows, int i, int j,
                               double dx, double dy, double dz, double r,
                               const RadialCurvature& c) {
    const double inv_r = 1.0 / r;
    const double ex = dx * inv_r, ey = dy * inv_r, ez = dz * inv_r;
    const double ux = rows.ux[i] - rows.ux[j];
    const double uy = rows.uy[i] - rows.uy[j];
    const double uz = rows.uz[i] - rows.uz[j];
    const double radial = c.d1 * inv_r;
    const double along = (c.d2 - radial) * (ex * ux + ey * uy + ez * uz);
    const double hx = radial * ux + along * ex;
    const double hy = radial * uy + along * ey;
    const double hz = radial * uz + along * ez;
    rows.hx[i] += hx;
    rows.hy[i] += hy;
    rows.hz[i] += hz;
    rows.hx[j] -= hx;
    rows.hy[j] -= hy;
    rows.hz[j] -= hz;
    rows.lap[i] += c.lap;
    rows.lap[j] += c.lap;
    const double g = 2.0 * c.dlap;
    rows.gx[i] += g * ex;
    rows.gy[i] += g * ey;
    rows.gz[i] += g * ez;
    rows.gx[j] -= g * ex;
    rows.gy[j] -= g * ey;
    rows.gz[j] -= g * ez;
}

/// @brief Visit the pairs `PairLoop::run` walks: the published half list
///        when `use_list`, otherwise every owned pair at minimum image
///        (single process only).
/// @param visit Called as `visit(i, j, entry, dx, dy, dz, r2)` for pairs
///        inside `rc_sq` (entry: CSR slot, −1 in the all-pairs walk).
template <class Visit>
void for_each_curvature_pair(const LaunchContext& ctx, double rc_sq,
                             bool use_list, const char* style, Visit&& visit) {
    const Frame& f = ctx.frame;
    const StorageReal* x = f.span(kAtom_x).data();
    const StorageReal* y = f.span(kAtom_y).data();
    const StorageReal* z = f.span(kAtom_z).data();
    if (use_list) {
        const PairList* pl = ctx.runtime.md.pair_list;
        if (pl == nullptr)
            throw std::runtime_error(std::string(style)
                                     + ": curvature needs the published "
                                       "neighbor list its forces walked");
        for (int ii = 0; ii < pl->inum; ++ii) {
            const int i = pl->ilist[ii];
            const int off = pl->firstneigh[i];
            const int nn = pl->numneigh[i];
            for (int jj = 0; jj < nn; ++jj) {
                const int j = pl->neighbors[off + jj];
                const double dx = x[i] - x[j];
                const double dy = y[i] - y[j];
                const double dz = z[i] - z[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq || r2 == 0.0)
                    continue;
                visit(i, j, off + jj, dx, dy, dz, r2);
            }
        }
        return;
    }
    if (multi_rank_dd(ctx.runtime))
        throw std::runtime_error(std::string(style)
                                 + ": spatial decomposition requires a "
                                   "half neighbor list");
    const int n = f.nlocal();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc_sq || r2 == 0.0)
                continue;
            visit(i, j, -1, dx, dy, dz, r2);
        }
}

} // namespace md
} // namespace cpu
} // namespace ATV
