/// @file cpu/gpw/internal/product_collint.h
/// @brief CP2K collint-style product collocate / integrate on UniformGrid.
///
/// Algorithm after CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3-Clause,
/// Ole Schuett et al.): expand (r−ra)^{la}(r−rb)^{lb} into powers of (r−rp),
/// form cxyz, precompute pol[dir][l][i] = (r−rp)^l exp(−p(r−rp)²) with exp
/// recurrence, contract cxyz→grid (collocate) or the transpose (integrate).
///
/// Geometry batching (same p, ra, rb, pc): many cartesian products from one
/// shell pair share pol; collocate sums their weighted cxyz then contracts
/// once; integrate builds a monom table once and contracts each product's
/// alphas. Portable xsimd on the z-fast axis — no AVX2 intrinsics.
///
/// Callers own product-list / P-scatter in multigrid.cpp.
#pragma once

#include "atomiverse/cpu/gpw/internal/multigrid.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

/// @brief Collocate one product: ρ += weight · exp · (r−ra)^{la}(r−rb)^{lb}.
/// @param weight Density weight (includes P and coeffs; prod.k already applied).
/// @param mag    Window magnitude (screening; typically |weight|).
void collint_collocate_product(const UniformGrid& grid, const GridProduct& prod,
                               double weight, double mag, double eps,
                               double* rho);

/// @brief Integrate one product shape (no weight / k factor).
/// @return ∫ V(r)·shape(r) dr / grid.weight()  (caller multiplies by k·w_cell).
double collint_integrate_product(const UniformGrid& grid,
                                 const GridProduct& prod, double mag,
                                 double eps, const double* V);

/// @brief Collocate a geometry-matched product group: one pol, summed cxyz.
/// @param prods   Contiguous products sharing p, ra, rb, pc (la/lb may differ).
/// @param weights Per-product density weight (length n); 0-skips allowed.
/// @param n       Group size.
/// @param mag     Window magnitude for the whole group (e.g. max|w|).
void collint_collocate_group(const UniformGrid& grid, const GridProduct* prods,
                             const double* weights, int n, double mag,
                             double eps, double* rho);

/// @brief Same as collint_collocate_group but products selected by index list
///        (avoids copying GridProduct into a temp buffer).
void collint_collocate_group_idx(const UniformGrid& grid,
                                const GridProduct* prods_base,
                                const int* idx, const double* weights, int n,
                                double mag, double eps, double* rho);

/// @brief Integrate a geometry-matched product group: one monom table.
/// @param out_acc Per-product ∫ V·shape (length n); zero when screened.
void collint_integrate_group(const UniformGrid& grid, const GridProduct* prods,
                             int n, double mag, double eps, const double* V,
                             double* out_acc);

/// @brief True if a and b share collint geometry (same pol table).
inline bool collint_same_geometry(const GridProduct& a, const GridProduct& b) {
    return a.p == b.p && a.pc[0] == b.pc[0] && a.pc[1] == b.pc[1]
           && a.pc[2] == b.pc[2] && a.ra[0] == b.ra[0] && a.ra[1] == b.ra[1]
           && a.ra[2] == b.ra[2] && a.rb[0] == b.rb[0] && a.rb[1] == b.rb[1]
           && a.rb[2] == b.rb[2];
}

/// @brief Stable total order for geometry batching (build_products sorts by this).
inline bool collint_geom_less(const GridProduct& a, const GridProduct& b) {
    if (a.p != b.p)
        return a.p < b.p;
    for (int d = 0; d < 3; ++d)
        if (a.pc[d] != b.pc[d])
            return a.pc[d] < b.pc[d];
    for (int d = 0; d < 3; ++d)
        if (a.ra[d] != b.ra[d])
            return a.ra[d] < b.ra[d];
    for (int d = 0; d < 3; ++d)
        if (a.rb[d] != b.rb[d])
            return a.rb[d] < b.rb[d];
    return false;
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
