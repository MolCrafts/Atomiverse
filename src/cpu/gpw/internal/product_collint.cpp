/// @file src/cpu/gpw/internal/product_collint.cpp
/// @brief CP2K ortho collint product collocate/integrate (see product_collint.h).
///
/// Near-literal port of CP2K `src/grid/cpu/grid_cpu_collint.h` ortho path
/// (BSD-3-Clause, Ole Schuett et al.) plus:
///   - process-wide sphere-bounds cache (keyed by dh + discretized radius)
///   - lp_z-specialized inner contract (CP2K GRID_MAX_LP_OPTIMIZED style)
///   - portable xsimd on contiguous z-runs
///
/// Axis map onto ATV z-fast layout `g[(ix*ny+iy)*nz+iz]`:
///   CP2K i (fast) ↔ ATV z,  j ↔ y,  k ↔ x.

#include "atomiverse/cpu/gpw/internal/product_collint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {
namespace {

constexpr int kMaxL = 12;
constexpr int kMaxCmax = 128;
constexpr int kMaxLpOpt = 6; // specialize inner loops for lp_z ≤ this

// Grow-only thread_local scratch.
thread_local std::vector<double> g_buf;
thread_local std::vector<int> g_imap;
thread_local std::vector<double> g_I;

inline void ensure_buf(std::size_t n) {
    if (g_buf.size() < n)
        g_buf.resize(n);
}
inline void ensure_imap(std::size_t n) {
    if (g_imap.size() < n)
        g_imap.resize(n);
}
inline void ensure_I(std::size_t n) {
    if (g_I.size() < n)
        g_I.resize(n);
}

inline int imax(int a, int b) { return a > b ? a : b; }
inline int imin(int a, int b) { return a < b ? a : b; }
inline int modulo(int a, int n) { return ((a % n) + n) % n; }

inline int binom(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n - k)
        k = n - k;
    int r = 1;
    for (int i = 0; i < k; ++i)
        r = r * (n - i) / (i + 1);
    return r;
}

inline int expand_axis(int la, int lb, double pc, double ra, double rb,
                       double alpha[kMaxL + 1]) {
    const int lp = la + lb;
    for (int i = 0; i <= lp; ++i)
        alpha[i] = 0.0;
    const double da = pc - ra;
    const double db = pc - rb;
    double da_pow[kMaxL + 1], db_pow[kMaxL + 1];
    da_pow[0] = 1.0;
    for (int i = 1; i <= la; ++i)
        da_pow[i] = da_pow[i - 1] * da;
    db_pow[0] = 1.0;
    for (int i = 1; i <= lb; ++i)
        db_pow[i] = db_pow[i - 1] * db;
    for (int k = 0; k <= la; ++k)
        for (int m = 0; m <= lb; ++m)
            alpha[k + m] += static_cast<double>(binom(la, k) * binom(lb, m))
                            * da_pow[la - k] * db_pow[lb - m];
    return lp;
}

inline double shape_radius(double mag, double p, int Ltot, double eps) {
    if (!(mag > 0.0) || !(p > 0.0))
        return 0.0;
    const double lc = std::log(std::max(mag, 1e-300));
    const double leps = std::log(std::max(eps, 1e-300));
    double r2 = std::max(0.0, (lc - leps) / p);
    for (int it = 0; it < 4; ++it) {
        const double r = std::sqrt(std::max(r2, 1e-30));
        r2 = std::max(0.0, (lc + Ltot * std::log(r) - leps) / p);
    }
    return std::sqrt(std::max(r2, 0.0));
}

// ─── Sphere-bounds cache (CP2K grid_sphere_cache) ───────────────────────────
// One entry per grid spacing (di,dj,dk). Storage holds bounds for imr=1..max.
struct SphereEntry {
    double di = 0, dj = 0, dk = 0, drmin = 0;
    int max_imr = 0;
    std::vector<int> offsets; // size max_imr
    std::vector<int> storage;
};

struct SphereCache {
    std::mutex mu;
    std::vector<SphereEntry> entries;
    int prev = -1;
};

SphereCache& sphere_cache() {
    static SphereCache c;
    return c;
}

int single_sphere_bounds(double disr_radius, double di, double dj, double dk,
                         int* bounds) {
    const double di_inv = 1.0 / di, dj_inv = 1.0 / dj, dk_inv = 1.0 / dk;
    int ibound = 0;
    const int kgmin =
        static_cast<int>(std::ceil(-1e-8 - disr_radius * dk_inv));
    if (bounds)
        bounds[ibound] = kgmin;
    ++ibound;
    for (int kg = kgmin; kg <= 0; ++kg) {
        const int kd = (2 * kg - 1) / 2;
        const double kr = kd * dk;
        const double kremain = disr_radius * disr_radius - kr * kr;
        const int jgmin = static_cast<int>(
            std::ceil(-1e-8 - std::sqrt(std::max(0.0, kremain)) * dj_inv));
        if (bounds)
            bounds[ibound] = jgmin;
        ++ibound;
        for (int jg = jgmin; jg <= 0; ++jg) {
            const int jd = (2 * jg - 1) / 2;
            const double jr = jd * dj;
            const double jremain = kremain - jr * jr;
            const int igmin = static_cast<int>(
                std::ceil(-1e-8 - std::sqrt(std::max(0.0, jremain)) * di_inv));
            if (bounds)
                bounds[ibound] = igmin;
            ++ibound;
        }
    }
    return ibound;
}

void rebuild_sphere_entry(SphereEntry& e, int max_imr) {
    e.max_imr = max_imr;
    e.offsets.assign(static_cast<std::size_t>(max_imr), 0);
    int total = 0;
    for (int imr = 1; imr <= max_imr; ++imr) {
        e.offsets[static_cast<std::size_t>(imr - 1)] = total;
        const double rad = e.drmin * static_cast<double>(imr);
        total += single_sphere_bounds(rad, e.di, e.dj, e.dk, nullptr);
    }
    e.storage.resize(static_cast<std::size_t>(total));
    for (int imr = 1; imr <= max_imr; ++imr) {
        const double rad = e.drmin * static_cast<double>(imr);
        const int off = e.offsets[static_cast<std::size_t>(imr - 1)];
        single_sphere_bounds(rad, e.di, e.dj, e.dk, e.storage.data() + off);
    }
}

/// Lookup discretized sphere bounds into thread_local buffer.
/// Hot path: per-thread last (di,dj,dk,imr) hit — no lock, no shared pointer.
const int* sphere_bounds_lookup(double radius, double di, double dj, double dk,
                                double* disr_radius_out) {
    thread_local std::vector<int> local;
    thread_local double t_di = -1, t_dj = -1, t_dk = -1, t_drmin = 0;
    thread_local int t_imr = -1;

    const double drmin = std::min({di, dj, dk});
    const int imr = std::max(1, static_cast<int>(std::ceil(radius / drmin)));
    *disr_radius_out = drmin * static_cast<double>(imr);
    if (t_di == di && t_dj == dj && t_dk == dk && t_imr == imr
        && !local.empty())
        return local.data();

    SphereCache& cache = sphere_cache();
    std::lock_guard<std::mutex> lock(cache.mu);
    SphereEntry* ent = nullptr;
    if (cache.prev >= 0
        && cache.prev < static_cast<int>(cache.entries.size())) {
        SphereEntry& e = cache.entries[static_cast<std::size_t>(cache.prev)];
        if (e.di == di && e.dj == dj && e.dk == dk)
            ent = &e;
    }
    if (!ent) {
        for (std::size_t i = 0; i < cache.entries.size(); ++i) {
            SphereEntry& e = cache.entries[i];
            if (e.di == di && e.dj == dj && e.dk == dk) {
                cache.prev = static_cast<int>(i);
                ent = &e;
                break;
            }
        }
    }
    if (!ent) {
        SphereEntry e;
        e.di = di;
        e.dj = dj;
        e.dk = dk;
        e.drmin = drmin;
        e.max_imr = 0;
        cache.entries.push_back(std::move(e));
        cache.prev = static_cast<int>(cache.entries.size() - 1);
        ent = &cache.entries.back();
    }
    if (ent->max_imr < imr)
        rebuild_sphere_entry(*ent, imr);
    const int off = ent->offsets[static_cast<std::size_t>(imr - 1)];
    int n = 0;
    if (imr < ent->max_imr)
        n = ent->offsets[static_cast<std::size_t>(imr)] - off;
    else
        n = static_cast<int>(ent->storage.size()) - off;
    if (n < 1)
        n = 1;
    local.resize(static_cast<std::size_t>(n));
    std::memcpy(local.data(), ent->storage.data() + off,
                static_cast<std::size_t>(n) * sizeof(int));
    t_di = di;
    t_dj = dj;
    t_dk = dk;
    t_drmin = drmin;
    t_imr = imr;
    return local.data();
}

void group_angular_max(const GridProduct* prods, int n, int& lpx, int& lpy,
                       int& lpz, int& Ltot) {
    lpx = lpy = lpz = Ltot = 0;
    for (int i = 0; i < n; ++i) {
        const auto& p = prods[i];
        lpx = std::max(lpx, p.la[0] + p.lb[0]);
        lpy = std::max(lpy, p.la[1] + p.lb[1]);
        lpz = std::max(lpz, p.la[2] + p.lb[2]);
        Ltot = std::max(Ltot, p.la[0] + p.la[1] + p.la[2] + p.lb[0] + p.lb[1]
                                  + p.lb[2]);
    }
}

void add_product_to_cxyz(const GridProduct& prod, double weight, int lpx,
                         int lpy, int lpz, double* cxyz) {
    if (weight == 0.0)
        return;
    double ax[kMaxL + 1], ay[kMaxL + 1], az[kMaxL + 1];
    const int lpx_p = expand_axis(prod.la[0], prod.lb[0], prod.pc[0],
                                  prod.ra[0], prod.rb[0], ax);
    const int lpy_p = expand_axis(prod.la[1], prod.lb[1], prod.pc[1],
                                  prod.ra[1], prod.rb[1], ay);
    const int lpz_p = expand_axis(prod.la[2], prod.lb[2], prod.pc[2],
                                  prod.ra[2], prod.rb[2], az);
    const int sy = lpy + 1, sz = lpz + 1;
    for (int lx = 0; lx <= lpx_p; ++lx)
        for (int ly = 0; ly <= lpy_p; ++ly)
            for (int lz = 0; lz <= lpz_p; ++lz)
                cxyz[(lx * sy + ly) * sz + lz] +=
                    weight * ax[lx] * ay[ly] * az[lz];
}

double contract_product_from_I(const GridProduct& prod, int lpx, int lpy,
                               int lpz, const double* I) {
    double ax[kMaxL + 1], ay[kMaxL + 1], az[kMaxL + 1];
    const int lpx_p = expand_axis(prod.la[0], prod.lb[0], prod.pc[0],
                                  prod.ra[0], prod.rb[0], ax);
    const int lpy_p = expand_axis(prod.la[1], prod.lb[1], prod.pc[1],
                                  prod.ra[1], prod.rb[1], ay);
    const int lpz_p = expand_axis(prod.la[2], prod.lb[2], prod.pc[2],
                                  prod.ra[2], prod.rb[2], az);
    const int sy = lpy + 1, sz = lpz + 1;
    double s = 0.0;
    for (int lx = 0; lx <= lpx_p; ++lx)
        for (int ly = 0; ly <= lpy_p; ++ly)
            for (int lz = 0; lz <= lpz_p; ++lz)
                s += I[(lx * sy + ly) * sz + lz] * ax[lx] * ay[ly] * az[lz];
    return s;
}

// ─── lp_z-specialized cx → grid (collocate) ─────────────────────────────────
template <int LpZ>
inline void collocate_cx_to_grid_lp(int lb, int ub, int cmax, int pol_len,
                                    const double* pol_i, const double* cx,
                                    const int* map0, const int* sec0,
                                    long base0, long base1, long base2,
                                    long base3, double* grid_io) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        for (; i + Wd - 1 <= istop; i += Wd) {
            const int ig = i + cube2grid;
            batch_t r0(0.0), r1(0.0), r2(0.0), r3(0.0);
#pragma GCC unroll 8
            for (int lz = 0; lz <= LpZ; ++lz) {
                const batch_t p =
                    batch_t::load_unaligned(pol_i + lz * pol_len + (i + cmax));
                r0 += p * batch_t(cx[lz * 4 + 0]);
                r1 += p * batch_t(cx[lz * 4 + 1]);
                r2 += p * batch_t(cx[lz * 4 + 2]);
                r3 += p * batch_t(cx[lz * 4 + 3]);
            }
            batch_t g0 = batch_t::load_unaligned(grid_io + base0 + ig);
            batch_t g1 = batch_t::load_unaligned(grid_io + base1 + ig);
            batch_t g2 = batch_t::load_unaligned(grid_io + base2 + ig);
            batch_t g3 = batch_t::load_unaligned(grid_io + base3 + ig);
            (g0 + r0).store_unaligned(grid_io + base0 + ig);
            (g1 + r1).store_unaligned(grid_io + base1 + ig);
            (g2 + r2).store_unaligned(grid_io + base2 + ig);
            (g3 + r3).store_unaligned(grid_io + base3 + ig);
        }
        for (; i <= istop; ++i) {
            const int ig = i + cube2grid;
            double reg0 = 0.0, reg1 = 0.0, reg2 = 0.0, reg3 = 0.0;
#pragma GCC unroll 8
            for (int lz = 0; lz <= LpZ; ++lz) {
                const double p = pol_i[lz * pol_len + (i + cmax)];
                reg0 += cx[lz * 4 + 0] * p;
                reg1 += cx[lz * 4 + 1] * p;
                reg2 += cx[lz * 4 + 2] * p;
                reg3 += cx[lz * 4 + 3] * p;
            }
            grid_io[base0 + ig] += reg0;
            grid_io[base1 + ig] += reg1;
            grid_io[base2 + ig] += reg2;
            grid_io[base3 + ig] += reg3;
        }
        istart = istop + 1;
    }
}

inline void collocate_cx_to_grid_dyn(int lp_z, int lb, int ub, int cmax,
                                     int pol_len, const double* pol_i,
                                     const double* cx, const int* map0,
                                     const int* sec0, long base0, long base1,
                                     long base2, long base3, double* grid_io) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        for (; i + Wd - 1 <= istop; i += Wd) {
            const int ig = i + cube2grid;
            batch_t r0(0.0), r1(0.0), r2(0.0), r3(0.0);
            for (int lz = 0; lz <= lp_z; ++lz) {
                const batch_t p =
                    batch_t::load_unaligned(pol_i + lz * pol_len + (i + cmax));
                r0 += p * batch_t(cx[lz * 4 + 0]);
                r1 += p * batch_t(cx[lz * 4 + 1]);
                r2 += p * batch_t(cx[lz * 4 + 2]);
                r3 += p * batch_t(cx[lz * 4 + 3]);
            }
            batch_t g0 = batch_t::load_unaligned(grid_io + base0 + ig);
            batch_t g1 = batch_t::load_unaligned(grid_io + base1 + ig);
            batch_t g2 = batch_t::load_unaligned(grid_io + base2 + ig);
            batch_t g3 = batch_t::load_unaligned(grid_io + base3 + ig);
            (g0 + r0).store_unaligned(grid_io + base0 + ig);
            (g1 + r1).store_unaligned(grid_io + base1 + ig);
            (g2 + r2).store_unaligned(grid_io + base2 + ig);
            (g3 + r3).store_unaligned(grid_io + base3 + ig);
        }
        for (; i <= istop; ++i) {
            const int ig = i + cube2grid;
            double reg0 = 0.0, reg1 = 0.0, reg2 = 0.0, reg3 = 0.0;
            for (int lz = 0; lz <= lp_z; ++lz) {
                const double p = pol_i[lz * pol_len + (i + cmax)];
                reg0 += cx[lz * 4 + 0] * p;
                reg1 += cx[lz * 4 + 1] * p;
                reg2 += cx[lz * 4 + 2] * p;
                reg3 += cx[lz * 4 + 3] * p;
            }
            grid_io[base0 + ig] += reg0;
            grid_io[base1 + ig] += reg1;
            grid_io[base2 + ig] += reg2;
            grid_io[base3 + ig] += reg3;
        }
        istart = istop + 1;
    }
}

inline void dispatch_collocate_cx(int lp_z, int lb, int ub, int cmax,
                                  int pol_len, const double* pol_i,
                                  const double* cx, const int* map0,
                                  const int* sec0, long base0, long base1,
                                  long base2, long base3, double* grid_io) {
    switch (lp_z) {
    case 0:
        collocate_cx_to_grid_lp<0>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 1:
        collocate_cx_to_grid_lp<1>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 2:
        collocate_cx_to_grid_lp<2>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 3:
        collocate_cx_to_grid_lp<3>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 4:
        collocate_cx_to_grid_lp<4>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 5:
        collocate_cx_to_grid_lp<5>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 6:
        collocate_cx_to_grid_lp<6>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    default:
        collocate_cx_to_grid_dyn(lp_z, lb, ub, cmax, pol_len, pol_i, cx, map0,
                                 sec0, base0, base1, base2, base3, grid_io);
        break;
    }
}

// ─── lp_z-specialized grid → cx (integrate): SIMD lane accumulate, one
// horizontal reduce per (lz, segment) instead of per vector chunk.
template <int LpZ>
inline void integrate_cx_from_grid_lp(int lb, int ub, int cmax, int pol_len,
                                      const double* pol_i, double* cx,
                                      const int* map0, const int* sec0,
                                      long base0, long base1, long base2,
                                      long base3, const double* grid_io) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        batch_t a0[LpZ + 1], a1[LpZ + 1], a2[LpZ + 1], a3[LpZ + 1];
#pragma GCC unroll 8
        for (int lz = 0; lz <= LpZ; ++lz) {
            a0[lz] = batch_t(0.0);
            a1[lz] = batch_t(0.0);
            a2[lz] = batch_t(0.0);
            a3[lz] = batch_t(0.0);
        }
        for (; i + Wd - 1 <= istop; i += Wd) {
            const int ig = i + cube2grid;
            const batch_t v0 = batch_t::load_unaligned(grid_io + base0 + ig);
            const batch_t v1 = batch_t::load_unaligned(grid_io + base1 + ig);
            const batch_t v2 = batch_t::load_unaligned(grid_io + base2 + ig);
            const batch_t v3 = batch_t::load_unaligned(grid_io + base3 + ig);
#pragma GCC unroll 8
            for (int lz = 0; lz <= LpZ; ++lz) {
                const batch_t p =
                    batch_t::load_unaligned(pol_i + lz * pol_len + (i + cmax));
                a0[lz] += v0 * p;
                a1[lz] += v1 * p;
                a2[lz] += v2 * p;
                a3[lz] += v3 * p;
            }
        }
#pragma GCC unroll 8
        for (int lz = 0; lz <= LpZ; ++lz) {
            cx[lz * 4 + 0] += xsimd::reduce_add(a0[lz]);
            cx[lz * 4 + 1] += xsimd::reduce_add(a1[lz]);
            cx[lz * 4 + 2] += xsimd::reduce_add(a2[lz]);
            cx[lz * 4 + 3] += xsimd::reduce_add(a3[lz]);
        }
        for (; i <= istop; ++i) {
            const int ig = i + cube2grid;
            const double v0 = grid_io[base0 + ig];
            const double v1 = grid_io[base1 + ig];
            const double v2 = grid_io[base2 + ig];
            const double v3 = grid_io[base3 + ig];
#pragma GCC unroll 8
            for (int lz = 0; lz <= LpZ; ++lz) {
                const double p = pol_i[lz * pol_len + (i + cmax)];
                cx[lz * 4 + 0] += v0 * p;
                cx[lz * 4 + 1] += v1 * p;
                cx[lz * 4 + 2] += v2 * p;
                cx[lz * 4 + 3] += v3 * p;
            }
        }
        istart = istop + 1;
    }
}

inline void integrate_cx_from_grid_dyn(int lp_z, int lb, int ub, int cmax,
                                       int pol_len, const double* pol_i,
                                       double* cx, const int* map0,
                                       const int* sec0, long base0, long base1,
                                       long base2, long base3,
                                       const double* grid_io) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        for (; i + Wd - 1 <= istop; i += Wd) {
            const int ig = i + cube2grid;
            const batch_t v0 = batch_t::load_unaligned(grid_io + base0 + ig);
            const batch_t v1 = batch_t::load_unaligned(grid_io + base1 + ig);
            const batch_t v2 = batch_t::load_unaligned(grid_io + base2 + ig);
            const batch_t v3 = batch_t::load_unaligned(grid_io + base3 + ig);
            for (int lz = 0; lz <= lp_z; ++lz) {
                const batch_t p =
                    batch_t::load_unaligned(pol_i + lz * pol_len + (i + cmax));
                // Accumulate lane products then one horizontal reduce (still
                // one reduce per lz; better than four reduce_add of v*p).
                cx[lz * 4 + 0] += xsimd::reduce_add(v0 * p);
                cx[lz * 4 + 1] += xsimd::reduce_add(v1 * p);
                cx[lz * 4 + 2] += xsimd::reduce_add(v2 * p);
                cx[lz * 4 + 3] += xsimd::reduce_add(v3 * p);
            }
        }
        for (; i <= istop; ++i) {
            const int ig = i + cube2grid;
            const double v0 = grid_io[base0 + ig];
            const double v1 = grid_io[base1 + ig];
            const double v2 = grid_io[base2 + ig];
            const double v3 = grid_io[base3 + ig];
            for (int lz = 0; lz <= lp_z; ++lz) {
                const double p = pol_i[lz * pol_len + (i + cmax)];
                cx[lz * 4 + 0] += v0 * p;
                cx[lz * 4 + 1] += v1 * p;
                cx[lz * 4 + 2] += v2 * p;
                cx[lz * 4 + 3] += v3 * p;
            }
        }
        istart = istop + 1;
    }
}

inline void dispatch_integrate_cx(int lp_z, int lb, int ub, int cmax,
                                  int pol_len, const double* pol_i, double* cx,
                                  const int* map0, const int* sec0, long base0,
                                  long base1, long base2, long base3,
                                  const double* grid_io) {
    switch (lp_z) {
    case 0:
        integrate_cx_from_grid_lp<0>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 1:
        integrate_cx_from_grid_lp<1>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 2:
        integrate_cx_from_grid_lp<2>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 3:
        integrate_cx_from_grid_lp<3>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 4:
        integrate_cx_from_grid_lp<4>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 5:
        integrate_cx_from_grid_lp<5>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 6:
        integrate_cx_from_grid_lp<6>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    default:
        integrate_cx_from_grid_dyn(lp_z, lb, ub, cmax, pol_len, pol_i, cx, map0,
                                   sec0, base0, base1, base2, base3, grid_io);
        break;
    }
}

template <bool Collocate>
void ortho_cxyz_to_grid(const UniformGrid& grid, const double pc[3],
                        double zetp, double radius, int lpx, int lpy, int lpz,
                        double* cxyz, double* grid_io) {
    const double di = grid.L[2] / static_cast<double>(grid.n[2]);
    const double dj = grid.L[1] / static_cast<double>(grid.n[1]);
    const double dk = grid.L[0] / static_cast<double>(grid.n[0]);
    if (!(radius > 0.0) || 2.0 * radius < std::max({di, dj, dk}))
        return;

    const double rp[3] = {pc[2] - grid.origin[2], pc[1] - grid.origin[1],
                          pc[0] - grid.origin[0]};
    int cubecenter[3];
    cubecenter[0] = static_cast<int>(std::floor(rp[0] / di));
    cubecenter[1] = static_cast<int>(std::floor(rp[1] / dj));
    cubecenter[2] = static_cast<int>(std::floor(rp[2] / dk));
    const double roffset[3] = {rp[0] - cubecenter[0] * di,
                               rp[1] - cubecenter[1] * dj,
                               rp[2] - cubecenter[2] * dk};
    const double dr[3] = {di, dj, dk};

    double disr_radius = 0.0;
    const int* sphere =
        sphere_bounds_lookup(radius, di, dj, dk, &disr_radius);

    int lb_cube[3], ub_cube[3];
    lb_cube[0] = static_cast<int>(std::ceil(-1e-8 - disr_radius / di));
    lb_cube[1] = static_cast<int>(std::ceil(-1e-8 - disr_radius / dj));
    lb_cube[2] = static_cast<int>(std::ceil(-1e-8 - disr_radius / dk));
    for (int d = 0; d < 3; ++d)
        ub_cube[d] = 1 - lb_cube[d];
    const int cmax = imax(imax(ub_cube[0], ub_cube[1]), ub_cube[2]);
    if (cmax <= 0 || cmax > kMaxCmax)
        return;

    const int lp_x = lpx, lp_y = lpy, lp_z = lpz;
    const int pol_len = 2 * cmax + 1;
    const int pol_lp[3] = {lp_z, lp_y, lp_x};
    int pol_off[3];
    int po = 0;
    for (int d = 0; d < 3; ++d) {
        pol_off[d] = po;
        po += (pol_lp[d] + 1) * pol_len;
    }
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxy_n = sy * sz * 2;
    const int cx_n = sz * 4;
    ensure_buf(static_cast<std::size_t>(po + cxy_n + cx_n));
    ensure_imap(static_cast<std::size_t>(3 * pol_len * 2 + 8));

    double* pol_base = g_buf.data();
    double* cxy = g_buf.data() + po;
    double* cx = cxy + cxy_n;

    for (int idir = 0; idir < 3; ++idir) {
        const double dri = dr[idir];
        const double ro = roffset[idir];
        const int lp = pol_lp[idir];
        double* pol = pol_base + pol_off[idir];
        const double t_exp_1 = std::exp(-zetp * dri * dri);
        const double t_exp_2 = t_exp_1 * t_exp_1;
        double t_exp_min_1 = std::exp(-zetp * (dri - ro) * (dri - ro));
        double t_exp_min_2 = std::exp(-2.0 * zetp * (dri - ro) * (-dri));
        for (int ig = 0; ig >= lb_cube[idir]; --ig) {
            const double rpg = static_cast<double>(ig) * dri - ro;
            t_exp_min_1 *= t_exp_min_2 * t_exp_1;
            t_exp_min_2 *= t_exp_2;
            double pg = t_exp_min_1;
            for (int l = 0; l <= lp; ++l) {
                pol[l * pol_len + (ig + cmax)] = pg;
                pg *= rpg;
            }
        }
        double t_exp_plus_1 = std::exp(-zetp * (-ro) * (-ro));
        double t_exp_plus_2 = std::exp(-2.0 * zetp * (-ro) * dri);
        for (int ig = 0; ig >= lb_cube[idir]; --ig) {
            const double rpg = static_cast<double>(1 - ig) * dri - ro;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg = t_exp_plus_1;
            for (int l = 0; l <= lp; ++l) {
                pol[l * pol_len + ((1 - ig) + cmax)] = pg;
                pg *= rpg;
            }
        }
    }

    const int n_atv[3] = {grid.n[2], grid.n[1], grid.n[0]};
    int* map0 = g_imap.data();
    int* map1 = map0 + pol_len;
    int* map2 = map1 + pol_len;
    int* sec0 = map2 + pol_len;
    int* sec1 = sec0 + pol_len;
    int* sec2 = sec1 + pol_len;
    int* maps[3] = {map0, map1, map2};
    int* secs[3] = {sec0, sec1, sec2};
    for (int d = 0; d < 3; ++d) {
        for (int k = -cmax; k <= cmax; ++k)
            maps[d][k + cmax] = modulo(cubecenter[d] + k, n_atv[d]);
        for (int kg = 2 * cmax; kg >= 0; --kg) {
            if (kg == 2 * cmax || maps[d][kg] != maps[d][kg + 1] - 1)
                secs[d][kg] = 0;
            else
                secs[d][kg] = secs[d][kg + 1] + 1;
        }
    }

    const double* pol_i = pol_base + pol_off[0];
    const double* pol_j = pol_base + pol_off[1];
    const double* pol_k = pol_base + pol_off[2];
    const long ny = grid.n[1], nz = grid.n[2];
    const int* sb = sphere;
    const int kstart = *sb++;

    for (int k1 = kstart; k1 <= 0; ++k1) {
        const int k2 = 1 - k1;
        const int kg1 = map2[k1 + cmax];
        const int kg2 = map2[k2 + cmax];
        std::fill(cxy, cxy + cxy_n, 0.0);

        if constexpr (Collocate) {
            for (int lx = 0; lx <= lp_x; ++lx) {
                const double p1 = pol_k[lx * pol_len + (k1 + cmax)];
                const double p2 = pol_k[lx * pol_len + (k2 + cmax)];
                if (p1 == 0.0 && p2 == 0.0)
                    continue;
                for (int ly = 0; ly <= lp_y; ++ly)
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const double c = cxyz[(lx * sy + ly) * sz + lz];
                        if (c == 0.0)
                            continue;
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxy[cxy_i + 0] += c * p1;
                        cxy[cxy_i + 1] += c * p2;
                    }
            }
        }

        const int jstart = *sb++;
        for (int j1 = jstart; j1 <= 0; ++j1) {
            const int j2 = 1 - j1;
            const int jg1 = map1[j1 + cmax];
            const int jg2 = map1[j2 + cmax];
            std::fill(cx, cx + cx_n, 0.0);

            if constexpr (Collocate) {
                for (int ly = 0; ly <= lp_y; ++ly) {
                    const double p1 = pol_j[ly * pol_len + (j1 + cmax)];
                    const double p2 = pol_j[ly * pol_len + (j2 + cmax)];
                    if (p1 == 0.0 && p2 == 0.0)
                        continue;
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        const double c0 = cxy[cxy_i + 0];
                        const double c1 = cxy[cxy_i + 1];
                        if (c0 == 0.0 && c1 == 0.0)
                            continue;
                        cx[lz * 4 + 0] += c0 * p1;
                        cx[lz * 4 + 1] += c1 * p1;
                        cx[lz * 4 + 2] += c0 * p2;
                        cx[lz * 4 + 3] += c1 * p2;
                    }
                }
            }

            const int lb = *sb++;
            const int ub = 1 - lb;
            const long base0 = (static_cast<long>(kg1) * ny + jg1) * nz;
            const long base1 = (static_cast<long>(kg2) * ny + jg1) * nz;
            const long base2 = (static_cast<long>(kg1) * ny + jg2) * nz;
            const long base3 = (static_cast<long>(kg2) * ny + jg2) * nz;

            if constexpr (Collocate) {
                dispatch_collocate_cx(lp_z, lb, ub, cmax, pol_len, pol_i, cx,
                                      map0, sec0, base0, base1, base2, base3,
                                      grid_io);
            } else {
                dispatch_integrate_cx(lp_z, lb, ub, cmax, pol_len, pol_i, cx,
                                      map0, sec0, base0, base1, base2, base3,
                                      grid_io);
                for (int ly = 0; ly <= lp_y; ++ly) {
                    const double p1 = pol_j[ly * pol_len + (j1 + cmax)];
                    const double p2 = pol_j[ly * pol_len + (j2 + cmax)];
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxy[cxy_i + 0] += cx[lz * 4 + 0] * p1;
                        cxy[cxy_i + 1] += cx[lz * 4 + 1] * p1;
                        cxy[cxy_i + 0] += cx[lz * 4 + 2] * p2;
                        cxy[cxy_i + 1] += cx[lz * 4 + 3] * p2;
                    }
                }
            }
        }

        if constexpr (!Collocate) {
            for (int lx = 0; lx <= lp_x; ++lx) {
                const double p1 = pol_k[lx * pol_len + (k1 + cmax)];
                const double p2 = pol_k[lx * pol_len + (k2 + cmax)];
                for (int ly = 0; ly <= lp_y; ++ly)
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxyz[(lx * sy + ly) * sz + lz] +=
                            cxy[cxy_i + 0] * p1 + cxy[cxy_i + 1] * p2;
                    }
            }
        }
    }
}

} // namespace

void collint_collocate_product(const UniformGrid& grid, const GridProduct& prod,
                               double weight, double mag, double eps,
                               double* rho) {
    double w = weight;
    collint_collocate_group(grid, &prod, &w, 1, mag, eps, rho);
}

double collint_integrate_product(const UniformGrid& grid,
                                 const GridProduct& prod, double mag,
                                 double eps, const double* V) {
    double acc = 0.0;
    collint_integrate_group(grid, &prod, 1, mag, eps, V, &acc);
    return acc;
}

void collint_collocate_group(const UniformGrid& grid, const GridProduct* prods,
                             const double* weights, int n, double mag,
                             double eps, double* rho) {
    if (n <= 0 || !(mag > 0.0))
        return;
    int lpx, lpy, lpz, Ltot;
    group_angular_max(prods, n, lpx, lpy, lpz, Ltot);
    if (lpx > kMaxL || lpy > kMaxL || lpz > kMaxL)
        return;
    const double radius = shape_radius(mag, prods[0].p, Ltot, eps);
    if (!(radius > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* cxyz = g_I.data();
    std::fill(cxyz, cxyz + cxyz_n, 0.0);
    for (int i = 0; i < n; ++i)
        add_product_to_cxyz(prods[i], weights[i], lpx, lpy, lpz, cxyz);
    ortho_cxyz_to_grid<true>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                             lpz, cxyz, rho);
}

void collint_collocate_group_idx(const UniformGrid& grid,
                                 const GridProduct* prods_base, const int* idx,
                                 const double* weights, int n, double mag,
                                 double eps, double* rho) {
    if (n <= 0 || !(mag > 0.0))
        return;
    // Angular max + cxyz from indexed monoms (no GridProduct copy).
    int lpx = 0, lpy = 0, lpz = 0, Ltot = 0;
    for (int i = 0; i < n; ++i) {
        const auto& p = prods_base[idx[i]];
        lpx = std::max(lpx, p.la[0] + p.lb[0]);
        lpy = std::max(lpy, p.la[1] + p.lb[1]);
        lpz = std::max(lpz, p.la[2] + p.lb[2]);
        Ltot = std::max(Ltot, p.la[0] + p.la[1] + p.la[2] + p.lb[0] + p.lb[1]
                                  + p.lb[2]);
    }
    if (lpx > kMaxL || lpy > kMaxL || lpz > kMaxL)
        return;
    const GridProduct& g0 = prods_base[idx[0]];
    const double radius = shape_radius(mag, g0.p, Ltot, eps);
    if (!(radius > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* cxyz = g_I.data();
    std::fill(cxyz, cxyz + cxyz_n, 0.0);
    for (int i = 0; i < n; ++i)
        add_product_to_cxyz(prods_base[idx[i]], weights[i], lpx, lpy, lpz,
                            cxyz);
    ortho_cxyz_to_grid<true>(grid, g0.pc, g0.p, radius, lpx, lpy, lpz, cxyz,
                             rho);
}

void collint_integrate_group(const UniformGrid& grid, const GridProduct* prods,
                             int n, double mag, double eps, const double* V,
                             double* out_acc) {
    for (int i = 0; i < n; ++i)
        out_acc[i] = 0.0;
    if (n <= 0 || !(mag > 0.0))
        return;
    int lpx, lpy, lpz, Ltot;
    group_angular_max(prods, n, lpx, lpy, lpz, Ltot);
    if (lpx > kMaxL || lpy > kMaxL || lpz > kMaxL)
        return;
    const double radius = shape_radius(mag, prods[0].p, Ltot, eps);
    if (!(radius > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* I = g_I.data();
    std::fill(I, I + cxyz_n, 0.0);
    ortho_cxyz_to_grid<false>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                              lpz, I, const_cast<double*>(V));
    for (int i = 0; i < n; ++i)
        out_acc[i] = contract_product_from_I(prods[i], lpx, lpy, lpz, I);
}

void collint_integrate_group_idx(const UniformGrid& grid,
                                 const GridProduct* prods_base, const int* idx,
                                 int n, double mag, double eps, const double* V,
                                 double* out_acc) {
    for (int i = 0; i < n; ++i)
        out_acc[i] = 0.0;
    if (n <= 0 || !(mag > 0.0))
        return;
    int lpx = 0, lpy = 0, lpz = 0, Ltot = 0;
    for (int i = 0; i < n; ++i) {
        const auto& p = prods_base[idx[i]];
        lpx = std::max(lpx, p.la[0] + p.lb[0]);
        lpy = std::max(lpy, p.la[1] + p.lb[1]);
        lpz = std::max(lpz, p.la[2] + p.lb[2]);
        Ltot = std::max(Ltot, p.la[0] + p.la[1] + p.la[2] + p.lb[0] + p.lb[1]
                                  + p.lb[2]);
    }
    if (lpx > kMaxL || lpy > kMaxL || lpz > kMaxL)
        return;
    const GridProduct& g0 = prods_base[idx[0]];
    const double radius = shape_radius(mag, g0.p, Ltot, eps);
    if (!(radius > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* I = g_I.data();
    std::fill(I, I + cxyz_n, 0.0);
    ortho_cxyz_to_grid<false>(grid, g0.pc, g0.p, radius, lpx, lpy, lpz, I,
                              const_cast<double*>(V));
    for (int i = 0; i < n; ++i)
        out_acc[i] =
            contract_product_from_I(prods_base[idx[i]], lpx, lpy, lpz, I);
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
