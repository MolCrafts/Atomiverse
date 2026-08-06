/// @file src/cpu/gpw/internal/product_collint.cpp
/// @brief CP2K ortho collint product collocate/integrate (see product_collint.h).
///
/// Near-literal port of CP2K `src/grid/cpu/grid_cpu_collint.h` ortho path
/// (BSD-3-Clause, Ole Schuett et al.) plus:
///   - process-wide sphere-bounds cache (keyed by dh + discretized radius)
///   - lp_z-specialized inner contract (CP2K GRID_MAX_LP_OPTIMIZED=9)
///   - portable xsimd on contiguous z-runs with 4-wide residual (AVX-512 hosts)
///   - integrate residual: short-segment scalar (skip batch zero+reduce),
///     zero-pol skip on reverse, Ltot triangular skip on rectangular monom walks,
///     cab shared-alpha from n≥4
///
/// Axis map onto ATV z-fast layout `g[(ix*ny+iy)*nz+iz]`:
///   CP2K i (fast) ↔ ATV z,  j ↔ y,  k ↔ x.

#include "atomiverse/cpu/gpw/internal/product_collint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {
namespace {

constexpr int kMaxL = 12;
constexpr int kMaxCmax = 128;
constexpr int kMaxLpOpt = 9; // match CP2K GRID_MAX_LP_OPTIMIZED

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


// Safe size probe: make_sized_batch_t may be void on some arches.
template <typename T, bool Ok = !std::is_void_v<T>>
struct batch_width {
    static constexpr int value = 0;
};
template <typename T>
struct batch_width<T, true> {
    static constexpr int value = static_cast<int>(T::size);
};

// 4-wide residual is an AVX-512-only win (native width 8, residual width 4).
// On NEON/aarch64 make_sized_batch_t<double,4> is void — never name it outside
// an #if __AVX512F__ block (if constexpr alone is not enough: the type is
// non-dependent and some hosts still diagnose the discarded arm).
#if defined(__AVX512F__)
using batch4_maybe_t = xsimd::make_sized_batch_t<double, 4>;
constexpr bool kHasBatch4 = batch_width<batch4_maybe_t>::value == 4;
constexpr int kNativeBatch =
    static_cast<int>(xsimd::batch<double>::size);
constexpr bool kUseBatch4Residual = kHasBatch4 && (kNativeBatch > 4);
#else
constexpr bool kUseBatch4Residual = false;
#endif


// Pascal triangle LUT for expand_axis (la,lb ≤ kMaxL).
inline int binom(int n, int k) {
    static int C[kMaxL + 1][kMaxL + 1];
    static bool init = false;
    if (!init) {
        for (int i = 0; i <= kMaxL; ++i) {
            C[i][0] = C[i][i] = 1;
            for (int j = 1; j < i; ++j)
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
        init = true;
    }
    if (k < 0 || k > n || n > kMaxL)
        return 0;
    return C[n][k];
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

/// Radius from assembled cartesian product coefficients (CP2K-style).
/// After cab expand, max|cxyz| is the true prefactor for r^L exp(-p r²);
/// using the density/geometry mag alone often oversizes the sphere.
inline double cmax_radius(const double* cxyz, int n, double p, int Ltot,
                          double eps) {
    double cmax = 0.0;
    for (int i = 0; i < n; ++i)
        cmax = std::max(cmax, std::abs(cxyz[i]));
    return shape_radius(cmax, p, Ltot, eps);
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
/// Hot path: per-thread (di,dj,dk) entry with per-imr vectors — no lock when
/// the imr was already copied for this spacing. Process cache still fills new
/// (di,dj,dk,imr) once under the mutex.
const int* sphere_bounds_lookup(double radius, double di, double dj, double dk,
                                double* disr_radius_out) {
    struct TlsSp {
        double di = -1, dj = -1, dk = -1, drmin = 0;
        std::vector<std::vector<int>> by_imr; // index imr-1
    };
    thread_local TlsSp tls;

    const double drmin = std::min({di, dj, dk});
    const int imr = std::max(1, static_cast<int>(std::ceil(radius / drmin)));
    *disr_radius_out = drmin * static_cast<double>(imr);

    if (!(tls.di == di && tls.dj == dj && tls.dk == dk)) {
        tls.di = di;
        tls.dj = dj;
        tls.dk = dk;
        tls.drmin = drmin;
        tls.by_imr.clear();
    }
    if (static_cast<int>(tls.by_imr.size()) >= imr
        && !tls.by_imr[static_cast<std::size_t>(imr - 1)].empty()) {
        return tls.by_imr[static_cast<std::size_t>(imr - 1)].data();
    }

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
    // Copy all imrs up to ent->max_imr into TLS so later radii hit without lock.
    if (static_cast<int>(tls.by_imr.size()) < ent->max_imr)
        tls.by_imr.resize(static_cast<std::size_t>(ent->max_imr));
    for (int r = 1; r <= ent->max_imr; ++r) {
        auto& slot = tls.by_imr[static_cast<std::size_t>(r - 1)];
        if (!slot.empty())
            continue;
        const int off = ent->offsets[static_cast<std::size_t>(r - 1)];
        int n = 0;
        if (r < ent->max_imr)
            n = ent->offsets[static_cast<std::size_t>(r)] - off;
        else
            n = static_cast<int>(ent->storage.size()) - off;
        if (n < 1)
            n = 1;
        slot.resize(static_cast<std::size_t>(n));
        std::memcpy(slot.data(), ent->storage.data() + off,
                    static_cast<std::size_t>(n) * sizeof(int));
    }
    return tls.by_imr[static_cast<std::size_t>(imr - 1)].data();
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
        for (int ly = 0; ly <= lpy_p; ++ly) {
            const double axy = weight * ax[lx] * ay[ly];
            if (axy == 0.0)
                continue;
            double* row = cxyz + (lx * sy + ly) * sz;
            for (int lz = 0; lz <= lpz_p; ++lz)
                row[lz] += axy * az[lz];
        }
}

/// CP2K cab_to_cxyz-style: build alpha[dir][lb][la][ls] once for the group
/// geometry, then scatter each monom weight. Wins when avg_g ≫ 1 (MAIN ~85).
void add_group_to_cxyz(const GridProduct* prods, const double* weights, int n,
                       int lpx, int lpy, int lpz, double* cxyz) {
    if (n <= 0)
        return;
    // Cab shared-alpha wins from small groups (was 8; MAIN tasks often 4–20).
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            add_product_to_cxyz(prods[i], weights[i], lpx, lpy, lpz, cxyz);
        return;
    }
    int max_la[3] = {0, 0, 0}, max_lb[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            max_la[d] = imax(max_la[d], prods[i].la[d]);
            max_lb[d] = imax(max_lb[d], prods[i].lb[d]);
        }
    }
    // alpha[d][lb][la][ls], ls ≤ max_la+max_lb ≤ lpx/y/z.
    // Stack bound: 3 * 13 * 13 * 13 doubles ≈ 50 KiB — OK for collint scratch.
    const int mla = imax(imax(max_la[0], max_la[1]), max_la[2]);
    const int mlb = imax(imax(max_lb[0], max_lb[1]), max_lb[2]);
    const int mlp = mla + mlb;
    if (mla > kMaxL || mlb > kMaxL || mlp > kMaxL) {
        for (int i = 0; i < n; ++i)
            add_product_to_cxyz(prods[i], weights[i], lpx, lpy, lpz, cxyz);
        return;
    }
    thread_local std::vector<double> alpha_buf;
    const int stride_ls = mlp + 1;
    const int stride_la = (mla + 1) * stride_ls;
    const int stride_lb = (mlb + 1) * stride_la;
    const int need = 3 * stride_lb;
    if (static_cast<int>(alpha_buf.size()) < need)
        alpha_buf.assign(static_cast<std::size_t>(need), 0.0);
    else
        std::memset(alpha_buf.data(), 0,
                    static_cast<std::size_t>(need) * sizeof(double));
    const double* pc = prods[0].pc;
    const double* ra = prods[0].ra;
    const double* rb = prods[0].rb;
    for (int d = 0; d < 3; ++d) {
        double* ad = alpha_buf.data() + d * stride_lb;
        const double drpa = pc[d] - ra[d];
        const double drpb = pc[d] - rb[d];
        for (int lxa = 0; lxa <= max_la[d]; ++lxa) {
            for (int lxb = 0; lxb <= max_lb[d]; ++lxb) {
                double binomial_k_lxa = 1.0;
                double a = 1.0;
                for (int k = 0; k <= lxa; ++k) {
                    double binomial_l_lxb = 1.0;
                    double b = 1.0;
                    for (int l = 0; l <= lxb; ++l) {
                        const int ls = lxa - k + lxb - l;
                        ad[(lxb * (mla + 1) + lxa) * stride_ls + ls] +=
                            binomial_k_lxa * binomial_l_lxb * a * b;
                        binomial_l_lxb *=
                            static_cast<double>(lxb - l) / static_cast<double>(l + 1);
                        b *= drpb;
                    }
                    binomial_k_lxa *=
                        static_cast<double>(lxa - k) / static_cast<double>(k + 1);
                    a *= drpa;
                }
            }
        }
    }
    const double* ax = alpha_buf.data();
    const double* ay = alpha_buf.data() + stride_lb;
    const double* az = alpha_buf.data() + 2 * stride_lb;
    const int sy = lpy + 1, sz = lpz + 1;
    for (int i = 0; i < n; ++i) {
        const double w = weights[i];
        if (w == 0.0)
            continue;
        const auto& p = prods[i];
        const int lax = p.la[0], lay = p.la[1], laz = p.la[2];
        const int lbx = p.lb[0], lby = p.lb[1], lbz = p.lb[2];
        const double* ax_row = ax + (lbx * (mla + 1) + lax) * stride_ls;
        const double* ay_row = ay + (lby * (mla + 1) + lay) * stride_ls;
        const double* az_row = az + (lbz * (mla + 1) + laz) * stride_ls;
        const int lpx_p = lax + lbx, lpy_p = lay + lby, lpz_p = laz + lbz;
        for (int lx = 0; lx <= lpx_p; ++lx) {
            if (ax_row[lx] == 0.0)
                continue;
            for (int ly = 0; ly <= lpy_p; ++ly) {
                const double axy = w * ax_row[lx] * ay_row[ly];
                if (axy == 0.0)
                    continue;
                double* row = cxyz + (lx * sy + ly) * sz;
                for (int lz = 0; lz <= lpz_p; ++lz)
                    row[lz] += axy * az_row[lz];
            }
        }
    }
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
        for (int ly = 0; ly <= lpy_p; ++ly) {
            const double axy = ax[lx] * ay[ly];
            if (axy == 0.0)
                continue;
            const double* row = I + (lx * sy + ly) * sz;
            for (int lz = 0; lz <= lpz_p; ++lz)
                s += row[lz] * axy * az[lz];
        }
    return s;
}

/// CP2K cab_to_cxyz transpose for integrate: shared alpha table once per
/// geometry, then contract each monom from the monom integral table I.
/// Wins when avg_g ≫ 1 (MAIN shell tasks ~ tens–hundreds of monoms).
void contract_group_from_I(const GridProduct* prods, int n, int lpx, int lpy,
                           int lpz, const double* I, double* out_acc) {
    if (n <= 0)
        return;
    // Cab shared-alpha wins from small groups (was 8; MAIN tasks often 4–20).
    if (n < 4) {
        for (int i = 0; i < n; ++i)
            out_acc[i] = contract_product_from_I(prods[i], lpx, lpy, lpz, I);
        return;
    }
    int max_la[3] = {0, 0, 0}, max_lb[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            max_la[d] = imax(max_la[d], prods[i].la[d]);
            max_lb[d] = imax(max_lb[d], prods[i].lb[d]);
        }
    }
    const int mla = imax(imax(max_la[0], max_la[1]), max_la[2]);
    const int mlb = imax(imax(max_lb[0], max_lb[1]), max_lb[2]);
    const int mlp = mla + mlb;
    if (mla > kMaxL || mlb > kMaxL || mlp > kMaxL) {
        for (int i = 0; i < n; ++i)
            out_acc[i] = contract_product_from_I(prods[i], lpx, lpy, lpz, I);
        return;
    }
    thread_local std::vector<double> alpha_buf;
    const int stride_ls = mlp + 1;
    const int stride_la = (mla + 1) * stride_ls;
    const int stride_lb = (mlb + 1) * stride_la;
    const int need = 3 * stride_lb;
    if (static_cast<int>(alpha_buf.size()) < need)
        alpha_buf.assign(static_cast<std::size_t>(need), 0.0);
    else
        std::memset(alpha_buf.data(), 0,
                    static_cast<std::size_t>(need) * sizeof(double));
    const double* pc = prods[0].pc;
    const double* ra = prods[0].ra;
    const double* rb = prods[0].rb;
    for (int d = 0; d < 3; ++d) {
        double* ad = alpha_buf.data() + d * stride_lb;
        const double drpa = pc[d] - ra[d];
        const double drpb = pc[d] - rb[d];
        for (int lxa = 0; lxa <= max_la[d]; ++lxa) {
            for (int lxb = 0; lxb <= max_lb[d]; ++lxb) {
                double binomial_k_lxa = 1.0;
                double a = 1.0;
                for (int k = 0; k <= lxa; ++k) {
                    double binomial_l_lxb = 1.0;
                    double b = 1.0;
                    for (int l = 0; l <= lxb; ++l) {
                        const int ls = lxa - k + lxb - l;
                        ad[(lxb * (mla + 1) + lxa) * stride_ls + ls] +=
                            binomial_k_lxa * binomial_l_lxb * a * b;
                        binomial_l_lxb *=
                            static_cast<double>(lxb - l) / static_cast<double>(l + 1);
                        b *= drpb;
                    }
                    binomial_k_lxa *=
                        static_cast<double>(lxa - k) / static_cast<double>(k + 1);
                    a *= drpa;
                }
            }
        }
    }
    const double* ax = alpha_buf.data();
    const double* ay = alpha_buf.data() + stride_lb;
    const double* az = alpha_buf.data() + 2 * stride_lb;
    const int sy = lpy + 1, sz = lpz + 1;
    for (int i = 0; i < n; ++i) {
        const auto& p = prods[i];
        const int lax = p.la[0], lay = p.la[1], laz = p.la[2];
        const int lbx = p.lb[0], lby = p.lb[1], lbz = p.lb[2];
        const double* ax_row = ax + (lbx * (mla + 1) + lax) * stride_ls;
        const double* ay_row = ay + (lby * (mla + 1) + lay) * stride_ls;
        const double* az_row = az + (lbz * (mla + 1) + laz) * stride_ls;
        const int lpx_p = lax + lbx, lpy_p = lay + lby, lpz_p = laz + lbz;
        double s = 0.0;
        for (int lx = 0; lx <= lpx_p; ++lx) {
            if (ax_row[lx] == 0.0)
                continue;
            for (int ly = 0; ly <= lpy_p; ++ly) {
                const double axy = ax_row[lx] * ay_row[ly];
                if (axy == 0.0)
                    continue;
                const double* row = I + (lx * sy + ly) * sz;
                for (int lz = 0; lz <= lpz_p; ++lz)
                    s += row[lz] * axy * az_row[lz];
            }
        }
        out_acc[i] = s;
    }
}

// ─── lp_z-specialized cx → grid (collocate) ─────────────────────────────────
// Native batch (AVX-512 = 8 on this host) plus a 4-wide residual pass so short
// sphere segments (common when the cube wraps) still vectorize — matches CP2K's
// AVX2 width-4 path without arch intrinsics.
template <typename Batch, int LpZ>
inline void collocate_cx_vec_pass(int& i, int istop, int cmax, int pol_len,
                                  const double* pol_i, const double* cx,
                                  int cube2grid, long base0, long base1,
                                  long base2, long base3, double* grid_io) {
    constexpr int Wd = static_cast<int>(Batch::size);
    for (; i + Wd - 1 <= istop; i += Wd) {
        const int ig = i + cube2grid;
        Batch r0(0.0), r1(0.0), r2(0.0), r3(0.0);
#pragma GCC unroll 10
        for (int lz = 0; lz <= LpZ; ++lz) {
            const Batch p =
                Batch::load_unaligned(pol_i + lz * pol_len + (i + cmax));
            r0 = xsimd::fma(p, Batch(cx[lz * 4 + 0]), r0);
            r1 = xsimd::fma(p, Batch(cx[lz * 4 + 1]), r1);
            r2 = xsimd::fma(p, Batch(cx[lz * 4 + 2]), r2);
            r3 = xsimd::fma(p, Batch(cx[lz * 4 + 3]), r3);
        }
        (Batch::load_unaligned(grid_io + base0 + ig) + r0)
            .store_unaligned(grid_io + base0 + ig);
        (Batch::load_unaligned(grid_io + base1 + ig) + r1)
            .store_unaligned(grid_io + base1 + ig);
        (Batch::load_unaligned(grid_io + base2 + ig) + r2)
            .store_unaligned(grid_io + base2 + ig);
        (Batch::load_unaligned(grid_io + base3 + ig) + r3)
            .store_unaligned(grid_io + base3 + ig);
    }
}

template <bool Enable, int LpZ>
inline void collocate_cx_4wide_run(int& i, int istop, int cmax, int pol_len,
                                   const double* pol_i, const double* cx,
                                   int cube2grid, long base0, long base1,
                                   long base2, long base3, double* grid_io) {
#if defined(__AVX512F__)
    if constexpr (Enable) {
        using batch4_t = xsimd::make_sized_batch_t<double, 4>;
        collocate_cx_vec_pass<batch4_t, LpZ>(i, istop, cmax, pol_len, pol_i, cx,
                                             cube2grid, base0, base1, base2,
                                             base3, grid_io);
    }
#else
    (void)i;
    (void)istop;
    (void)cmax;
    (void)pol_len;
    (void)pol_i;
    (void)cx;
    (void)cube2grid;
    (void)base0;
    (void)base1;
    (void)base2;
    (void)base3;
    (void)grid_io;
#endif
}

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
        collocate_cx_vec_pass<batch_t, LpZ>(i, istop, cmax, pol_len, pol_i, cx,
                                            cube2grid, base0, base1, base2,
                                            base3, grid_io);
        collocate_cx_4wide_run<kUseBatch4Residual, LpZ>(
            i, istop, cmax, pol_len, pol_i, cx, cube2grid, base0, base1, base2,
            base3, grid_io);
        for (; i <= istop; ++i) {
            const int ig = i + cube2grid;
            double reg0 = 0.0, reg1 = 0.0, reg2 = 0.0, reg3 = 0.0;
#pragma GCC unroll 10
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

template <bool Enable>
inline void collocate_cx_4wide_dyn_run(int& i, int istop, int lp_z, int cmax,
                                       int pol_len, const double* pol_i,
                                       const double* cx, int cube2grid,
                                       long base0, long base1, long base2,
                                       long base3, double* grid_io) {
#if defined(__AVX512F__)
    if constexpr (Enable) {
        using batch4_t = xsimd::make_sized_batch_t<double, 4>;
        for (; i + 3 <= istop; i += 4) {
            const int ig = i + cube2grid;
            batch4_t r0(0.0), r1(0.0), r2(0.0), r3(0.0);
            for (int lz = 0; lz <= lp_z; ++lz) {
                const batch4_t p = batch4_t::load_unaligned(
                    pol_i + lz * pol_len + (i + cmax));
                r0 = xsimd::fma(p, batch4_t(cx[lz * 4 + 0]), r0);
                r1 = xsimd::fma(p, batch4_t(cx[lz * 4 + 1]), r1);
                r2 = xsimd::fma(p, batch4_t(cx[lz * 4 + 2]), r2);
                r3 = xsimd::fma(p, batch4_t(cx[lz * 4 + 3]), r3);
            }
            (batch4_t::load_unaligned(grid_io + base0 + ig) + r0)
                .store_unaligned(grid_io + base0 + ig);
            (batch4_t::load_unaligned(grid_io + base1 + ig) + r1)
                .store_unaligned(grid_io + base1 + ig);
            (batch4_t::load_unaligned(grid_io + base2 + ig) + r2)
                .store_unaligned(grid_io + base2 + ig);
            (batch4_t::load_unaligned(grid_io + base3 + ig) + r3)
                .store_unaligned(grid_io + base3 + ig);
        }
    }
#else
    (void)i;
    (void)istop;
    (void)lp_z;
    (void)cmax;
    (void)pol_len;
    (void)pol_i;
    (void)cx;
    (void)cube2grid;
    (void)base0;
    (void)base1;
    (void)base2;
    (void)base3;
    (void)grid_io;
#endif
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
                r0 = xsimd::fma(p, batch_t(cx[lz * 4 + 0]), r0);
                r1 = xsimd::fma(p, batch_t(cx[lz * 4 + 1]), r1);
                r2 = xsimd::fma(p, batch_t(cx[lz * 4 + 2]), r2);
                r3 = xsimd::fma(p, batch_t(cx[lz * 4 + 3]), r3);
            }
            (batch_t::load_unaligned(grid_io + base0 + ig) + r0)
                .store_unaligned(grid_io + base0 + ig);
            (batch_t::load_unaligned(grid_io + base1 + ig) + r1)
                .store_unaligned(grid_io + base1 + ig);
            (batch_t::load_unaligned(grid_io + base2 + ig) + r2)
                .store_unaligned(grid_io + base2 + ig);
            (batch_t::load_unaligned(grid_io + base3 + ig) + r3)
                .store_unaligned(grid_io + base3 + ig);
        }
        collocate_cx_4wide_dyn_run<kUseBatch4Residual>(
            i, istop, lp_z, cmax, pol_len, pol_i, cx, cube2grid, base0, base1,
            base2, base3, grid_io);
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
    case 7:
        collocate_cx_to_grid_lp<7>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 8:
        collocate_cx_to_grid_lp<8>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
                                   base0, base1, base2, base3, grid_io);
        break;
    case 9:
        collocate_cx_to_grid_lp<9>(lb, ub, cmax, pol_len, pol_i, cx, map0, sec0,
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
template <typename Batch, int LpZ>
inline void integrate_cx_vec_pass(int& i, int istop, int cmax, int pol_len,
                                  const double* pol_i, Batch* a0, Batch* a1,
                                  Batch* a2, Batch* a3, int cube2grid,
                                  long base0, long base1, long base2,
                                  long base3, const double* grid_io) {
    constexpr int Wd = static_cast<int>(Batch::size);
    for (; i + Wd - 1 <= istop; i += Wd) {
        const int ig = i + cube2grid;
        const Batch v0 = Batch::load_unaligned(grid_io + base0 + ig);
        const Batch v1 = Batch::load_unaligned(grid_io + base1 + ig);
        const Batch v2 = Batch::load_unaligned(grid_io + base2 + ig);
        const Batch v3 = Batch::load_unaligned(grid_io + base3 + ig);
#pragma GCC unroll 10
        for (int lz = 0; lz <= LpZ; ++lz) {
            const Batch p =
                Batch::load_unaligned(pol_i + lz * pol_len + (i + cmax));
            a0[lz] = xsimd::fma(v0, p, a0[lz]);
            a1[lz] = xsimd::fma(v1, p, a1[lz]);
            a2[lz] = xsimd::fma(v2, p, a2[lz]);
            a3[lz] = xsimd::fma(v3, p, a3[lz]);
        }
    }
}

template <bool Enable, int LpZ>
inline void integrate_cx_4wide_run(int& i, int istop, int cmax, int pol_len,
                                   const double* pol_i, double* cx,
                                   int cube2grid, long base0, long base1,
                                   long base2, long base3,
                                   const double* grid_io) {
#if defined(__AVX512F__)
    if constexpr (Enable) {
        using batch4_t = xsimd::make_sized_batch_t<double, 4>;
        batch4_t b0[LpZ + 1], b1[LpZ + 1], b2[LpZ + 1], b3[LpZ + 1];
#pragma GCC unroll 10
        for (int lz = 0; lz <= LpZ; ++lz) {
            b0[lz] = batch4_t(0.0);
            b1[lz] = batch4_t(0.0);
            b2[lz] = batch4_t(0.0);
            b3[lz] = batch4_t(0.0);
        }
        integrate_cx_vec_pass<batch4_t, LpZ>(i, istop, cmax, pol_len, pol_i, b0,
                                             b1, b2, b3, cube2grid, base0, base1,
                                             base2, base3, grid_io);
#pragma GCC unroll 10
        for (int lz = 0; lz <= LpZ; ++lz) {
            cx[lz * 4 + 0] += xsimd::reduce_add(b0[lz]);
            cx[lz * 4 + 1] += xsimd::reduce_add(b1[lz]);
            cx[lz * 4 + 2] += xsimd::reduce_add(b2[lz]);
            cx[lz * 4 + 3] += xsimd::reduce_add(b3[lz]);
        }
    }
#else
    (void)i;
    (void)istop;
    (void)cmax;
    (void)pol_len;
    (void)pol_i;
    (void)cx;
    (void)cube2grid;
    (void)base0;
    (void)base1;
    (void)base2;
    (void)base3;
    (void)grid_io;
#endif
}

template <int LpZ>
inline void integrate_cx_scalar_run(int& i, int istop, int cmax, int pol_len,
                                    const double* pol_i, double* cx,
                                    int cube2grid, long base0, long base1,
                                    long base2, long base3,
                                    const double* grid_io) {
    for (; i <= istop; ++i) {
        const int ig = i + cube2grid;
        const double v0 = grid_io[base0 + ig];
        const double v1 = grid_io[base1 + ig];
        const double v2 = grid_io[base2 + ig];
        const double v3 = grid_io[base3 + ig];
#pragma GCC unroll 10
        for (int lz = 0; lz <= LpZ; ++lz) {
            const double p = pol_i[lz * pol_len + (i + cmax)];
            cx[lz * 4 + 0] += v0 * p;
            cx[lz * 4 + 1] += v1 * p;
            cx[lz * 4 + 2] += v2 * p;
            cx[lz * 4 + 3] += v3 * p;
        }
    }
}

// Integrate prefers width-4 accumulate (CP2K AVX2 idiom). On AVX-512 the
// native width-8 path needs 4*(LpZ+1) ZMMs — spills for LpZ≳3 and loses to
// 4-wide with half the live vectors + shorter segments still vectorized.
template <int LpZ>
inline void integrate_cx_from_grid_lp(int lb, int ub, int cmax, int pol_len,
                                      const double* pol_i, double* cx,
                                      const int* map0, const int* sec0,
                                      long base0, long base1, long base2,
                                      long base3, const double* grid_io) {
#if defined(__AVX512F__)
    // Width-4 primary on AVX-512 hosts (kUseBatch4Residual implies batch4 ok).
    constexpr int Wd = 4;
#else
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
#endif
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        const int len = istop - istart + 1;
        // Short sphere segments: scalar only (no batch zero+reduce).
        if (len < Wd) {
            integrate_cx_scalar_run<LpZ>(i, istop, cmax, pol_len, pol_i, cx,
                                         cube2grid, base0, base1, base2, base3,
                                         grid_io);
            istart = istop + 1;
            continue;
        }
#if defined(__AVX512F__)
        // Always-on 4-wide segment accumulate (not residual-only).
        {
            using batch4_t = xsimd::make_sized_batch_t<double, 4>;
            batch4_t a0[LpZ + 1], a1[LpZ + 1], a2[LpZ + 1], a3[LpZ + 1];
#pragma GCC unroll 10
            for (int lz = 0; lz <= LpZ; ++lz) {
                a0[lz] = batch4_t(0.0);
                a1[lz] = batch4_t(0.0);
                a2[lz] = batch4_t(0.0);
                a3[lz] = batch4_t(0.0);
            }
            integrate_cx_vec_pass<batch4_t, LpZ>(
                i, istop, cmax, pol_len, pol_i, a0, a1, a2, a3, cube2grid,
                base0, base1, base2, base3, grid_io);
#pragma GCC unroll 10
            for (int lz = 0; lz <= LpZ; ++lz) {
                cx[lz * 4 + 0] += xsimd::reduce_add(a0[lz]);
                cx[lz * 4 + 1] += xsimd::reduce_add(a1[lz]);
                cx[lz * 4 + 2] += xsimd::reduce_add(a2[lz]);
                cx[lz * 4 + 3] += xsimd::reduce_add(a3[lz]);
            }
        }
#else
        {
            using batch_t = xsimd::batch<double>;
            batch_t a0[LpZ + 1], a1[LpZ + 1], a2[LpZ + 1], a3[LpZ + 1];
#pragma GCC unroll 10
            for (int lz = 0; lz <= LpZ; ++lz) {
                a0[lz] = batch_t(0.0);
                a1[lz] = batch_t(0.0);
                a2[lz] = batch_t(0.0);
                a3[lz] = batch_t(0.0);
            }
            integrate_cx_vec_pass<batch_t, LpZ>(
                i, istop, cmax, pol_len, pol_i, a0, a1, a2, a3, cube2grid,
                base0, base1, base2, base3, grid_io);
#pragma GCC unroll 10
            for (int lz = 0; lz <= LpZ; ++lz) {
                cx[lz * 4 + 0] += xsimd::reduce_add(a0[lz]);
                cx[lz * 4 + 1] += xsimd::reduce_add(a1[lz]);
                cx[lz * 4 + 2] += xsimd::reduce_add(a2[lz]);
                cx[lz * 4 + 3] += xsimd::reduce_add(a3[lz]);
            }
        }
#endif
        integrate_cx_scalar_run<LpZ>(i, istop, cmax, pol_len, pol_i, cx,
                                     cube2grid, base0, base1, base2, base3,
                                     grid_io);
        istart = istop + 1;
    }
}

template <bool Enable>
inline void integrate_cx_4wide_dyn_run(int& i, int istop, int lp_z, int cmax,
                                       int pol_len, const double* pol_i,
                                       double* cx, int cube2grid, long base0,
                                       long base1, long base2, long base3,
                                       const double* grid_io) {
#if defined(__AVX512F__)
    if constexpr (Enable) {
        using batch4_t = xsimd::make_sized_batch_t<double, 4>;
        constexpr int kDynMax = 24;
        batch4_t b0[kDynMax + 1], b1[kDynMax + 1], b2[kDynMax + 1],
            b3[kDynMax + 1];
        for (int lz = 0; lz <= lp_z; ++lz) {
            b0[lz] = batch4_t(0.0);
            b1[lz] = batch4_t(0.0);
            b2[lz] = batch4_t(0.0);
            b3[lz] = batch4_t(0.0);
        }
        for (; i + 3 <= istop; i += 4) {
            const int ig = i + cube2grid;
            const batch4_t v0 =
                batch4_t::load_unaligned(grid_io + base0 + ig);
            const batch4_t v1 =
                batch4_t::load_unaligned(grid_io + base1 + ig);
            const batch4_t v2 =
                batch4_t::load_unaligned(grid_io + base2 + ig);
            const batch4_t v3 =
                batch4_t::load_unaligned(grid_io + base3 + ig);
            for (int lz = 0; lz <= lp_z; ++lz) {
                const batch4_t p = batch4_t::load_unaligned(
                    pol_i + lz * pol_len + (i + cmax));
                b0[lz] = xsimd::fma(v0, p, b0[lz]);
                b1[lz] = xsimd::fma(v1, p, b1[lz]);
                b2[lz] = xsimd::fma(v2, p, b2[lz]);
                b3[lz] = xsimd::fma(v3, p, b3[lz]);
            }
        }
        for (int lz = 0; lz <= lp_z; ++lz) {
            cx[lz * 4 + 0] += xsimd::reduce_add(b0[lz]);
            cx[lz * 4 + 1] += xsimd::reduce_add(b1[lz]);
            cx[lz * 4 + 2] += xsimd::reduce_add(b2[lz]);
            cx[lz * 4 + 3] += xsimd::reduce_add(b3[lz]);
        }
    }
#else
    (void)i;
    (void)istop;
    (void)lp_z;
    (void)cmax;
    (void)pol_len;
    (void)pol_i;
    (void)cx;
    (void)cube2grid;
    (void)base0;
    (void)base1;
    (void)base2;
    (void)base3;
    (void)grid_io;
#endif
}

inline void integrate_cx_from_grid_dyn(int lp_z, int lb, int ub, int cmax,
                                       int pol_len, const double* pol_i,
                                       double* cx, const int* map0,
                                       const int* sec0, long base0, long base1,
                                       long base2, long base3,
                                       const double* grid_io) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    // Segment-level accumulators (same shape as lp-specialized path).
    constexpr int kDynMax = 24;
    for (int istart = lb; istart <= ub;) {
        const int istop = imin(ub, istart + sec0[istart + cmax]);
        const int cube2grid = map0[istart + cmax] - istart;
        int i = istart;
        const int len = istop - istart + 1;
        if (len < Wd) {
            // Short segment: 4-wide residual (AVX-512) + scalar, no batch zero.
            if (lp_z <= kDynMax) {
                integrate_cx_4wide_dyn_run<kUseBatch4Residual>(
                    i, istop, lp_z, cmax, pol_len, pol_i, cx, cube2grid, base0,
                    base1, base2, base3, grid_io);
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
            continue;
        }
        if (lp_z <= kDynMax) {
            batch_t a0[kDynMax + 1], a1[kDynMax + 1], a2[kDynMax + 1],
                a3[kDynMax + 1];
            for (int lz = 0; lz <= lp_z; ++lz) {
                a0[lz] = batch_t(0.0);
                a1[lz] = batch_t(0.0);
                a2[lz] = batch_t(0.0);
                a3[lz] = batch_t(0.0);
            }
            for (; i + Wd - 1 <= istop; i += Wd) {
                const int ig = i + cube2grid;
                const batch_t v0 =
                    batch_t::load_unaligned(grid_io + base0 + ig);
                const batch_t v1 =
                    batch_t::load_unaligned(grid_io + base1 + ig);
                const batch_t v2 =
                    batch_t::load_unaligned(grid_io + base2 + ig);
                const batch_t v3 =
                    batch_t::load_unaligned(grid_io + base3 + ig);
                for (int lz = 0; lz <= lp_z; ++lz) {
                    const batch_t p = batch_t::load_unaligned(
                        pol_i + lz * pol_len + (i + cmax));
                    a0[lz] = xsimd::fma(v0, p, a0[lz]);
                    a1[lz] = xsimd::fma(v1, p, a1[lz]);
                    a2[lz] = xsimd::fma(v2, p, a2[lz]);
                    a3[lz] = xsimd::fma(v3, p, a3[lz]);
                }
            }
            for (int lz = 0; lz <= lp_z; ++lz) {
                cx[lz * 4 + 0] += xsimd::reduce_add(a0[lz]);
                cx[lz * 4 + 1] += xsimd::reduce_add(a1[lz]);
                cx[lz * 4 + 2] += xsimd::reduce_add(a2[lz]);
                cx[lz * 4 + 3] += xsimd::reduce_add(a3[lz]);
            }
            integrate_cx_4wide_dyn_run<kUseBatch4Residual>(
                i, istop, lp_z, cmax, pol_len, pol_i, cx, cube2grid, base0,
                base1, base2, base3, grid_io);
        } else {
            for (; i + Wd - 1 <= istop; i += Wd) {
                const int ig = i + cube2grid;
                const batch_t v0 =
                    batch_t::load_unaligned(grid_io + base0 + ig);
                const batch_t v1 =
                    batch_t::load_unaligned(grid_io + base1 + ig);
                const batch_t v2 =
                    batch_t::load_unaligned(grid_io + base2 + ig);
                const batch_t v3 =
                    batch_t::load_unaligned(grid_io + base3 + ig);
                for (int lz = 0; lz <= lp_z; ++lz) {
                    const batch_t p = batch_t::load_unaligned(
                        pol_i + lz * pol_len + (i + cmax));
                    cx[lz * 4 + 0] += xsimd::reduce_add(v0 * p);
                    cx[lz * 4 + 1] += xsimd::reduce_add(v1 * p);
                    cx[lz * 4 + 2] += xsimd::reduce_add(v2 * p);
                    cx[lz * 4 + 3] += xsimd::reduce_add(v3 * p);
                }
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
    case 7:
        integrate_cx_from_grid_lp<7>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 8:
        integrate_cx_from_grid_lp<8>(lb, ub, cmax, pol_len, pol_i, cx, map0,
                                     sec0, base0, base1, base2, base3, grid_io);
        break;
    case 9:
        integrate_cx_from_grid_lp<9>(lb, ub, cmax, pol_len, pol_i, cx, map0,
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
                        int Ltot, double* cxyz, double* grid_io) {
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

    // Angular monoms with lx+ly+lz > Ltot are structurally zero (each product
    // monom has total L ≤ Ltot). Skip them in rectangular walks — CP2K uses
    // triangular packing for the same reason.
    const int Lcap = (Ltot >= 0) ? Ltot : (lp_x + lp_y + lp_z);

    for (int k1 = kstart; k1 <= 0; ++k1) {
        const int k2 = 1 - k1;
        const int kg1 = map2[k1 + cmax];
        const int kg2 = map2[k2 + cmax];
        std::memset(cxy, 0, static_cast<std::size_t>(cxy_n) * sizeof(double));

        if constexpr (Collocate) {
            for (int lx = 0; lx <= lp_x; ++lx) {
                const double p1 = pol_k[lx * pol_len + (k1 + cmax)];
                const double p2 = pol_k[lx * pol_len + (k2 + cmax)];
                if (p1 == 0.0 && p2 == 0.0)
                    continue;
                const int ly_max = imin(lp_y, Lcap - lx);
                for (int ly = 0; ly <= ly_max; ++ly) {
                    const int lz_max = imin(lp_z, Lcap - lx - ly);
                    for (int lz = 0; lz <= lz_max; ++lz) {
                        const double c = cxyz[(lx * sy + ly) * sz + lz];
                        if (c == 0.0)
                            continue;
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxy[cxy_i + 0] += c * p1;
                        cxy[cxy_i + 1] += c * p2;
                    }
                }
            }
        }

        const int jstart = *sb++;
        for (int j1 = jstart; j1 <= 0; ++j1) {
            const int j2 = 1 - j1;
            const int jg1 = map1[j1 + cmax];
            const int jg2 = map1[j2 + cmax];
            std::memset(cx, 0, static_cast<std::size_t>(cx_n) * sizeof(double));

            if constexpr (Collocate) {
                for (int ly = 0; ly <= lp_y; ++ly) {
                    const double p1 = pol_j[ly * pol_len + (j1 + cmax)];
                    const double p2 = pol_j[ly * pol_len + (j2 + cmax)];
                    if (p1 == 0.0 && p2 == 0.0)
                        continue;
                    const int lz_max = imin(lp_z, Lcap - ly);
                    for (int lz = 0; lz <= lz_max; ++lz) {
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
                // Reverse cxy ← cx: skip zero pol (collocate already does).
                for (int ly = 0; ly <= lp_y; ++ly) {
                    const double p1 = pol_j[ly * pol_len + (j1 + cmax)];
                    const double p2 = pol_j[ly * pol_len + (j2 + cmax)];
                    if (p1 == 0.0 && p2 == 0.0)
                        continue;
                    const int lz_max = imin(lp_z, Lcap - ly);
                    for (int lz = 0; lz <= lz_max; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxy[cxy_i + 0] +=
                            cx[lz * 4 + 0] * p1 + cx[lz * 4 + 2] * p2;
                        cxy[cxy_i + 1] +=
                            cx[lz * 4 + 1] * p1 + cx[lz * 4 + 3] * p2;
                    }
                }
            }
        }

        if constexpr (!Collocate) {
            // Reverse cxyz ← cxy with Ltot triangular skip + zero pol.
            for (int lx = 0; lx <= lp_x; ++lx) {
                const double p1 = pol_k[lx * pol_len + (k1 + cmax)];
                const double p2 = pol_k[lx * pol_len + (k2 + cmax)];
                if (p1 == 0.0 && p2 == 0.0)
                    continue;
                const int ly_max = imin(lp_y, Lcap - lx);
                for (int ly = 0; ly <= ly_max; ++ly) {
                    const int lz_max = imin(lp_z, Lcap - lx - ly);
                    for (int lz = 0; lz <= lz_max; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxyz[(lx * sy + ly) * sz + lz] +=
                            cxy[cxy_i + 0] * p1 + cxy[cxy_i + 1] * p2;
                    }
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
    // mag-based radius is a conservative outer bound; after cab expand, use
    // min(mag, cmax) so we only shrink (never enlarge) the sphere.
    const double r_mag = shape_radius(mag, prods[0].p, Ltot, eps);
    if (!(r_mag > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* cxyz = g_I.data();
    std::memset(cxyz, 0, static_cast<std::size_t>(cxyz_n) * sizeof(double));
    add_group_to_cxyz(prods, weights, n, lpx, lpy, lpz, cxyz);
    const double r_c = cmax_radius(cxyz, cxyz_n, prods[0].p, Ltot, eps);
    const double radius = (r_c > 0.0) ? std::min(r_mag, r_c) : r_mag;
    if (!(radius > 0.0))
        return;
    ortho_cxyz_to_grid<true>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                             lpz, Ltot, cxyz, rho);
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
    const double r_mag = shape_radius(mag, g0.p, Ltot, eps);
    if (!(r_mag > 0.0))
        return;
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    ensure_I(static_cast<std::size_t>(cxyz_n));
    double* cxyz = g_I.data();
    std::memset(cxyz, 0, static_cast<std::size_t>(cxyz_n) * sizeof(double));
    // Geometry-shared cab expand without packing GridProduct (MAIN avg_g ~85).
    if (n >= 4) {
        int max_la[3] = {0, 0, 0}, max_lb[3] = {0, 0, 0};
        for (int i = 0; i < n; ++i) {
            const auto& p = prods_base[idx[i]];
            for (int d = 0; d < 3; ++d) {
                max_la[d] = imax(max_la[d], p.la[d]);
                max_lb[d] = imax(max_lb[d], p.lb[d]);
            }
        }
        const int mla = imax(imax(max_la[0], max_la[1]), max_la[2]);
        const int mlb = imax(imax(max_lb[0], max_lb[1]), max_lb[2]);
        const int mlp = mla + mlb;
        if (mla <= kMaxL && mlb <= kMaxL && mlp <= kMaxL) {
            thread_local std::vector<double> alpha_buf;
            const int stride_ls = mlp + 1;
            const int stride_la = (mla + 1) * stride_ls;
            const int stride_lb = (mlb + 1) * stride_la;
            const int need = 3 * stride_lb;
            if (static_cast<int>(alpha_buf.size()) < need)
                alpha_buf.assign(static_cast<std::size_t>(need), 0.0);
            else
                std::memset(alpha_buf.data(), 0,
                            static_cast<std::size_t>(need) * sizeof(double));
            const double* pc = g0.pc;
            const double* ra = g0.ra;
            const double* rb = g0.rb;
            for (int d = 0; d < 3; ++d) {
                double* ad = alpha_buf.data() + d * stride_lb;
                const double drpa = pc[d] - ra[d];
                const double drpb = pc[d] - rb[d];
                for (int lxa = 0; lxa <= max_la[d]; ++lxa) {
                    for (int lxb = 0; lxb <= max_lb[d]; ++lxb) {
                        double binomial_k_lxa = 1.0;
                        double a = 1.0;
                        for (int k = 0; k <= lxa; ++k) {
                            double binomial_l_lxb = 1.0;
                            double b = 1.0;
                            for (int l = 0; l <= lxb; ++l) {
                                const int ls = lxa - k + lxb - l;
                                ad[(lxb * (mla + 1) + lxa) * stride_ls + ls] +=
                                    binomial_k_lxa * binomial_l_lxb * a * b;
                                binomial_l_lxb *= static_cast<double>(lxb - l)
                                                  / static_cast<double>(l + 1);
                                b *= drpb;
                            }
                            binomial_k_lxa *= static_cast<double>(lxa - k)
                                              / static_cast<double>(k + 1);
                            a *= drpa;
                        }
                    }
                }
            }
            const double* ax = alpha_buf.data();
            const double* ay = alpha_buf.data() + stride_lb;
            const double* az = alpha_buf.data() + 2 * stride_lb;
            for (int i = 0; i < n; ++i) {
                const double w = weights[i];
                if (w == 0.0)
                    continue;
                const auto& p = prods_base[idx[i]];
                const int lax = p.la[0], lay = p.la[1], laz = p.la[2];
                const int lbx = p.lb[0], lby = p.lb[1], lbz = p.lb[2];
                const double* ax_row =
                    ax + (lbx * (mla + 1) + lax) * stride_ls;
                const double* ay_row =
                    ay + (lby * (mla + 1) + lay) * stride_ls;
                const double* az_row =
                    az + (lbz * (mla + 1) + laz) * stride_ls;
                const int lpx_p = lax + lbx, lpy_p = lay + lby,
                          lpz_p = laz + lbz;
                for (int lx = 0; lx <= lpx_p; ++lx) {
                    if (ax_row[lx] == 0.0)
                        continue;
                    for (int ly = 0; ly <= lpy_p; ++ly) {
                        const double axy = w * ax_row[lx] * ay_row[ly];
                        if (axy == 0.0)
                            continue;
                        double* row = cxyz + (lx * sy + ly) * sz;
                        for (int lz = 0; lz <= lpz_p; ++lz)
                            row[lz] += axy * az_row[lz];
                    }
                }
            }
        } else {
            for (int i = 0; i < n; ++i)
                add_product_to_cxyz(prods_base[idx[i]], weights[i], lpx, lpy,
                                    lpz, cxyz);
        }
    } else {
        for (int i = 0; i < n; ++i)
            add_product_to_cxyz(prods_base[idx[i]], weights[i], lpx, lpy, lpz,
                                cxyz);
    }
    const double r_c = cmax_radius(cxyz, cxyz_n, g0.p, Ltot, eps);
    const double radius = (r_c > 0.0) ? std::min(r_mag, r_c) : r_mag;
    if (!(radius > 0.0))
        return;
    ortho_cxyz_to_grid<true>(grid, g0.pc, g0.p, radius, lpx, lpy, lpz, Ltot,
                             cxyz, rho);
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
    std::memset(I, 0, static_cast<std::size_t>(cxyz_n) * sizeof(double));
    ortho_cxyz_to_grid<false>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                              lpz, Ltot, I, const_cast<double*>(V));
    contract_group_from_I(prods, n, lpx, lpy, lpz, I, out_acc);
}

void collint_integrate_group_idx(const UniformGrid& grid,
                                 const GridProduct* prods_base, const int* idx,
                                 int n, double mag, double eps, const double* V,
                                 double* out_acc) {
    for (int i = 0; i < n; ++i)
        out_acc[i] = 0.0;
    if (n <= 0 || !(mag > 0.0))
        return;
    // Contiguous-path preferred for shell tasks; idx path packs a temp group
    // only when the caller still needs sparse selection.
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
    std::memset(I, 0, static_cast<std::size_t>(cxyz_n) * sizeof(double));
    ortho_cxyz_to_grid<false>(grid, g0.pc, g0.p, radius, lpx, lpy, lpz, Ltot, I,
                              const_cast<double*>(V));
    // Gather monoms into a temp contiguous buffer so the cab-style contract
    // can share one alpha table (MAIN-class shell tasks).
    thread_local std::vector<GridProduct> packed;
    if (static_cast<int>(packed.size()) < n)
        packed.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        packed[static_cast<std::size_t>(i)] = prods_base[idx[i]];
    contract_group_from_I(packed.data(), n, lpx, lpy, lpz, I, out_acc);
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
