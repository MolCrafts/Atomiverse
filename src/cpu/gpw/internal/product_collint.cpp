/// @file src/cpu/gpw/internal/product_collint.cpp
/// @brief CP2K collint-style product collocate/integrate (see product_collint.h).
///
/// Structure and recurrence follow CP2K grid_cpu_collint.h (BSD-3-Clause).
/// SIMD uses portable xsimd (project law — no arch intrinsics).
/// Grow-only thread_local scratch: product loops fire millions of times/SCF.

#include "atomiverse/cpu/gpw/internal/product_collint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {
namespace {

constexpr int kMaxL = 12; // per-axis la+lb headroom (DZVP stays well below)

// Grow-only per-thread scratch (mirrors multigrid collocate_one pools).
thread_local std::vector<double> g_pol;   // pol packs + cxyz + cyz + cz + Fz
thread_local std::vector<long> g_wrap;    // wrap indices per axis
thread_local std::vector<double> g_e2;    // E2 per axis for sphere cull
thread_local std::vector<double> g_I;     // monom table (integrate)
thread_local std::vector<double> g_gaus;  // temp gaussian 1D

inline void ensure_d(std::size_t n) {
    if (g_pol.size() < n)
        g_pol.resize(n);
}
inline void ensure_w(std::size_t n) {
    if (g_wrap.size() < n)
        g_wrap.resize(n);
}
inline void ensure_e2(std::size_t n) {
    if (g_e2.size() < n)
        g_e2.resize(n);
}
inline void ensure_I(std::size_t n) {
    if (g_I.size() < n)
        g_I.resize(n);
}
inline void ensure_gaus(std::size_t n) {
    if (g_gaus.size() < n)
        g_gaus.resize(n);
}

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

/// Expand (x−ra)^la (x−rb)^lb = Σ_l alpha[l] (x−pc)^l.
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

struct AxisGeom {
    long lo[3];
    int len[3];
    int lpx, lpy, lpz;
    int pol_off[3];
    int wrap_off[3];
    int e2_off[3];
    int pol_total;
    int wrap_total;
    int e2_total;
    double rc2;
    double p;
    double pc[3];
    double origin[3];
    double L[3];
    long n[3];
};

/// Build window + pol + wrap + E2 for max angular orders. Returns false if empty.
bool prepare_pol(const UniformGrid& grid, const GridProduct& geom, int lpx,
                 int lpy, int lpz, double mag, int Ltot, double eps,
                 AxisGeom& ag) {
    if (lpx > kMaxL || lpy > kMaxL || lpz > kMaxL)
        return false;
    const double rc = shape_radius(mag, geom.p, Ltot, eps);
    if (!(rc > 0.0))
        return false;

    ag.p = geom.p;
    ag.rc2 = rc * rc;
    for (int d = 0; d < 3; ++d) {
        ag.n[d] = grid.n[d];
        ag.L[d] = grid.L[d];
        ag.origin[d] = grid.origin[d];
        ag.pc[d] = geom.pc[d];
        const double dx = grid.L[d] / static_cast<double>(grid.n[d]);
        const long ih = static_cast<long>(std::ceil(rc / dx));
        const long ic = static_cast<long>(std::floor(geom.pc[d] / dx));
        ag.lo[d] = ic - ih;
        ag.len[d] = static_cast<int>(2 * ih + 1);
        if (ag.len[d] <= 0)
            return false;
    }
    ag.lpx = lpx;
    ag.lpy = lpy;
    ag.lpz = lpz;

    int po = 0, wo = 0, eo = 0;
    const int lp[3] = {lpx, lpy, lpz};
    for (int d = 0; d < 3; ++d) {
        ag.pol_off[d] = po;
        ag.wrap_off[d] = wo;
        ag.e2_off[d] = eo;
        po += (lp[d] + 1) * ag.len[d];
        wo += ag.len[d];
        eo += ag.len[d];
    }
    ag.pol_total = po;
    ag.wrap_total = wo;
    ag.e2_total = eo;
    // Scratch: pol pack + cxyz + cyz + cz + Fz (reuse tail of g_pol)
    const int sx = lpx + 1, sy = lpy + 1, sz = lpz + 1;
    const int extra = sx * sy * sz + sy * sz + sz + ag.len[2];
    ensure_d(static_cast<std::size_t>(po + extra));
    ensure_w(static_cast<std::size_t>(wo));
    ensure_e2(static_cast<std::size_t>(eo));
    ensure_gaus(static_cast<std::size_t>(
        std::max({ag.len[0], ag.len[1], ag.len[2]})));

    double* pol_base = g_pol.data();
    long* wrap_base = g_wrap.data();
    double* e2_base = g_e2.data();
    double* gaus = g_gaus.data();

    for (int d = 0; d < 3; ++d) {
        const int lpd = lp[d];
        double* pol = pol_base + ag.pol_off[d];
        long* W = wrap_base + ag.wrap_off[d];
        double* E2 = e2_base + ag.e2_off[d];
        const double dxd = ag.L[d] / static_cast<double>(ag.n[d]);
        const double A = std::exp(-ag.p * dxd * dxd);
        const double A2 = A * A;
        const int tc = (ag.len[d] - 1) / 2;
        const double rc_ =
            ag.origin[d]
            + ag.L[d] * static_cast<double>(ag.lo[d] + tc)
                  / static_cast<double>(ag.n[d]);
        const double ec = rc_ - ag.pc[d];
        // CP2K outward exp recurrence from centre.
        gaus[static_cast<std::size_t>(tc)] = std::exp(-ag.p * ec * ec);
        {
            double g = gaus[static_cast<std::size_t>(tc)];
            double B = std::exp(-2.0 * ag.p * ec * dxd);
            for (int t = tc + 1; t < ag.len[d]; ++t) {
                g *= B * A;
                gaus[static_cast<std::size_t>(t)] = g;
                B *= A2;
            }
        }
        {
            double g = gaus[static_cast<std::size_t>(tc)];
            double C = std::exp(2.0 * ag.p * ec * dxd);
            for (int t = tc - 1; t >= 0; --t) {
                g *= C * A;
                gaus[static_cast<std::size_t>(t)] = g;
                C *= A2;
            }
        }
        for (int t = 0; t < ag.len[d]; ++t) {
            const long idx = ag.lo[d] + t;
            const double r =
                ag.origin[d]
                + ag.L[d] * static_cast<double>(idx)
                      / static_cast<double>(ag.n[d]);
            const double e = r - ag.pc[d];
            E2[t] = e * e;
            double pg = gaus[static_cast<std::size_t>(t)];
            for (int l = 0; l <= lpd; ++l) {
                pol[l * ag.len[d] + t] = pg;
                pg *= e;
            }
            W[t] = ((idx % ag.n[d]) + ag.n[d]) % ag.n[d];
        }
    }
    return true;
}

inline double* cxyz_buf(const AxisGeom& ag) {
    return g_pol.data() + ag.pol_total;
}
inline double* cyz_buf(const AxisGeom& ag) {
    const int sx = ag.lpx + 1, sy = ag.lpy + 1, sz = ag.lpz + 1;
    return g_pol.data() + ag.pol_total + sx * sy * sz;
}
inline double* cz_buf(const AxisGeom& ag) {
    const int sx = ag.lpx + 1, sy = ag.lpy + 1, sz = ag.lpz + 1;
    return g_pol.data() + ag.pol_total + sx * sy * sz + sy * sz;
}
inline double* Fz_buf(const AxisGeom& ag) {
    const int sx = ag.lpx + 1, sy = ag.lpy + 1, sz = ag.lpz + 1;
    return g_pol.data() + ag.pol_total + sx * sy * sz + sy * sz + sz;
}

/// Contract cxyz → grid (collocate) or accumulate monoms from V (integrate).
template <bool Collocate>
double contract_cxyz(const AxisGeom& ag, double* cxyz, double* rho,
                     const double* V) {
    const int lpx = ag.lpx, lpy = ag.lpy, lpz = ag.lpz;
    const int sx = lpx + 1, sy = lpy + 1, sz = lpz + 1;
    const double* polx = g_pol.data() + ag.pol_off[0];
    const double* poly = g_pol.data() + ag.pol_off[1];
    const double* polz = g_pol.data() + ag.pol_off[2];
    const long* Wx = g_wrap.data() + ag.wrap_off[0];
    const long* Wy = g_wrap.data() + ag.wrap_off[1];
    const long* Wz = g_wrap.data() + ag.wrap_off[2];
    const double* E2x = g_e2.data() + ag.e2_off[0];
    const double* E2y = g_e2.data() + ag.e2_off[1];
    double* cyz = cyz_buf(ag);
    double* cz = cz_buf(ag);
    double* Fz = Fz_buf(ag);

    const double rc2 = ag.rc2;
    const double invdz =
        static_cast<double>(ag.n[2]) / ag.L[2];
    const double pc_z = ag.pc[2];
    const double origin_z = ag.origin[2];
    const long lo_z = ag.lo[2];
    const long n_z = ag.n[2];
    const int len_z = ag.len[2];
    const long n1 = ag.n[1];

    // Integrate monoms: accumulate into cxyz as I[lz,ly,lx]
    if constexpr (!Collocate)
        std::fill(cxyz, cxyz + sx * sy * sz, 0.0);

    double acc = 0.0;
    using batch_t = xsimd::batch<double>;
    constexpr long Wd = static_cast<long>(batch_t::size);

    for (int ti = 0; ti < ag.len[0]; ++ti) {
        if (E2x[ti] > rc2)
            continue;
        // cyz[lz,ly] = Σ_lx cxyz · polx  (collocate) / reverse (integrate)
        if constexpr (Collocate) {
            for (int lz = 0; lz <= lpz; ++lz)
                for (int ly = 0; ly <= lpy; ++ly) {
                    double s = 0.0;
                    for (int lx = 0; lx <= lpx; ++lx)
                        s += cxyz[(lz * sy + ly) * sx + lx]
                             * polx[lx * ag.len[0] + ti];
                    cyz[lz * sy + ly] = s;
                }
        } else {
            std::fill(cyz, cyz + sy * sz, 0.0);
        }
        const long ix = Wx[ti];
        for (int tj = 0; tj < ag.len[1]; ++tj) {
            const double r2xy = E2x[ti] + E2y[tj];
            if (r2xy > rc2)
                continue;
            if constexpr (Collocate) {
                for (int lz = 0; lz <= lpz; ++lz) {
                    double s = 0.0;
                    for (int ly = 0; ly <= lpy; ++ly)
                        s += cyz[lz * sy + ly] * poly[ly * ag.len[1] + tj];
                    cz[lz] = s;
                }
                for (int tk = 0; tk < len_z; ++tk) {
                    double s = 0.0;
                    for (int lz = 0; lz <= lpz; ++lz)
                        s += cz[lz] * polz[lz * len_z + tk];
                    Fz[tk] = s;
                }
            } else {
                std::fill(cz, cz + sz, 0.0);
            }

            const long base = (ix * n1 + Wy[tj]) * n_z;
            const double zr = std::sqrt(std::max(rc2 - r2xy, 0.0));
            long klo =
                static_cast<long>(std::ceil((pc_z - zr - origin_z) * invdz))
                - lo_z;
            long khi =
                static_cast<long>(std::floor((pc_z + zr - origin_z) * invdz))
                - lo_z;
            if (klo < 0)
                klo = 0;
            if (khi > len_z - 1)
                khi = len_z - 1;

            if constexpr (Collocate) {
                long tk = klo;
                while (tk <= khi) {
                    const long z0 = Wz[tk];
                    const long run = std::min(khi - tk + 1, n_z - z0);
                    const long g0 = base + z0;
                    long t = 0;
                    for (; t + Wd <= run; t += Wd) {
                        batch_t r = batch_t::load_unaligned(rho + g0 + t);
                        r += batch_t::load_unaligned(Fz + tk + t);
                        r.store_unaligned(rho + g0 + t);
                    }
                    for (; t < run; ++t)
                        rho[g0 + t] += Fz[tk + t];
                    tk += run;
                }
            } else {
                // Build Fz from V (z-run), then reverse-contract into monoms.
                std::fill(Fz, Fz + len_z, 0.0);
                long tk = klo;
                while (tk <= khi) {
                    const long z0 = Wz[tk];
                    const long run = std::min(khi - tk + 1, n_z - z0);
                    const long g0 = base + z0;
                    for (long t = 0; t < run; ++t)
                        Fz[tk + t] = V[g0 + t];
                    tk += run;
                }
                // cz[lz] = Σ_tk Fz[tk] * polz[lz,tk]
                for (int lz = 0; lz <= lpz; ++lz) {
                    double s = 0.0;
                    for (int tk = klo; tk <= khi; ++tk)
                        s += Fz[tk] * polz[lz * len_z + tk];
                    cz[lz] = s;
                }
                // cyz[lz,ly] += cz[lz] * poly[ly,tj]
                for (int lz = 0; lz <= lpz; ++lz)
                    for (int ly = 0; ly <= lpy; ++ly)
                        cyz[lz * sy + ly] += cz[lz] * poly[ly * ag.len[1] + tj];
            }
        }
        if constexpr (!Collocate) {
            // cxyz[lz,ly,lx] += cyz[lz,ly] * polx[lx,ti]
            for (int lz = 0; lz <= lpz; ++lz)
                for (int ly = 0; ly <= lpy; ++ly) {
                    const double cy = cyz[lz * sy + ly];
                    if (cy == 0.0)
                        continue;
                    for (int lx = 0; lx <= lpx; ++lx)
                        cxyz[(lz * sy + ly) * sx + lx] +=
                            cy * polx[lx * ag.len[0] + ti];
                }
        }
    }
    (void)acc;
    return 0.0;
}

void add_product_to_cxyz(const GridProduct& prod, double weight, int lpx,
                         int lpy, int lpz, double* cxyz) {
    if (weight == 0.0)
        return;
    double ax[kMaxL + 1], ay[kMaxL + 1], az[kMaxL + 1];
    const int lpx_p =
        expand_axis(prod.la[0], prod.lb[0], prod.pc[0], prod.ra[0], prod.rb[0],
                    ax);
    const int lpy_p =
        expand_axis(prod.la[1], prod.lb[1], prod.pc[1], prod.ra[1], prod.rb[1],
                    ay);
    const int lpz_p =
        expand_axis(prod.la[2], prod.lb[2], prod.pc[2], prod.ra[2], prod.rb[2],
                    az);
    const int sx = lpx + 1, sy = lpy + 1;
    for (int lz = 0; lz <= lpz_p; ++lz)
        for (int ly = 0; ly <= lpy_p; ++ly)
            for (int lx = 0; lx <= lpx_p; ++lx)
                cxyz[(lz * sy + ly) * sx + lx] +=
                    weight * ax[lx] * ay[ly] * az[lz];
}

double contract_product_from_I(const GridProduct& prod, int lpx, int lpy,
                               int lpz, const double* I) {
    double ax[kMaxL + 1], ay[kMaxL + 1], az[kMaxL + 1];
    const int lpx_p =
        expand_axis(prod.la[0], prod.lb[0], prod.pc[0], prod.ra[0], prod.rb[0],
                    ax);
    const int lpy_p =
        expand_axis(prod.la[1], prod.lb[1], prod.pc[1], prod.ra[1], prod.rb[1],
                    ay);
    const int lpz_p =
        expand_axis(prod.la[2], prod.lb[2], prod.pc[2], prod.ra[2], prod.rb[2],
                    az);
    const int sx = lpx + 1, sy = lpy + 1;
    double s = 0.0;
    for (int lz = 0; lz <= lpz_p; ++lz)
        for (int ly = 0; ly <= lpy_p; ++ly)
            for (int lx = 0; lx <= lpx_p; ++lx)
                s += I[(lz * sy + ly) * sx + lx] * ax[lx] * ay[ly] * az[lz];
    return s;
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
    AxisGeom ag;
    if (!prepare_pol(grid, prods[0], lpx, lpy, lpz, mag, Ltot, eps, ag))
        return;
    const int sx = lpx + 1, sy = lpy + 1, sz = lpz + 1;
    double* cxyz = cxyz_buf(ag);
    std::fill(cxyz, cxyz + sx * sy * sz, 0.0);
    for (int i = 0; i < n; ++i)
        add_product_to_cxyz(prods[i], weights[i], lpx, lpy, lpz, cxyz);
    (void)contract_cxyz<true>(ag, cxyz, rho, nullptr);
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
    AxisGeom ag;
    if (!prepare_pol(grid, prods[0], lpx, lpy, lpz, mag, Ltot, eps, ag))
        return;
    const int sx = lpx + 1, sy = lpy + 1, sz = lpz + 1;
    ensure_I(static_cast<std::size_t>(sx * sy * sz));
    double* I = g_I.data();
    (void)contract_cxyz<false>(ag, I, nullptr, V);
    for (int i = 0; i < n; ++i)
        out_acc[i] = contract_product_from_I(prods[i], lpx, lpy, lpz, I);
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
