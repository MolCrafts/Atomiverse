/// @file src/cpu/gpw/internal/product_collint.cpp
/// @brief CP2K ortho collint product collocate/integrate (see product_collint.h).
///
/// Near-literal port of CP2K `src/grid/cpu/grid_cpu_collint.h` ortho path
/// (BSD-3-Clause, Ole Schuett et al.):
///   - even cube + reflection pairs (k,1−k) / (j,1−j) → 4 grid points / i
///   - sphere-bounds iterator (no per-column sqrt in the hot nest)
///   - pol[dir][l][ig] via exp recurrence
///   - cxyz → cxy → cx → grid (collocate) / transpose (integrate)
///
/// Axis map onto ATV z-fast layout `g[(ix*ny+iy)*nz+iz]`:
///   CP2K i (fast) ↔ ATV z,  j ↔ y,  k ↔ x.
/// Portable xsimd on contiguous z-runs (project law — no AVX2 intrinsics).

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

constexpr int kMaxL = 12;
constexpr int kMaxCmax = 128; // half-cube; ~25 Bohr / 0.1 Å is still well under

// Grow-only thread_local scratch.
thread_local std::vector<double> g_buf;  // pol + cxyz + cxy + cx + gaus
thread_local std::vector<int> g_imap;    // map[3][2*cmax+1] + sections + sphere
thread_local std::vector<double> g_I;    // monom table (integrate)

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
inline int modulo(int a, int n) {
    // positive mod (CP2K)
    return ((a % n) + n) % n;
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

/// CP2K single_sphere_bounds: stream kgmin; for each kg≤0: jgmin; for each jg≤0: igmin.
/// (i,j,k) axes use spacing (di,dj,dk). Returns count of ints written (or needed).
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
    // Layout [lx][ly][lz] with lz fastest (CP2K lxp along i = our z).
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

/// CP2K ortho cxyz↔grid with reflection. Axes (i,j,k)=(z,y,x) on ATV packing.
template <bool Collocate>
void ortho_cxyz_to_grid(const UniformGrid& grid, const double pc[3],
                        double zetp, double radius, int lpx, int lpy, int lpz,
                        double* cxyz, double* grid_io) {
    // Spacings along (i,j,k) = (z,y,x).
    const double di = grid.L[2] / static_cast<double>(grid.n[2]);
    const double dj = grid.L[1] / static_cast<double>(grid.n[1]);
    const double dk = grid.L[0] / static_cast<double>(grid.n[0]);
    if (!(radius > 0.0) || 2.0 * radius < std::max({di, dj, dk}))
        return;

    // Cubecenter in (i,j,k) from absolute pc − origin.
    const double rp[3] = {pc[2] - grid.origin[2], pc[1] - grid.origin[1],
                          pc[0] - grid.origin[0]};
    int cubecenter[3];
    cubecenter[0] = static_cast<int>(std::floor(rp[0] / di)); // i ← z
    cubecenter[1] = static_cast<int>(std::floor(rp[1] / dj)); // j ← y
    cubecenter[2] = static_cast<int>(std::floor(rp[2] / dk)); // k ← x
    const double roffset[3] = {rp[0] - cubecenter[0] * di,
                               rp[1] - cubecenter[1] * dj,
                               rp[2] - cubecenter[2] * dk};
    const double dr[3] = {di, dj, dk};

    // Sphere bounds (CP2K stream). Discretize radius to integer grid steps.
    const double drmin = std::min({di, dj, dk});
    const int imr = std::max(1, static_cast<int>(std::ceil(radius / drmin)));
    const double disr_radius = drmin * static_cast<double>(imr);

    const int nbounds = single_sphere_bounds(disr_radius, di, dj, dk, nullptr);
    // Cube half-width from discretized radius.
    int lb_cube[3], ub_cube[3];
    lb_cube[0] = static_cast<int>(std::ceil(-1e-8 - disr_radius / di));
    lb_cube[1] = static_cast<int>(std::ceil(-1e-8 - disr_radius / dj));
    lb_cube[2] = static_cast<int>(std::ceil(-1e-8 - disr_radius / dk));
    for (int d = 0; d < 3; ++d)
        ub_cube[d] = 1 - lb_cube[d];
    const int cmax = imax(imax(ub_cube[0], ub_cube[1]), ub_cube[2]);
    if (cmax <= 0 || cmax > kMaxCmax)
        return;

    const int lp_x = lpx, lp_y = lpy, lp_z = lpz; // monoms along x,y,z
    // pol layout: pol_off[d] + l * (2*cmax+1) + (ig+cmax)
    // d: 0=i=z (lp_z), 1=j=y (lp_y), 2=k=x (lp_x)
    const int pol_len = 2 * cmax + 1;
    const int pol_lp[3] = {lp_z, lp_y, lp_x};
    int pol_off[3];
    int po = 0;
    for (int d = 0; d < 3; ++d) {
        pol_off[d] = po;
        po += (pol_lp[d] + 1) * pol_len;
    }
    const int sy = lpy + 1, sz = lpz + 1;
    const int cxyz_n = (lpx + 1) * sy * sz;
    const int cxy_n = sy * sz * 2;
    const int cx_n = sz * 4;
    ensure_buf(static_cast<std::size_t>(po + cxy_n + cx_n + pol_len));
    ensure_imap(static_cast<std::size_t>(3 * pol_len * 2 + nbounds + 8));

    double* pol_base = g_buf.data();
    double* cxy = g_buf.data() + po;
    double* cx = cxy + cxy_n;
    // Build pol — CP2K exp recurrence (even cube, ig and 1−ig pairs).
    for (int idir = 0; idir < 3; ++idir) {
        const double dri = dr[idir];
        const double ro = roffset[idir];
        const int lp = pol_lp[idir];
        double* pol = pol_base + pol_off[idir];
        const double t_exp_1 = std::exp(-zetp * dri * dri);
        const double t_exp_2 = t_exp_1 * t_exp_1;
        // Negative / zero side (ig = 0, −1, −2, … → lb)
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
        // Positive side (grid indices 1,2,3,…)
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

    // map[d][ig+cmax] → wrapped grid index along ATV axis.
    // d=0 → z (n[2]), d=1 → y (n[1]), d=2 → x (n[0])
    const int n_atv[3] = {grid.n[2], grid.n[1], grid.n[0]};
    int* map0 = g_imap.data();
    int* map1 = map0 + pol_len;
    int* map2 = map1 + pol_len;
    int* sec0 = map2 + pol_len;
    int* sec1 = sec0 + pol_len;
    int* sec2 = sec1 + pol_len;
    int* sphere = sec2 + pol_len;
    int* maps[3] = {map0, map1, map2};
    int* secs[3] = {sec0, sec1, sec2};
    for (int d = 0; d < 3; ++d) {
        for (int k = -cmax; k <= cmax; ++k)
            maps[d][k + cmax] =
                modulo(cubecenter[d] + k, n_atv[d]);
        for (int kg = 2 * cmax; kg >= 0; --kg) {
            if (kg == 2 * cmax
                || maps[d][kg] != maps[d][kg + 1] - 1)
                secs[d][kg] = 0;
            else
                secs[d][kg] = secs[d][kg + 1] + 1;
        }
    }
    single_sphere_bounds(disr_radius, di, dj, dk, sphere);

    const double* pol_i = pol_base + pol_off[0]; // z
    const double* pol_j = pol_base + pol_off[1]; // y
    const double* pol_k = pol_base + pol_off[2]; // x

    const long nx = grid.n[0], ny = grid.n[1], nz = grid.n[2];
    int* sb = sphere;
    const int kstart = *sb++;

    for (int k1 = kstart; k1 <= 0; ++k1) {
        const int k2 = 1 - k1;
        const int kg1 = map2[k1 + cmax]; // ix
        const int kg2 = map2[k2 + cmax];
        std::fill(cxy, cxy + cxy_n, 0.0);

        if constexpr (Collocate) {
            // cxyz → cxy : contract lx with pol_k
            for (int lx = 0; lx <= lp_x; ++lx) {
                const double p1 = pol_k[lx * pol_len + (k1 + cmax)];
                const double p2 = pol_k[lx * pol_len + (k2 + cmax)];
                for (int ly = 0; ly <= lp_y; ++ly)
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const double c =
                            cxyz[(lx * sy + ly) * sz + lz];
                        const int cxy_i = (ly * sz + lz) * 2;
                        cxy[cxy_i + 0] += c * p1;
                        cxy[cxy_i + 1] += c * p2;
                    }
            }
        }

        const int jstart = *sb++;
        for (int j1 = jstart; j1 <= 0; ++j1) {
            const int j2 = 1 - j1;
            const int jg1 = map1[j1 + cmax]; // iy
            const int jg2 = map1[j2 + cmax];
            std::fill(cx, cx + cx_n, 0.0);

            if constexpr (Collocate) {
                // cxy → cx : contract ly with pol_j
                for (int ly = 0; ly <= lp_y; ++ly) {
                    const double p1 = pol_j[ly * pol_len + (j1 + cmax)];
                    const double p2 = pol_j[ly * pol_len + (j2 + cmax)];
                    for (int lz = 0; lz <= lp_z; ++lz) {
                        const int cxy_i = (ly * sz + lz) * 2;
                        cx[lz * 4 + 0] += cxy[cxy_i + 0] * p1;
                        cx[lz * 4 + 1] += cxy[cxy_i + 1] * p1;
                        cx[lz * 4 + 2] += cxy[cxy_i + 0] * p2;
                        cx[lz * 4 + 3] += cxy[cxy_i + 1] * p2;
                    }
                }
            }

            // Sphere i-bounds for this (k,j) column pair.
            const int lb = *sb++;
            const int ub = 1 - lb;

            // Base linear indices for the 4 reflected (ix,iy) pairs; +iz later.
            const long base0 = (static_cast<long>(kg1) * ny + jg1) * nz;
            const long base1 = (static_cast<long>(kg2) * ny + jg1) * nz;
            const long base2 = (static_cast<long>(kg1) * ny + jg2) * nz;
            const long base3 = (static_cast<long>(kg2) * ny + jg2) * nz;

            if constexpr (Collocate) {
                using batch_t = xsimd::batch<double>;
                constexpr int Wd = static_cast<int>(batch_t::size);
                for (int istart = lb; istart <= ub;) {
                    const int istop =
                        imin(ub, istart + sec0[istart + cmax]);
                    const int cube2grid = map0[istart + cmax] - istart;
                    // Contiguous z-run: vectorize over successive i when the
                    // map is homogeneous (section) and ig increases by 1.
                    int i = istart;
                    // Vector path only when grid indices are contiguous.
                    if (cube2grid + istart >= 0) {
                        for (; i + Wd - 1 <= istop; i += Wd) {
                            const int ig = i + cube2grid;
                            batch_t r0(0.0), r1(0.0), r2(0.0), r3(0.0);
                            for (int lz = 0; lz <= lp_z; ++lz) {
                                // pol along i is contiguous in i for fixed lz.
                                const batch_t p = batch_t::load_unaligned(
                                    pol_i + lz * pol_len + (i + cmax));
                                r0 += p * batch_t(cx[lz * 4 + 0]);
                                r1 += p * batch_t(cx[lz * 4 + 1]);
                                r2 += p * batch_t(cx[lz * 4 + 2]);
                                r3 += p * batch_t(cx[lz * 4 + 3]);
                            }
                            batch_t g0 =
                                batch_t::load_unaligned(grid_io + base0 + ig);
                            batch_t g1 =
                                batch_t::load_unaligned(grid_io + base1 + ig);
                            batch_t g2 =
                                batch_t::load_unaligned(grid_io + base2 + ig);
                            batch_t g3 =
                                batch_t::load_unaligned(grid_io + base3 + ig);
                            (g0 + r0).store_unaligned(grid_io + base0 + ig);
                            (g1 + r1).store_unaligned(grid_io + base1 + ig);
                            (g2 + r2).store_unaligned(grid_io + base2 + ig);
                            (g3 + r3).store_unaligned(grid_io + base3 + ig);
                        }
                    }
                    for (; i <= istop; ++i) {
                        const int ig = i + cube2grid;
                        double reg0 = 0.0, reg1 = 0.0, reg2 = 0.0,
                               reg3 = 0.0;
                        for (int lz = 0; lz <= lp_z; ++lz) {
                            const double p =
                                pol_i[lz * pol_len + (i + cmax)];
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
            } else {
                // Integrate: gather V into cx, then reverse-contract.
                using batch_t = xsimd::batch<double>;
                constexpr int Wd = static_cast<int>(batch_t::size);
                for (int istart = lb; istart <= ub;) {
                    const int istop =
                        imin(ub, istart + sec0[istart + cmax]);
                    const int cube2grid = map0[istart + cmax] - istart;
                    int i = istart;
                    if (cube2grid + istart >= 0) {
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
                                // Horizontal add into scalar cx slots.
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
                            const double p =
                                pol_i[lz * pol_len + (i + cmax)];
                            cx[lz * 4 + 0] += v0 * p;
                            cx[lz * 4 + 1] += v1 * p;
                            cx[lz * 4 + 2] += v2 * p;
                            cx[lz * 4 + 3] += v3 * p;
                        }
                    }
                    istart = istop + 1;
                }
                // cx → cxy
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
            // cxy → cxyz
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
    ensure_I(static_cast<std::size_t>(cxyz_n)); // reuse g_I as cxyz scratch
    double* cxyz = g_I.data();
    std::fill(cxyz, cxyz + cxyz_n, 0.0);
    for (int i = 0; i < n; ++i)
        add_product_to_cxyz(prods[i], weights[i], lpx, lpy, lpz, cxyz);
    ortho_cxyz_to_grid<true>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                             lpz, cxyz, rho);
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
    // const_cast: integrate only reads V; signature uses double* for shared
    // collocate/integrate template (grid_io is V when integrating).
    ortho_cxyz_to_grid<false>(grid, prods[0].pc, prods[0].p, radius, lpx, lpy,
                              lpz, I, const_cast<double*>(V));
    for (int i = 0; i < n; ++i)
        out_acc[i] = contract_product_from_I(prods[i], lpx, lpy, lpz, I);
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
