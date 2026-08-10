/// @file src/cpu/gpw/internal/one_electron.cpp
/// @brief McMurchie-Davidson overlap and kinetic integrals over GridPrimitives.
///
/// Hot path uses a memoized Hermite table per Cartesian axis (no exponential
/// recursion tree) and a fused prim_ST that shares 1D overlaps between S and T.
/// Geometry-key reuse keeps tables live across consecutive prim pairs that share
/// (α, center) — common for solid-harmonic cart terms and shell-mate AOs.
///
/// Pure s–s contracted work (dominant on H / s shells, and on slab image
/// sweeps) is vectorized with portable xsimd over the monom-pair / image axis
/// (raw spans + scalar tail). Mixed-angular (p/d) multi-image ST uses the same
/// image-axis xsimd over McMurchie–Davidson Hermite tables (L≤d, shell-centered).
/// Image screening packs live T first so the ST kernel sees a short list.

#include "atomiverse/cpu/gpw/internal/one_electron.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

// GPW MOLOPT is ≤ d (l=2); kinetic needs j+2 ⇒ 4. Headroom for f/g.
constexpr int kMaxAng = 8;

/// Memoized Hermite E_t^{i,j} for one axis (Helgaker 9.5.6–7).
/// Generation stamp avoids memset of the whole table on every prim pair
/// (thread_local reuse in prim_ST / prim_overlap / prim_kinetic).
struct HermiteTab {
    double p = 0, mu = 0, xab = 0, xpa = 0, xpb = 0, sqrt_pi_p = 0;
    double memo[kMaxAng + 1][kMaxAng + 1][2 * kMaxAng + 1];
    unsigned stamp[kMaxAng + 1][kMaxAng + 1][2 * kMaxAng + 1] = {};
    unsigned gen = 0;
    // Bottom-up fill extent (inclusive). -1 ⇒ use recursive E only.
    int filled_i = -1, filled_j = -1;

    void init(double a, double b, double Ax, double Bx) {
        p = a + b;
        mu = a * b / p;
        xab = Ax - Bx;
        const double Px = (a * Ax + b * Bx) / p;
        xpa = Px - Ax;
        xpb = Px - Bx;
        sqrt_pi_p = std::sqrt(M_PI / p);
        ++gen;
        if (gen == 0) {
            std::memset(stamp, 0, sizeof(stamp));
            gen = 1;
        }
        filled_i = filled_j = -1;
    }

    /// Bottom-up McMurchie–Davidson fill of E_t^{i,j} for i≤imax, j≤jmax
    /// (and all t). Kinetic needs j+2 on the ket side — pass jmax = l_b+2.
    /// ~2–4× fewer calls than recursive memo for multi-monom shell pairs.
    void fill_upto(int imax, int jmax) {
        imax = std::min(imax, kMaxAng);
        jmax = std::min(jmax, kMaxAng);
        if (imax < 0 || jmax < 0)
            return;
        if (filled_i >= imax && filled_j >= jmax)
            return;
        // Grow to the max of previous fill and request (first call: filled=-1).
        if (filled_i >= 0)
            imax = std::max(imax, filled_i);
        if (filled_j >= 0)
            jmax = std::max(jmax, filled_j);
        const double oo2p = 0.5 / p;
        memo[0][0][0] = std::exp(-mu * xab * xab);
        stamp[0][0][0] = gen;
        // j=0 column: raise i  (E_t^{i,0})
        for (int i = 1; i <= imax; ++i) {
            for (int t = 0; t <= i; ++t) {
                double v = 0.0;
                if (t - 1 >= 0 && t - 1 <= i - 1)
                    v += oo2p * memo[i - 1][0][t - 1];
                if (t <= i - 1)
                    v += xpa * memo[i - 1][0][t];
                if (t + 1 <= i - 1)
                    v += static_cast<double>(t + 1) * memo[i - 1][0][t + 1];
                memo[i][0][t] = v;
                stamp[i][0][t] = gen;
            }
        }
        // raise j for all i
        for (int j = 1; j <= jmax; ++j) {
            for (int i = 0; i <= imax; ++i) {
                for (int t = 0; t <= i + j; ++t) {
                    double v = 0.0;
                    if (t - 1 >= 0 && t - 1 <= i + j - 1)
                        v += oo2p * memo[i][j - 1][t - 1];
                    if (t <= i + j - 1)
                        v += xpb * memo[i][j - 1][t];
                    if (t + 1 <= i + j - 1)
                        v += static_cast<double>(t + 1) * memo[i][j - 1][t + 1];
                    memo[i][j][t] = v;
                    stamp[i][j][t] = gen;
                }
            }
        }
        filled_i = imax;
        filled_j = jmax;
    }

    double E(int i, int j, int t) {
        if (t < 0 || t > i + j || i < 0 || j < 0)
            return 0.0;
        if (i > kMaxAng || j > kMaxAng)
            return 0.0; // refuse — caller should not hit this for GPW
        if (stamp[i][j][t] == gen)
            return memo[i][j][t];
        // If bottom-up filled this range, missing stamp ⇒ true zero (t gap).
        if (filled_i >= i && filled_j >= j)
            return 0.0;
        double v;
        if (i == 0 && j == 0) {
            v = std::exp(-mu * xab * xab); // t==0
        } else {
            const double oo2p = 0.5 / p;
            if (j == 0) {
                v = oo2p * E(i - 1, j, t - 1) + xpa * E(i - 1, j, t)
                    + static_cast<double>(t + 1) * E(i - 1, j, t + 1);
            } else {
                v = oo2p * E(i, j - 1, t - 1) + xpb * E(i, j - 1, t)
                    + static_cast<double>(t + 1) * E(i, j - 1, t + 1);
            }
        }
        stamp[i][j][t] = gen;
        memo[i][j][t] = v;
        return v;
    }

    double S1d(int i, int j) {
        if (i < 0 || j < 0)
            return 0.0;
        return E(i, j, 0) * sqrt_pi_p;
    }

    /// Fast S1d when fill_upto covers (i,j) — direct memo read.
    double S1d_filled(int i, int j) const {
        if (i < 0 || j < 0)
            return 0.0;
        return memo[i][j][0] * sqrt_pi_p;
    }
};

/// Last (α, center) key for which hx/hy/hz are valid (thread_local).
struct HermiteGeomKey {
    double aa = 0, ba = 0;
    double ac[3] = {0, 0, 0};
    double bc[3] = {0, 0, 0};
    bool live = false;

    bool matches(double a_alpha, double b_alpha, const double a_c[3],
                 const double b_c[3]) const {
        return live && aa == a_alpha && ba == b_alpha && ac[0] == a_c[0]
               && ac[1] == a_c[1] && ac[2] == a_c[2] && bc[0] == b_c[0]
               && bc[1] == b_c[1] && bc[2] == b_c[2];
    }

    void set(double a_alpha, double b_alpha, const double a_c[3],
             const double b_c[3]) {
        aa = a_alpha;
        ba = b_alpha;
        ac[0] = a_c[0];
        ac[1] = a_c[1];
        ac[2] = a_c[2];
        bc[0] = b_c[0];
        bc[1] = b_c[1];
        bc[2] = b_c[2];
        live = true;
    }
};

/// Ensure per-axis Hermite tables match (α_a, α_b, A, B); skip re-init on hit.
inline void hermite_ensure(HermiteTab& hx, HermiteTab& hy, HermiteTab& hz,
                           HermiteGeomKey& key, double a_alpha, double b_alpha,
                           const double a_c[3], const double b_c[3]) {
    if (key.matches(a_alpha, b_alpha, a_c, b_c))
        return;
    hx.init(a_alpha, b_alpha, a_c[0], b_c[0]);
    hy.init(a_alpha, b_alpha, a_c[1], b_c[1]);
    hz.init(a_alpha, b_alpha, a_c[2], b_c[2]);
    key.set(a_alpha, b_alpha, a_c, b_c);
}

/// 1D kinetic with known b (ket exponent).
inline double kinetic_1d_tab(HermiteTab& h, int i, int j, double b) {
    double t = -2.0 * b * b * h.S1d(i, j + 2)
               + b * (2 * j + 1) * h.S1d(i, j);
    if (j >= 2)
        t -= 0.5 * j * (j - 1) * h.S1d(i, j - 2);
    return t;
}

/// Core ST from live Hermite tables + angular/coeffs.
inline void st_from_tabs(HermiteTab& hx, HermiteTab& hy, HermiteTab& hz,
                         int alx, int aly, int alz, int blx, int bly, int blz,
                         double a_alpha, double b_alpha, double ca, double cb,
                         double& s_out, double& t_out) {
    const double sx = hx.S1d(alx, blx);
    const double sy = hy.S1d(aly, bly);
    const double sz = hz.S1d(alz, blz);
    const double tx = kinetic_1d_tab(hx, alx, blx, b_alpha);
    const double ty = kinetic_1d_tab(hy, aly, bly, b_alpha);
    const double tz = kinetic_1d_tab(hz, alz, blz, b_alpha);
    (void)a_alpha;
    const double c = ca * cb;
    s_out = c * sx * sy * sz;
    t_out = c * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
}

/// s–s closed form (no Hermite).
inline void st_ss(double a_alpha, double b_alpha, const double a_c[3],
                  const double b_c[3], double ca, double cb, double& s_out,
                  double& t_out) {
    const double p = a_alpha + b_alpha;
    const double mu = a_alpha * b_alpha / p;
    const double dx = a_c[0] - b_c[0];
    const double dy = a_c[1] - b_c[1];
    const double dz = a_c[2] - b_c[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    // (π/p)^{3/2} = (π/p) √(π/p) — avoid std::pow.
    const double pi_over_p = M_PI / p;
    const double S =
        ca * cb * pi_over_p * std::sqrt(pi_over_p) * std::exp(-mu * r2);
    s_out = S;
    t_out = mu * (3.0 - 2.0 * mu * r2) * S;
}

/// True when every monom is pure s (l=0). Common for H / valence-s shells.
inline bool basis_is_pure_s(const GridBasisFunction& f) {
    for (const auto& p : f)
        if (p.l[0] != 0 || p.l[1] != 0 || p.l[2] != 0)
            return false;
    return !f.empty();
}

/// Flatten pure-s monom pairs into (α_a, α_b, c_a·c_b, μ, pref) tables.
/// Shared by fixed-geometry and multi-image SIMD paths.
struct SsPairPack {
    static constexpr int kMaxPairs = 64; // MOLOPT ≤ ~8×8
    double aa[kMaxPairs];
    double ab[kMaxPairs];
    double mu[kMaxPairs];
    double pref[kMaxPairs]; // c_a c_b (π/p)^{3/2}
    int np = 0;

    void build(const GridBasisFunction& a, const GridBasisFunction& b) {
        np = 0;
        for (const auto& pa : a) {
            for (const auto& pb : b) {
                if (np >= kMaxPairs)
                    return; // caller falls back if truncated
                const double p = pa.alpha + pb.alpha;
                const double m = pa.alpha * pb.alpha / p;
                const double pi_over_p = M_PI / p;
                aa[np] = pa.alpha;
                ab[np] = pb.alpha;
                mu[np] = m;
                pref[np] = pa.coeff * pb.coeff * pi_over_p * std::sqrt(pi_over_p);
                ++np;
            }
        }
    }
};

/// Portable xsimd s–s contracted ST for fixed centers (shared r²).
/// Vectorizes the **flattened monom-pair** axis (na·nb), not the ket axis alone
/// — MOLOPT 4×4 = 16 pairs fills two AVX-512 lanes of 8. Falls back to scalar
/// `st_ss` when np < native width. Not bit-identical (xsimd::exp + reduce).
inline void contracted_ST_ss_simd(const GridBasisFunction& a,
                                  const GridBasisFunction& b,
                                  const double bc[3], double& s_out,
                                  double& t_out) {
    using batch_t = xsimd::batch<double>;
    constexpr int W = static_cast<int>(batch_t::size);

    SsPairPack pack;
    pack.build(a, b);
    const int np = pack.np;
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    // Truncation or tiny work → pure scalar (no SIMD tax).
    if (np != na * nb || np < W) {
        double s = 0.0, t = 0.0;
        for (const auto& pa : a) {
            for (const auto& pb : b) {
                double s_p = 0.0, t_p = 0.0;
                st_ss(pa.alpha, pb.alpha, pa.center, bc, pa.coeff, pb.coeff, s_p,
                      t_p);
                s += s_p;
                t += t_p;
            }
        }
        s_out = s;
        t_out = t;
        return;
    }

    const double* ac = a[0].center;
    const double dx = ac[0] - bc[0];
    const double dy = ac[1] - bc[1];
    const double dz = ac[2] - bc[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const batch_t r2_v(r2);
    const batch_t three(3.0);
    const batch_t two(2.0);

    batch_t acc_s(0.0);
    batch_t acc_t(0.0);
    int ip = 0;
    for (; ip + W <= np; ip += W) {
        const batch_t mu_v = batch_t::load_unaligned(pack.mu + ip);
        const batch_t pref_v = batch_t::load_unaligned(pack.pref + ip);
        const batch_t S = pref_v * xsimd::exp(-mu_v * r2_v);
        const batch_t T = mu_v * (three - two * mu_v * r2_v) * S;
        acc_s += S;
        acc_t += T;
    }
    double s_lane[16];
    double t_lane[16];
    acc_s.store_unaligned(s_lane);
    acc_t.store_unaligned(t_lane);
    double s = 0.0, t = 0.0;
    for (int lane = 0; lane < W; ++lane) {
        s += s_lane[lane];
        t += t_lane[lane];
    }
    for (; ip < np; ++ip) {
        const double S = pack.pref[ip] * std::exp(-pack.mu[ip] * r2);
        s += S;
        t += pack.mu[ip] * (3.0 - 2.0 * pack.mu[ip] * r2) * S;
    }
    s_out = s;
    t_out = t;
}

inline bool image_in_reach(const double ac[3], const double bc[3],
                           const double T[3], double reach2) {
    const double dx = ac[0] - bc[0] - T[0];
    const double dy = ac[1] - bc[1] - T[1];
    const double dz = ac[2] - bc[2] - T[2];
    return dx * dx + dy * dy + dz * dz <= reach2;
}

/// Pack live lattice images + precomputed r² into a caller buffer.
/// @return number of live images written (≤ nmax).
inline int pack_live_images_r2(const double ac[3], const double bc[3],
                               const double L[3], const int m[3],
                               double reach2, double* r2, int nmax) {
    int n = 0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double dx = ac[0] - bc[0] - tx * L[0];
                const double dy = ac[1] - bc[1] - ty * L[1];
                const double dz = ac[2] - bc[2] - tz * L[2];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > reach2)
                    continue;
                if (n >= nmax)
                    return n; // truncated — caller falls back
                r2[n++] = d2;
            }
    return n;
}

/// Pack live image displacements d = A − B − T (needed for Hermite xab per axis).
/// @return number of live images written (≤ nmax); on truncation returns nmax.
inline int pack_live_images_dxyz(const double ac[3], const double bc[3],
                                 const double L[3], const int m[3],
                                 double reach2, double* dx, double* dy,
                                 double* dz, int nmax) {
    int n = 0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double ddx = ac[0] - bc[0] - tx * L[0];
                const double ddy = ac[1] - bc[1] - ty * L[1];
                const double ddz = ac[2] - bc[2] - tz * L[2];
                const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (d2 > reach2)
                    continue;
                if (n >= nmax)
                    return n;
                dx[n] = ddx;
                dy[n] = ddy;
                dz[n] = ddz;
                ++n;
            }
    return n;
}

inline int basis_max_cart_l(const GridBasisFunction& f) {
    int m = 0;
    for (const auto& p : f)
        m = std::max(m, std::max(p.l[0], std::max(p.l[1], p.l[2])));
    return m;
}

inline bool basis_same_center(const GridBasisFunction& f) {
    if (f.empty())
        return true;
    const double* c = f[0].center;
    for (const auto& p : f)
        if (p.center[0] != c[0] || p.center[1] != c[1] || p.center[2] != c[2])
            return false;
    return true;
}

// GPW MOLOPT ≤ d (l=2); kinetic needs j+2 ⇒ fill ≤ 4. Stack-sized Hermite
// batches for portable image-axis xsimd; higher L → scalar fallback.
constexpr int kHermSimdAng = 4;

/// One-axis McMurchie–Davidson E_t^{i,j} as an xsimd batch over lattice images.
struct HermiteAxisBatch {
    using batch_t = xsimd::batch<double>;
    batch_t memo[kHermSimdAng + 1][kHermSimdAng + 1][2 * kHermSimdAng + 1];
    batch_t sqrt_pi_p;

    void fill(const batch_t& xab, double a, double b, int imax, int jmax) {
        imax = std::min(imax, kHermSimdAng);
        jmax = std::min(jmax, kHermSimdAng);
        const double p = a + b;
        const double mu = a * b / p;
        const double oo2p = 0.5 / p;
        const batch_t xpa = batch_t(-b / p) * xab;
        const batch_t xpb = batch_t(a / p) * xab;
        sqrt_pi_p = batch_t(std::sqrt(M_PI / p));
        memo[0][0][0] = xsimd::exp(batch_t(-mu) * xab * xab);
        for (int i = 1; i <= imax; ++i) {
            for (int t = 0; t <= i; ++t) {
                batch_t v(0.0);
                if (t - 1 >= 0 && t - 1 <= i - 1)
                    v += batch_t(oo2p) * memo[i - 1][0][t - 1];
                if (t <= i - 1)
                    v += xpa * memo[i - 1][0][t];
                if (t + 1 <= i - 1)
                    v += batch_t(static_cast<double>(t + 1))
                         * memo[i - 1][0][t + 1];
                memo[i][0][t] = v;
            }
        }
        for (int j = 1; j <= jmax; ++j) {
            for (int i = 0; i <= imax; ++i) {
                for (int t = 0; t <= i + j; ++t) {
                    batch_t v(0.0);
                    if (t - 1 >= 0 && t - 1 <= i + j - 1)
                        v += batch_t(oo2p) * memo[i][j - 1][t - 1];
                    if (t <= i + j - 1)
                        v += xpb * memo[i][j - 1][t];
                    if (t + 1 <= i + j - 1)
                        v += batch_t(static_cast<double>(t + 1))
                             * memo[i][j - 1][t + 1];
                    memo[i][j][t] = v;
                }
            }
        }
    }

    batch_t S1d(int i, int j) const {
        if (i < 0 || j < 0)
            return batch_t(0.0);
        return memo[i][j][0] * sqrt_pi_p;
    }

    batch_t kinetic_1d(int i, int j, double b_exp) const {
        batch_t tt = batch_t(-2.0 * b_exp * b_exp) * S1d(i, j + 2)
                     + batch_t(b_exp * (2 * j + 1)) * S1d(i, j);
        if (j >= 2)
            tt -= batch_t(0.5 * j * (j - 1)) * S1d(i, j - 2);
        return tt;
    }
};

/// Mixed-angular multi-image ST: image-axis portable xsimd Hermite.
/// Monoms that share (α_a, α_b) share one Hermite fill per image batch.
/// Not bit-identical (xsimd::exp + reduce).
inline void contracted_ST_angular_images_simd(const GridBasisFunction& a,
                                              const GridBasisFunction& b,
                                              const double* dx,
                                              const double* dy,
                                              const double* dz, int nimg,
                                              double& s_out, double& t_out) {
    using batch_t = xsimd::batch<double>;
    constexpr int W = static_cast<int>(batch_t::size);

    double s = 0.0, t = 0.0;
    HermiteAxisBatch hx, hy, hz;
    const batch_t three(3.0);
    const batch_t two(2.0);
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());

    for (int ia0 = 0; ia0 < na; ++ia0) {
        const auto& pa0 = a[static_cast<std::size_t>(ia0)];
        bool seen_a = false;
        for (int ja = 0; ja < ia0; ++ja) {
            if (a[static_cast<std::size_t>(ja)].alpha == pa0.alpha) {
                seen_a = true;
                break;
            }
        }
        if (seen_a)
            continue;
        for (int ib0 = 0; ib0 < nb; ++ib0) {
            const auto& pb0 = b[static_cast<std::size_t>(ib0)];
            bool seen_b = false;
            for (int jb = 0; jb < ib0; ++jb) {
                if (b[static_cast<std::size_t>(jb)].alpha == pb0.alpha) {
                    seen_b = true;
                    break;
                }
            }
            if (seen_b)
                continue;

            int ia_list[64], ib_list[64];
            int na_g = 0, nb_g = 0;
            int imax = 0, jmax_l = 0;
            bool all_ss = true;
            for (int ia = 0; ia < na; ++ia) {
                const auto& pa = a[static_cast<std::size_t>(ia)];
                if (pa.alpha != pa0.alpha)
                    continue;
                if (na_g < 64)
                    ia_list[na_g++] = ia;
                imax = std::max(
                    imax, std::max(pa.l[0], std::max(pa.l[1], pa.l[2])));
                if (pa.l[0] || pa.l[1] || pa.l[2])
                    all_ss = false;
            }
            for (int ib = 0; ib < nb; ++ib) {
                const auto& pb = b[static_cast<std::size_t>(ib)];
                if (pb.alpha != pb0.alpha)
                    continue;
                if (nb_g < 64)
                    ib_list[nb_g++] = ib;
                jmax_l = std::max(
                    jmax_l, std::max(pb.l[0], std::max(pb.l[1], pb.l[2])));
                if (pb.l[0] || pb.l[1] || pb.l[2])
                    all_ss = false;
            }
            if (na_g == 0 || nb_g == 0)
                continue;

            if (all_ss) {
                for (int ka = 0; ka < na_g; ++ka) {
                    const auto& pa = a[static_cast<std::size_t>(ia_list[ka])];
                    for (int kb = 0; kb < nb_g; ++kb) {
                        const auto& pb =
                            b[static_cast<std::size_t>(ib_list[kb])];
                        const double p = pa.alpha + pb.alpha;
                        const double mu = pa.alpha * pb.alpha / p;
                        const double pi_over_p = M_PI / p;
                        const double pref = pa.coeff * pb.coeff * pi_over_p
                                            * std::sqrt(pi_over_p);
                        batch_t acc_s(0.0), acc_t(0.0);
                        int im = 0;
                        for (; im + W <= nimg; im += W) {
                            const batch_t dx_v =
                                batch_t::load_unaligned(dx + im);
                            const batch_t dy_v =
                                batch_t::load_unaligned(dy + im);
                            const batch_t dz_v =
                                batch_t::load_unaligned(dz + im);
                            const batch_t r2_v =
                                dx_v * dx_v + dy_v * dy_v + dz_v * dz_v;
                            const batch_t S =
                                batch_t(pref)
                                * xsimd::exp(batch_t(-mu) * r2_v);
                            const batch_t T = batch_t(mu)
                                              * (three - two * batch_t(mu)
                                                             * r2_v)
                                              * S;
                            acc_s += S;
                            acc_t += T;
                        }
                        s += xsimd::reduce_add(acc_s);
                        t += xsimd::reduce_add(acc_t);
                        for (; im < nimg; ++im) {
                            const double r2 = dx[im] * dx[im]
                                              + dy[im] * dy[im]
                                              + dz[im] * dz[im];
                            const double S = pref * std::exp(-mu * r2);
                            s += S;
                            t += mu * (3.0 - 2.0 * mu * r2) * S;
                        }
                    }
                }
                continue;
            }

            const int jmax = jmax_l + 2;
            if (imax > kHermSimdAng || jmax > kHermSimdAng) {
                for (int ka = 0; ka < na_g; ++ka) {
                    const auto& pa =
                        a[static_cast<std::size_t>(ia_list[ka])];
                    for (int kb = 0; kb < nb_g; ++kb) {
                        const auto& pb =
                            b[static_cast<std::size_t>(ib_list[kb])];
                        for (int im = 0; im < nimg; ++im) {
                            const double bcs[3] = {pa.center[0] - dx[im],
                                                   pa.center[1] - dy[im],
                                                   pa.center[2] - dz[im]};
                            HermiteTab hx_s, hy_s, hz_s;
                            hx_s.init(pa.alpha, pb.alpha, pa.center[0],
                                      bcs[0]);
                            hy_s.init(pa.alpha, pb.alpha, pa.center[1],
                                      bcs[1]);
                            hz_s.init(pa.alpha, pb.alpha, pa.center[2],
                                      bcs[2]);
                            const int ifill = std::min(kMaxAng, imax);
                            const int jfill = std::min(kMaxAng, jmax);
                            hx_s.fill_upto(ifill, jfill);
                            hy_s.fill_upto(ifill, jfill);
                            hz_s.fill_upto(ifill, jfill);
                            auto kin1d = [](const HermiteTab& h, int i, int j,
                                            double b_exp) {
                                double tt =
                                    -2.0 * b_exp * b_exp
                                        * h.S1d_filled(i, j + 2)
                                    + b_exp * (2 * j + 1)
                                          * h.S1d_filled(i, j);
                                if (j >= 2)
                                    tt -= 0.5 * j * (j - 1)
                                          * h.S1d_filled(i, j - 2);
                                return tt;
                            };
                            const double sx =
                                hx_s.S1d_filled(pa.l[0], pb.l[0]);
                            const double sy =
                                hy_s.S1d_filled(pa.l[1], pb.l[1]);
                            const double sz =
                                hz_s.S1d_filled(pa.l[2], pb.l[2]);
                            const double tx =
                                kin1d(hx_s, pa.l[0], pb.l[0], pb.alpha);
                            const double ty =
                                kin1d(hy_s, pa.l[1], pb.l[1], pb.alpha);
                            const double tz =
                                kin1d(hz_s, pa.l[2], pb.l[2], pb.alpha);
                            const double c = pa.coeff * pb.coeff;
                            s += c * sx * sy * sz;
                            t += c
                                 * (tx * sy * sz + sx * ty * sz
                                    + sx * sy * tz);
                        }
                    }
                }
                continue;
            }

            int im = 0;
            for (; im + W <= nimg; im += W) {
                const batch_t xab_x = batch_t::load_unaligned(dx + im);
                const batch_t xab_y = batch_t::load_unaligned(dy + im);
                const batch_t xab_z = batch_t::load_unaligned(dz + im);
                hx.fill(xab_x, pa0.alpha, pb0.alpha, imax, jmax);
                hy.fill(xab_y, pa0.alpha, pb0.alpha, imax, jmax);
                hz.fill(xab_z, pa0.alpha, pb0.alpha, imax, jmax);
                for (int ka = 0; ka < na_g; ++ka) {
                    const auto& pa =
                        a[static_cast<std::size_t>(ia_list[ka])];
                    for (int kb = 0; kb < nb_g; ++kb) {
                        const auto& pb =
                            b[static_cast<std::size_t>(ib_list[kb])];
                        const batch_t sx = hx.S1d(pa.l[0], pb.l[0]);
                        const batch_t sy = hy.S1d(pa.l[1], pb.l[1]);
                        const batch_t sz = hz.S1d(pa.l[2], pb.l[2]);
                        const batch_t tx =
                            hx.kinetic_1d(pa.l[0], pb.l[0], pb.alpha);
                        const batch_t ty =
                            hy.kinetic_1d(pa.l[1], pb.l[1], pb.alpha);
                        const batch_t tz =
                            hz.kinetic_1d(pa.l[2], pb.l[2], pb.alpha);
                        const batch_t c(pa.coeff * pb.coeff);
                        s += xsimd::reduce_add(c * sx * sy * sz);
                        t += xsimd::reduce_add(
                            c * (tx * sy * sz + sx * ty * sz + sx * sy * tz));
                    }
                }
            }
            for (; im < nimg; ++im) {
                const double bcs[3] = {pa0.center[0] - dx[im],
                                       pa0.center[1] - dy[im],
                                       pa0.center[2] - dz[im]};
                HermiteTab hx_s, hy_s, hz_s;
                hx_s.init(pa0.alpha, pb0.alpha, pa0.center[0], bcs[0]);
                hy_s.init(pa0.alpha, pb0.alpha, pa0.center[1], bcs[1]);
                hz_s.init(pa0.alpha, pb0.alpha, pa0.center[2], bcs[2]);
                hx_s.fill_upto(imax, jmax);
                hy_s.fill_upto(imax, jmax);
                hz_s.fill_upto(imax, jmax);
                auto kin1d = [](const HermiteTab& h, int i, int j,
                                double b_exp) {
                    double tt = -2.0 * b_exp * b_exp * h.S1d_filled(i, j + 2)
                                + b_exp * (2 * j + 1) * h.S1d_filled(i, j);
                    if (j >= 2)
                        tt -= 0.5 * j * (j - 1) * h.S1d_filled(i, j - 2);
                    return tt;
                };
                for (int ka = 0; ka < na_g; ++ka) {
                    const auto& pa =
                        a[static_cast<std::size_t>(ia_list[ka])];
                    for (int kb = 0; kb < nb_g; ++kb) {
                        const auto& pb =
                            b[static_cast<std::size_t>(ib_list[kb])];
                        const double sx =
                            hx_s.S1d_filled(pa.l[0], pb.l[0]);
                        const double sy =
                            hy_s.S1d_filled(pa.l[1], pb.l[1]);
                        const double sz =
                            hz_s.S1d_filled(pa.l[2], pb.l[2]);
                        const double tx =
                            kin1d(hx_s, pa.l[0], pb.l[0], pb.alpha);
                        const double ty =
                            kin1d(hy_s, pa.l[1], pb.l[1], pb.alpha);
                        const double tz =
                            kin1d(hz_s, pa.l[2], pb.l[2], pb.alpha);
                        const double c = pa.coeff * pb.coeff;
                        s += c * sx * sy * sz;
                        t += c
                             * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
                    }
                }
            }
        }
    }
    s_out = s;
    t_out = t;
}

/// Scalar pure-s multi-image ST (no heap, no SIMD). Used when the live image
/// count is below the SIMD break-even (≈ 2·native width on this host).
inline void contracted_ST_ss_images_scalar(const GridBasisFunction& a,
                                           const GridBasisFunction& b,
                                           const double L[3], const int m[3],
                                           double reach2, double& s_out,
                                           double& t_out) {
    const double* ac = a[0].center;
    const double* bc0 = b[0].center;
    double s = 0.0, t = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double dx = ac[0] - bc0[0] - tx * L[0];
                const double dy = ac[1] - bc0[1] - ty * L[1];
                const double dz = ac[2] - bc0[2] - tz * L[2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > reach2)
                    continue;
                for (const auto& pa : a) {
                    for (const auto& pb : b) {
                        const double p = pa.alpha + pb.alpha;
                        const double mu = pa.alpha * pb.alpha / p;
                        const double pi_over_p = M_PI / p;
                        const double S = pa.coeff * pb.coeff * pi_over_p
                                         * std::sqrt(pi_over_p)
                                         * std::exp(-mu * r2);
                        s += S;
                        t += mu * (3.0 - 2.0 * mu * r2) * S;
                    }
                }
            }
    s_out = s;
    t_out = t;
}

/// Pure s–s multi-image ST with precomputed live r². Image-axis xsimd; monom
/// pairs reuse each r² batch. Call only when nimg ≥ 2·W (slab-class).
inline void contracted_ST_ss_images_simd(const GridBasisFunction& a,
                                         const GridBasisFunction& b,
                                         const double* r2, int nimg,
                                         double& s_out, double& t_out) {
    using batch_t = xsimd::batch<double>;
    constexpr int W = static_cast<int>(batch_t::size);

    SsPairPack pack;
    pack.build(a, b);
    const int np = pack.np;
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    if (np != na * nb) {
        double s = 0.0, t = 0.0;
        for (int im = 0; im < nimg; ++im) {
            const double r = r2[im];
            for (const auto& pa : a) {
                for (const auto& pb : b) {
                    const double p = pa.alpha + pb.alpha;
                    const double mu = pa.alpha * pb.alpha / p;
                    const double pi_over_p = M_PI / p;
                    const double S = pa.coeff * pb.coeff * pi_over_p
                                     * std::sqrt(pi_over_p)
                                     * std::exp(-mu * r);
                    s += S;
                    t += mu * (3.0 - 2.0 * mu * r) * S;
                }
            }
        }
        s_out = s;
        t_out = t;
        return;
    }

    const batch_t three(3.0);
    const batch_t two(2.0);
    batch_t acc_s(0.0);
    batch_t acc_t(0.0);
    int im = 0;
    for (; im + W <= nimg; im += W) {
        const batch_t r2_v = batch_t::load_unaligned(r2 + im);
        for (int ip = 0; ip < np; ++ip) {
            const batch_t mu_v(pack.mu[ip]);
            const batch_t pref_v(pack.pref[ip]);
            const batch_t S = pref_v * xsimd::exp(-mu_v * r2_v);
            const batch_t T = mu_v * (three - two * mu_v * r2_v) * S;
            acc_s += S;
            acc_t += T;
        }
    }
    double s_lane[16];
    double t_lane[16];
    acc_s.store_unaligned(s_lane);
    acc_t.store_unaligned(t_lane);
    double s = 0.0, t = 0.0;
    for (int lane = 0; lane < W; ++lane) {
        s += s_lane[lane];
        t += t_lane[lane];
    }
    for (; im < nimg; ++im) {
        const double r = r2[im];
        for (int ip = 0; ip < np; ++ip) {
            const double S = pack.pref[ip] * std::exp(-pack.mu[ip] * r);
            s += S;
            t += pack.mu[ip] * (3.0 - 2.0 * pack.mu[ip] * r) * S;
        }
    }
    s_out = s;
    t_out = t;
}

/// Fused S+T over all live lattice images. Pure s–s and mixed-angular (L≤d)
/// use image-axis xsimd on slab-class image counts; otherwise scalar Hermite.
inline void contracted_ST_periodic_impl(const GridBasisFunction& a,
                                        const GridBasisFunction& b,
                                        const double L[3], const int m[3],
                                        double reach, double& s_out,
                                        double& t_out) {
    if (a.empty() || b.empty()) {
        s_out = 0.0;
        t_out = 0.0;
        return;
    }
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const double reach2 = reach * reach;
    using batch_t = xsimd::batch<double>;
    // SIMD break-even: ≥ one full native vector of live images (slab-class).
    constexpr int kSimdImgMin = static_cast<int>(batch_t::size);

    if (basis_is_pure_s(a) && basis_is_pure_s(b)) {
        const int nimg_est =
            (2 * m[0] + 1) * (2 * m[1] + 1) * (2 * m[2] + 1);
        // Tiny image windows (four fair twins): stay on scalar.
        if (nimg_est < kSimdImgMin) {
            contracted_ST_ss_images_scalar(a, b, L, m, reach2, s_out, t_out);
            return;
        }
        // Stack pack — GPW image half-counts stay modest; 512 covers
        // m ≤ (4,4,8) = 9·9·17. Truncation falls back to scalar.
        constexpr int kR2Max = 512;
        double r2_live[kR2Max];
        const int nimg =
            pack_live_images_r2(ac, bc, L, m, reach2, r2_live, kR2Max);
        if (nimg == 0) {
            s_out = 0.0;
            t_out = 0.0;
            return;
        }
        if (nimg >= kR2Max) {
            // Truncated pack — full scalar (no partial image set).
            contracted_ST_ss_images_scalar(a, b, L, m, reach2, s_out, t_out);
            return;
        }
        if (nimg < kSimdImgMin) {
            // Screened-down: scalar over the already-packed r² (do not rescreen).
            SsPairPack pack;
            pack.build(a, b);
            double s = 0.0, t = 0.0;
            for (int im = 0; im < nimg; ++im) {
                const double r = r2_live[im];
                for (int ip = 0; ip < pack.np; ++ip) {
                    const double S =
                        pack.pref[ip] * std::exp(-pack.mu[ip] * r);
                    s += S;
                    t += pack.mu[ip] * (3.0 - 2.0 * pack.mu[ip] * r) * S;
                }
            }
            s_out = s;
            t_out = t;
            return;
        }
        contracted_ST_ss_images_simd(a, b, r2_live, nimg, s_out, t_out);
        return;
    }

    // Mixed angular: slab-class image-axis Hermite xsimd when L ≤ d and
    // monoms share shell centers; otherwise per-image contracted_ST_shift.
    const int nimg_est = (2 * m[0] + 1) * (2 * m[1] + 1) * (2 * m[2] + 1);
    const int max_la = basis_max_cart_l(a);
    const int max_lb = basis_max_cart_l(b);
    const bool hermite_simd_ok =
        nimg_est >= kSimdImgMin && max_la <= 2 && max_lb <= 2
        && basis_same_center(a) && basis_same_center(b);

    if (hermite_simd_ok) {
        constexpr int kImgMax = 512;
        double dx_live[kImgMax], dy_live[kImgMax], dz_live[kImgMax];
        const int nimg = pack_live_images_dxyz(ac, bc, L, m, reach2, dx_live,
                                               dy_live, dz_live, kImgMax);
        if (nimg == 0) {
            s_out = 0.0;
            t_out = 0.0;
            return;
        }
        if (nimg < kImgMax && nimg >= kSimdImgMin) {
            contracted_ST_angular_images_simd(a, b, dx_live, dy_live, dz_live,
                                              nimg, s_out, t_out);
            return;
        }
        if (nimg < kImgMax) {
            double s = 0.0, t = 0.0;
            for (int im = 0; im < nimg; ++im) {
                const double T[3] = {ac[0] - bc[0] - dx_live[im],
                                     ac[1] - bc[1] - dy_live[im],
                                     ac[2] - bc[2] - dz_live[im]};
                double s_im = 0.0, t_im = 0.0;
                contracted_ST_shift(a, b, T, s_im, t_im);
                s += s_im;
                t += t_im;
            }
            s_out = s;
            t_out = t;
            return;
        }
    }

    // Per-image Hermite path (fill_upto + monom reuse).
    double s = 0.0, t = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                if (!image_in_reach(ac, bc, T, reach2))
                    continue;
                double s_im = 0.0, t_im = 0.0;
                contracted_ST_shift(a, b, T, s_im, t_im);
                s += s_im;
                t += t_im;
            }
    s_out = s;
    t_out = t;
}

} // namespace

double prim_overlap(const GridPrimitive& a, const GridPrimitive& b) {
    // Pure s–s: closed form (matches prim_ST).
    if (a.l[0] == 0 && a.l[1] == 0 && a.l[2] == 0 && b.l[0] == 0
        && b.l[1] == 0 && b.l[2] == 0) {
        double s = 0.0, t = 0.0;
        st_ss(a.alpha, b.alpha, a.center, b.center, a.coeff, b.coeff, s, t);
        (void)t;
        return s;
    }
    thread_local HermiteTab hx, hy, hz;
    thread_local HermiteGeomKey key;
    hermite_ensure(hx, hy, hz, key, a.alpha, b.alpha, a.center, b.center);
    return a.coeff * b.coeff * hx.S1d(a.l[0], b.l[0]) * hy.S1d(a.l[1], b.l[1])
           * hz.S1d(a.l[2], b.l[2]);
}

double prim_kinetic(const GridPrimitive& a, const GridPrimitive& b) {
    if (a.l[0] == 0 && a.l[1] == 0 && a.l[2] == 0 && b.l[0] == 0
        && b.l[1] == 0 && b.l[2] == 0) {
        double s = 0.0, t = 0.0;
        st_ss(a.alpha, b.alpha, a.center, b.center, a.coeff, b.coeff, s, t);
        (void)s;
        return t;
    }
    thread_local HermiteTab hx, hy, hz;
    thread_local HermiteGeomKey key;
    hermite_ensure(hx, hy, hz, key, a.alpha, b.alpha, a.center, b.center);
    const double sx = hx.S1d(a.l[0], b.l[0]);
    const double sy = hy.S1d(a.l[1], b.l[1]);
    const double sz = hz.S1d(a.l[2], b.l[2]);
    const double tx = kinetic_1d_tab(hx, a.l[0], b.l[0], b.alpha);
    const double ty = kinetic_1d_tab(hy, a.l[1], b.l[1], b.alpha);
    const double tz = kinetic_1d_tab(hz, a.l[2], b.l[2], b.alpha);
    return a.coeff * b.coeff * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
}

void prim_ST(const GridPrimitive& a, const GridPrimitive& b, double& s_out,
             double& t_out) {
    // s–s fast path: no Hermite table (does not touch geometry key).
    if (a.l[0] == 0 && a.l[1] == 0 && a.l[2] == 0 && b.l[0] == 0
        && b.l[1] == 0 && b.l[2] == 0) {
        st_ss(a.alpha, b.alpha, a.center, b.center, a.coeff, b.coeff, s_out,
              t_out);
        return;
    }
    thread_local HermiteTab hx, hy, hz;
    thread_local HermiteGeomKey key;
    hermite_ensure(hx, hy, hz, key, a.alpha, b.alpha, a.center, b.center);
    st_from_tabs(hx, hy, hz, a.l[0], a.l[1], a.l[2], b.l[0], b.l[1], b.l[2],
                 a.alpha, b.alpha, a.coeff, b.coeff, s_out, t_out);
}


void contracted_ST_shift(const GridBasisFunction& a,
                         const GridBasisFunction& b, const double T[3],
                         double& s_out, double& t_out) {
    // Pure s–s: portable xsimd over flattened monom pairs (shared r²).
    // Dominant on H shells and multi-k STVnl image sweeps that land on s.
    if (basis_is_pure_s(a) && basis_is_pure_s(b)) {
        // All monoms of a contracted s shell share the same center — shift
        // once from b[0] (MOLOPT / CP2K convention).
        const double bc[3] = {b[0].center[0] + T[0], b[0].center[1] + T[1],
                              b[0].center[2] + T[2]};
        contracted_ST_ss_simd(a, b, bc, s_out, t_out);
        return;
    }

    // Local Hermite + bottom-up fill per (α_a, α_b, geometry). Grouping monoms
    // that share exponents fills the E table once (kinetic needs j+2).
    HermiteTab hx, hy, hz;
    HermiteGeomKey key;
    double s = 0.0, t = 0.0;
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    for (int ia = 0; ia < na; ++ia) {
        const auto& pa = a[static_cast<std::size_t>(ia)];
        for (int ib = 0; ib < nb; ++ib) {
            const auto& pb = b[static_cast<std::size_t>(ib)];
            const double bc[3] = {pb.center[0] + T[0], pb.center[1] + T[1],
                                  pb.center[2] + T[2]};
            double s_p = 0.0, t_p = 0.0;
            if (pa.l[0] == 0 && pa.l[1] == 0 && pa.l[2] == 0 && pb.l[0] == 0
                && pb.l[1] == 0 && pb.l[2] == 0) {
                st_ss(pa.alpha, pb.alpha, pa.center, bc, pa.coeff, pb.coeff, s_p,
                      t_p);
                s += s_p;
                t += t_p;
                continue;
            }
            // On geometry miss: init + scan remaining monoms with same (α,α')
            // to learn max L for a single fill_upto.
            if (!key.matches(pa.alpha, pb.alpha, pa.center, bc)) {
                hx.init(pa.alpha, pb.alpha, pa.center[0], bc[0]);
                hy.init(pa.alpha, pb.alpha, pa.center[1], bc[1]);
                hz.init(pa.alpha, pb.alpha, pa.center[2], bc[2]);
                key.set(pa.alpha, pb.alpha, pa.center, bc);
                int max_la = 0, max_lb = 0;
                for (int ja = ia; ja < na; ++ja) {
                    const auto& qa = a[static_cast<std::size_t>(ja)];
                    if (qa.alpha != pa.alpha)
                        continue;
                    if (qa.center[0] != pa.center[0] || qa.center[1] != pa.center[1]
                        || qa.center[2] != pa.center[2])
                        continue;
                    max_la = std::max(
                        max_la, std::max(qa.l[0], std::max(qa.l[1], qa.l[2])));
                }
                for (int jb = 0; jb < nb; ++jb) {
                    const auto& qb = b[static_cast<std::size_t>(jb)];
                    if (qb.alpha != pb.alpha)
                        continue;
                    max_lb = std::max(
                        max_lb, std::max(qb.l[0], std::max(qb.l[1], qb.l[2])));
                }
                // Kinetic uses S1d(i, j+2) ⇒ need j up to max_lb+2.
                const int jfill = std::min(kMaxAng, max_lb + 2);
                hx.fill_upto(max_la, jfill);
                hy.fill_upto(max_la, jfill);
                hz.fill_upto(max_la, jfill);
            }
            // Direct filled-table ST (no recursive E).
            auto kin1d = [](const HermiteTab& h, int i, int j, double b_exp) {
                double tt = -2.0 * b_exp * b_exp * h.S1d_filled(i, j + 2)
                            + b_exp * (2 * j + 1) * h.S1d_filled(i, j);
                if (j >= 2)
                    tt -= 0.5 * j * (j - 1) * h.S1d_filled(i, j - 2);
                return tt;
            };
            const double sx = hx.S1d_filled(pa.l[0], pb.l[0]);
            const double sy = hy.S1d_filled(pa.l[1], pb.l[1]);
            const double sz = hz.S1d_filled(pa.l[2], pb.l[2]);
            const double tx = kin1d(hx, pa.l[0], pb.l[0], pb.alpha);
            const double ty = kin1d(hy, pa.l[1], pb.l[1], pb.alpha);
            const double tz = kin1d(hz, pa.l[2], pb.l[2], pb.alpha);
            const double c = pa.coeff * pb.coeff;
            s += c * sx * sy * sz;
            t += c * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
        }
    }
    s_out = s;
    t_out = t;
}


double overlap(const GridBasisFunction& a, const GridBasisFunction& b) {
    if (basis_is_pure_s(a) && basis_is_pure_s(b)) {
        double s = 0.0, t = 0.0;
        contracted_ST_ss_simd(a, b, b[0].center, s, t);
        (void)t;
        return s;
    }
    double s = 0.0;
    for (const auto& pa : a)
        for (const auto& pb : b)
            s += prim_overlap(pa, pb);
    return s;
}

double kinetic(const GridBasisFunction& a, const GridBasisFunction& b) {
    if (basis_is_pure_s(a) && basis_is_pure_s(b)) {
        double s = 0.0, t = 0.0;
        contracted_ST_ss_simd(a, b, b[0].center, s, t);
        (void)s;
        return t;
    }
    double t = 0.0;
    for (const auto& pa : a)
        for (const auto& pb : b)
            t += prim_kinetic(pa, pb);
    return t;
}

void contracted_ST_periodic(const GridBasisFunction& a,
                            const GridBasisFunction& b, const double L[3],
                            const int m[3], double reach, double& s_out,
                            double& t_out) {
    contracted_ST_periodic_impl(a, b, L, m, reach, s_out, t_out);
}

double overlap(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach) {
    double s = 0.0, t = 0.0;
    contracted_ST_periodic_impl(a, b, L, m, reach, s, t);
    (void)t;
    return s;
}

double kinetic(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach) {
    double s = 0.0, t = 0.0;
    contracted_ST_periodic_impl(a, b, L, m, reach, s, t);
    (void)s;
    return t;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
