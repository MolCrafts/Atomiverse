/// @file src/cpu/gpw/internal/one_electron.cpp
/// @brief McMurchie-Davidson overlap and kinetic integrals over GridPrimitives.
///
/// Hot path uses a memoized Hermite table per Cartesian axis (no exponential
/// recursion tree) and a fused prim_ST that shares 1D overlaps between S and T.
/// Geometry-key reuse keeps tables live across consecutive prim pairs that share
/// (α, center) — common for solid-harmonic cart terms and shell-mate AOs.

#include "atomiverse/cpu/gpw/internal/one_electron.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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

} // namespace

double prim_overlap(const GridPrimitive& a, const GridPrimitive& b) {
    thread_local HermiteTab hx, hy, hz;
    thread_local HermiteGeomKey key;
    hermite_ensure(hx, hy, hz, key, a.alpha, b.alpha, a.center, b.center);
    return a.coeff * b.coeff * hx.S1d(a.l[0], b.l[0]) * hy.S1d(a.l[1], b.l[1])
           * hz.S1d(a.l[2], b.l[2]);
}

double prim_kinetic(const GridPrimitive& a, const GridPrimitive& b) {
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
    // Local Hermite + bottom-up fill per (α_a, α_b, geometry). Grouping monoms
    // that share exponents fills the E table once (kinetic needs j+2).
    HermiteTab hx, hy, hz;
    HermiteGeomKey key;
    double s = 0.0, t = 0.0;
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    // Track max angular per current geometry so fill_upto covers all monoms.
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
    double s = 0.0;
    for (const auto& pa : a)
        for (const auto& pb : b)
            s += prim_overlap(pa, pb);
    return s;
}

double kinetic(const GridBasisFunction& a, const GridBasisFunction& b) {
    double t = 0.0;
    for (const auto& pa : a)
        for (const auto& pb : b)
            t += prim_kinetic(pa, pb);
    return t;
}

namespace {
inline bool image_in_reach(const double ac[3], const double bc[3],
                           const double T[3], double reach2) {
    const double dx = ac[0] - bc[0] - T[0];
    const double dy = ac[1] - bc[1] - T[1];
    const double dz = ac[2] - bc[2] - T[2];
    return dx * dx + dy * dy + dz * dz <= reach2;
}
} // namespace

double overlap(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach) {
    if (a.empty() || b.empty())
        return 0.0;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const double reach2 = reach * reach;
    double s = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                if (!image_in_reach(ac, bc, T, reach2))
                    continue;
                for (const auto& pa : a)
                    for (const auto& pb : b) {
                        GridPrimitive q = pb;
                        q.center[0] += T[0];
                        q.center[1] += T[1];
                        q.center[2] += T[2];
                        s += prim_overlap(pa, q);
                    }
            }
    return s;
}

double kinetic(const GridBasisFunction& a, const GridBasisFunction& b,
               const double L[3], const int m[3], double reach) {
    if (a.empty() || b.empty())
        return 0.0;
    const double* ac = a[0].center;
    const double* bc = b[0].center;
    const double reach2 = reach * reach;
    double t = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                if (!image_in_reach(ac, bc, T, reach2))
                    continue;
                for (const auto& pa : a)
                    for (const auto& pb : b) {
                        GridPrimitive q = pb;
                        q.center[0] += T[0];
                        q.center[1] += T[1];
                        q.center[2] += T[2];
                        t += prim_kinetic(pa, q);
                    }
            }
    return t;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
