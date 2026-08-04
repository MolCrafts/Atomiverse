/// @file src/cpu/gpw/internal/one_electron.cpp
/// @brief McMurchie-Davidson overlap and kinetic integrals over GridPrimitives.
///
/// Hot path uses a memoized Hermite table per Cartesian axis (no exponential
/// recursion tree) and a fused prim_ST that shares 1D overlaps between S and T.

#include "atomiverse/cpu/gpw/internal/one_electron.h"

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
    }

    double E(int i, int j, int t) {
        if (t < 0 || t > i + j || i < 0 || j < 0)
            return 0.0;
        if (i > kMaxAng || j > kMaxAng)
            return 0.0; // refuse — caller should not hit this for GPW
        if (stamp[i][j][t] == gen)
            return memo[i][j][t];
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
};

/// 1D kinetic with known b (ket exponent).
inline double kinetic_1d_tab(HermiteTab& h, int i, int j, double b) {
    double t = -2.0 * b * b * h.S1d(i, j + 2)
               + b * (2 * j + 1) * h.S1d(i, j);
    if (j >= 2)
        t -= 0.5 * j * (j - 1) * h.S1d(i, j - 2);
    return t;
}

} // namespace

double prim_overlap(const GridPrimitive& a, const GridPrimitive& b) {
    thread_local HermiteTab hx, hy, hz;
    hx.init(a.alpha, b.alpha, a.center[0], b.center[0]);
    hy.init(a.alpha, b.alpha, a.center[1], b.center[1]);
    hz.init(a.alpha, b.alpha, a.center[2], b.center[2]);
    return a.coeff * b.coeff * hx.S1d(a.l[0], b.l[0]) * hy.S1d(a.l[1], b.l[1])
           * hz.S1d(a.l[2], b.l[2]);
}

double prim_kinetic(const GridPrimitive& a, const GridPrimitive& b) {
    thread_local HermiteTab hx, hy, hz;
    hx.init(a.alpha, b.alpha, a.center[0], b.center[0]);
    hy.init(a.alpha, b.alpha, a.center[1], b.center[1]);
    hz.init(a.alpha, b.alpha, a.center[2], b.center[2]);
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
    // s–s fast path: no Hermite table.
    if (a.l[0] == 0 && a.l[1] == 0 && a.l[2] == 0 && b.l[0] == 0
        && b.l[1] == 0 && b.l[2] == 0) {
        const double p = a.alpha + b.alpha;
        const double mu = a.alpha * b.alpha / p;
        const double dx = a.center[0] - b.center[0];
        const double dy = a.center[1] - b.center[1];
        const double dz = a.center[2] - b.center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double S =
            a.coeff * b.coeff * std::pow(M_PI / p, 1.5) * std::exp(-mu * r2);
        s_out = S;
        // ⟨a|−½∇²|b⟩ for s–s = μ(3 − 2μ r²) S  (same as 3-axis kinetic_1d).
        t_out = mu * (3.0 - 2.0 * mu * r2) * S;
        return;
    }
    thread_local HermiteTab hx, hy, hz;
    hx.init(a.alpha, b.alpha, a.center[0], b.center[0]);
    hy.init(a.alpha, b.alpha, a.center[1], b.center[1]);
    hz.init(a.alpha, b.alpha, a.center[2], b.center[2]);
    const double sx = hx.S1d(a.l[0], b.l[0]);
    const double sy = hy.S1d(a.l[1], b.l[1]);
    const double sz = hz.S1d(a.l[2], b.l[2]);
    const double tx = kinetic_1d_tab(hx, a.l[0], b.l[0], b.alpha);
    const double ty = kinetic_1d_tab(hy, a.l[1], b.l[1], b.alpha);
    const double tz = kinetic_1d_tab(hz, a.l[2], b.l[2], b.alpha);
    const double c = a.coeff * b.coeff;
    s_out = c * sx * sy * sz;
    t_out = c * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
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
