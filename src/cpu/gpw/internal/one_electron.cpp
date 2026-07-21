/// @file src/cpu/gpw/internal/one_electron.cpp
/// @brief McMurchie-Davidson overlap and kinetic integrals over GridPrimitives.

#include "atomiverse/cpu/gpw/internal/one_electron.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

/// @brief Hermite expansion coefficient E_t^{i,j} for one Cartesian direction
///        (McMurchie-Davidson). `p` is the combined exponent, `mu = a*b/p`,
///        `xab = A-B`, `xpa = P-A`, `xpb = P-B` along this direction.
///
/// Recurrence (Helgaker, Molecular Electronic-Structure Theory, eq. 9.5.6-7):
///   E_t^{0,0} = exp(-mu xab^2),
///   E_t^{i+1,j} = 1/(2p) E_{t-1}^{i,j} + xpa E_t^{i,j} + (t+1) E_{t+1}^{i,j},
///   E_t^{i,j+1} = 1/(2p) E_{t-1}^{i,j} + xpb E_t^{i,j} + (t+1) E_{t+1}^{i,j},
/// with E_t = 0 for t < 0 or t > i+j. Recursion depth is small (l <= a few).
double hermite_E(int i, int j, int t, double p, double mu, double xab,
                 double xpa, double xpb) {
    if (t < 0 || t > i + j)
        return 0.0;
    if (i == 0 && j == 0)
        return std::exp(-mu * xab * xab); // t == 0 here
    const double oo2p = 1.0 / (2.0 * p);
    if (j == 0) {
        // decrement i
        return oo2p * hermite_E(i - 1, j, t - 1, p, mu, xab, xpa, xpb)
               + xpa * hermite_E(i - 1, j, t, p, mu, xab, xpa, xpb)
               + (t + 1) * hermite_E(i - 1, j, t + 1, p, mu, xab, xpa, xpb);
    }
    // decrement j
    return oo2p * hermite_E(i, j - 1, t - 1, p, mu, xab, xpa, xpb)
           + xpb * hermite_E(i, j - 1, t, p, mu, xab, xpa, xpb)
           + (t + 1) * hermite_E(i, j - 1, t + 1, p, mu, xab, xpa, xpb);
}

/// @brief 1D overlap S^{i,j} = E_0^{i,j} sqrt(pi/p) for one Cartesian direction.
double overlap_1d(int i, int j, double a, double b, double Ax, double Bx) {
    if (i < 0 || j < 0)
        return 0.0;
    const double p = a + b;
    const double mu = a * b / p;
    const double Px = (a * Ax + b * Bx) / p;
    const double E0 = hermite_E(i, j, 0, p, mu, Ax - Bx, Px - Ax, Px - Bx);
    return E0 * std::sqrt(M_PI / p);
}

/// @brief 1D kinetic block T^{i,j} along one direction (Helgaker eq. 9.3.41):
///   T_ij = -2 b^2 S_{i,j+2} + b(2j+1) S_{i,j} - 1/2 j(j-1) S_{i,j-2}.
double kinetic_1d(int i, int j, double a, double b, double Ax, double Bx) {
    double t = -2.0 * b * b * overlap_1d(i, j + 2, a, b, Ax, Bx)
               + b * (2 * j + 1) * overlap_1d(i, j, a, b, Ax, Bx);
    if (j >= 2)
        t -= 0.5 * j * (j - 1) * overlap_1d(i, j - 2, a, b, Ax, Bx);
    return t;
}

} // namespace

double prim_overlap(const GridPrimitive& a, const GridPrimitive& b) {
    const double sx =
        overlap_1d(a.l[0], b.l[0], a.alpha, b.alpha, a.center[0], b.center[0]);
    const double sy =
        overlap_1d(a.l[1], b.l[1], a.alpha, b.alpha, a.center[1], b.center[1]);
    const double sz =
        overlap_1d(a.l[2], b.l[2], a.alpha, b.alpha, a.center[2], b.center[2]);
    return a.coeff * b.coeff * sx * sy * sz;
}

double prim_kinetic(const GridPrimitive& a, const GridPrimitive& b) {
    const double sx =
        overlap_1d(a.l[0], b.l[0], a.alpha, b.alpha, a.center[0], b.center[0]);
    const double sy =
        overlap_1d(a.l[1], b.l[1], a.alpha, b.alpha, a.center[1], b.center[1]);
    const double sz =
        overlap_1d(a.l[2], b.l[2], a.alpha, b.alpha, a.center[2], b.center[2]);
    const double tx =
        kinetic_1d(a.l[0], b.l[0], a.alpha, b.alpha, a.center[0], b.center[0]);
    const double ty =
        kinetic_1d(a.l[1], b.l[1], a.alpha, b.alpha, a.center[1], b.center[1]);
    const double tz =
        kinetic_1d(a.l[2], b.l[2], a.alpha, b.alpha, a.center[2], b.center[2]);
    return a.coeff * b.coeff * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
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
// True if the image-shifted center of `b` is within `reach` of `a`'s center, so
// some primitive pair can contribute. `ac`/`bc` are the function centers.
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
                        GridPrimitive q = pb; // image of b: center += T
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
