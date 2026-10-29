/// @file core/pppm_ik.h
/// @brief Hockney–Eastwood PPPM influence for ik differentiation.
///
/// Orthorhombic and triclinic cells. The scalar G(k) is the ik Green's
/// function (LAMMPS `PPPM::compute_gf_ik` / `compute_gf_ik_triclinic`):
/// assignment window [sinc]^(2·order), analytic denominator, and the
/// alias sum truncated where the Gaussian falls below 1e-7. The alias
/// bound is the LAMMPS truncation, with no extra image.
///
/// k = 2π H^{-T} h, with lattice vectors stored as columns of H and
/// `inv` the row-major inverse (s = inv · r). Forces that consume this
/// G are ik forces: Ê(k) = i k G(k) ρ̂(k), interpolated with the same
/// B-spline θ that spread the charge. This header does not apply that
/// differentiation.
#pragma once

#include <cmath>

#include "atomiverse/core/units.h"

#if defined(__CUDACC__)
#define ATV_PPPM_HD __host__ __device__
#else
#define ATV_PPPM_HD
#endif

namespace ATV::longrange {

/// @brief LAMMPS mode index `i - K*(2*i/K)`. Nyquist of an even mesh is −K/2.
ATV_PPPM_HD inline int pppm_signed_mode(int m, int mesh) {
    return m - mesh * ((2 * m) / mesh);
}

ATV_PPPM_HD inline double pppm_powsinxx(double arg, int twoorder) {
    if (::fabs(arg) < 1e-12)
        return 1.0;
    const double s = ::sin(arg) / arg;
    double p = 1.0;
    for (int i = 0; i < twoorder; ++i)
        p *= s;
    return p;
}

/// @brief Γ(2n) coefficients of the PPPM denominator. `gf_b` has `order` slots.
inline void pppm_gf_b_fill(int order, double* gf_b) {
    for (int l = 0; l < order; ++l)
        gf_b[l] = 0.0;
    if (order < 1)
        return;
    gf_b[0] = 1.0;
    for (int m = 1; m < order; ++m) {
        for (int l = m; l > 0; --l)
            gf_b[l] = 4.0
                      * (gf_b[l] * (l - m) * (l - m - 0.5)
                         - gf_b[l - 1] * (l - m - 1) * (l - m - 1));
        gf_b[0] = 4.0 * (gf_b[0] * (0 - m) * (0 - m - 0.5));
    }
    long double ifact = 1;
    for (int k = 1; k < 2 * order; ++k)
        ifact *= k;
    const double gaminv = static_cast<double>(1.0L / ifact);
    for (int l = 0; l < order; ++l)
        gf_b[l] *= gaminv;
}

ATV_PPPM_HD inline double pppm_gf_denom_scalar(const double* gf_b, int order,
                                               double x, double y, double z) {
    double sx = 0, sy = 0, sz = 0;
    double xpow = 1, ypow = 1, zpow = 1;
    for (int l = 0; l < order; ++l) {
        sx += gf_b[l] * xpow;
        sy += gf_b[l] * ypow;
        sz += gf_b[l] * zpow;
        xpow *= x;
        ypow *= y;
        zpow *= z;
    }
    // LAMMPS `PPPM::gf_denom` returns (Sx Sy Sz)². The polynomial is
    // Σ W²; the ik influence divides by |Σ W²|².
    const double s = sx * sy * sz;
    return s * s;
}

/// @brief Cartesian wavevector of integer mode (h0, h1, h2) [1/Å].
ATV_PPPM_HD inline void pppm_wavevector(int h0, int h1, int h2,
                                        const double* inv, double k[3]) {
    const int h[3] = {h0, h1, h2};
    const double twopi = 2.0 * ATV::units::pi;
    for (int r = 0; r < 3; ++r) {
        double m = 0.0;
        for (int d = 0; d < 3; ++d)
            m += static_cast<double>(h[d]) * inv[3 * d + r];
        k[r] = twopi * m;
    }
}

/// @brief One ik Green's function value. Zero mode and a vanished k return 0.
///
/// @param L   Real-space period along each lattice vector [Å], alias cutoff only.
/// @param inv Row-major inverse cell.
ATV_PPPM_HD inline double pppm_green_ik(int h0, int h1, int h2,
                                        const double L[3], const int K[3],
                                        double alpha, int order,
                                        const double* gf_b, const double* inv) {
    if (h0 == 0 && h1 == 0 && h2 == 0)
        return 0.0;
    double k[3];
    pppm_wavevector(h0, h1, h2, inv, k);
    const double sqk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
    if (sqk < 1e-30)
        return 0.0;
    const double pi = ATV::units::pi;
    const int h[3] = {h0, h1, h2};
    const double sn0 = ::sin(pi * static_cast<double>(h[0]) / K[0]);
    const double sn1 = ::sin(pi * static_cast<double>(h[1]) / K[1]);
    const double sn2 = ::sin(pi * static_cast<double>(h[2]) / K[2]);
    const double denominator =
        pppm_gf_denom_scalar(gf_b, order, sn0 * sn0, sn1 * sn1, sn2 * sn2);
    if (!(denominator > 1e-30))
        return 0.0;

    constexpr double kEpsHoc = 1.0e-7;
    const double hoc = ::pow(-::log(kEpsHoc), 0.25);
    int nb[3];
    for (int d = 0; d < 3; ++d) {
        const double bound =
            (alpha * L[d] / (pi * static_cast<double>(K[d]))) * hoc;
        int n = static_cast<int>(bound);
        if (n < 0)
            n = 0;
        nb[d] = n;
    }
    const int twoorder = 2 * order;
    const double twopi = 2.0 * pi;
    double sum1 = 0.0;
    for (int nx = -nb[0]; nx <= nb[0]; ++nx) {
        const double argx =
            pi * (static_cast<double>(h0) / K[0] + static_cast<double>(nx));
        const double wx = pppm_powsinxx(argx, twoorder);
        for (int ny = -nb[1]; ny <= nb[1]; ++ny) {
            const double argy =
                pi * (static_cast<double>(h1) / K[1] + static_cast<double>(ny));
            const double wy = pppm_powsinxx(argy, twoorder);
            for (int nz = -nb[2]; nz <= nb[2]; ++nz) {
                const double argz = pi
                                    * (static_cast<double>(h2) / K[2]
                                       + static_cast<double>(nz));
                const double wz = pppm_powsinxx(argz, twoorder);
                const int nalias[3] = {nx, ny, nz};
                double q[3] = {k[0], k[1], k[2]};
                for (int d = 0; d < 3; ++d) {
                    const double shift = twopi * static_cast<double>(K[d])
                                         * static_cast<double>(nalias[d]);
                    for (int r = 0; r < 3; ++r)
                        q[r] += shift * inv[3 * d + r];
                }
                const double q2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
                if (q2 < 1e-30)
                    continue;
                const double gauss = ::exp(-0.25 * q2 / (alpha * alpha));
                const double dot1 = k[0] * q[0] + k[1] * q[1] + k[2] * q[2];
                sum1 += (dot1 / q2) * gauss * wx * wy * wz;
            }
        }
    }
    return (4.0 * pi / sqk) * sum1 / denominator;
}

/// @brief Fill an r2c half-spectrum G(k). Layout idx = m2 + n2·(m1 + K1·m0).
inline void pppm_fill_green_r2c(const int K[3], const double L[3],
                                const double* inv, double alpha, int order,
                                double* green) {
    double gf_b[12] = {};
    pppm_gf_b_fill(order, gf_b);
    const int n2 = K[2] / 2 + 1;
    for (int m0 = 0; m0 < K[0]; ++m0) {
        const int h0 = pppm_signed_mode(m0, K[0]);
        for (int m1 = 0; m1 < K[1]; ++m1) {
            const int h1 = pppm_signed_mode(m1, K[1]);
            for (int m2 = 0; m2 < n2; ++m2) {
                const int h2 = pppm_signed_mode(m2, K[2]);
                const int idx = m2 + n2 * (m1 + K[1] * m0);
                green[idx] =
                    pppm_green_ik(h0, h1, h2, L, K, alpha, order, gf_b, inv);
            }
        }
    }
}

} // namespace ATV::longrange

#undef ATV_PPPM_HD
