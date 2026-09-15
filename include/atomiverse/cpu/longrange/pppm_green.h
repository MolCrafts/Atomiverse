/// @file cpu/longrange/pppm_green.h
/// @brief Hockney–Eastwood PPPM Green's function (orthorhombic, ik/ad-ready).
///
/// After LAMMPS `PPPM::compute_gf_ik` / `gf_denom` (src/KSPACE/pppm.cpp).
/// This is the particle-mesh influence that distinguishes **true PPPM** from
/// Essmann smooth PME (B-spline Euler modulus only).
///
/// Orthorhombic cell, analytic differentiation force path (AD): greensfn
/// multiplies ρ̂(k) to get V̂(k); forces from ∇V via B-spline dθ (same
/// interpolate as PME). Energy: (1/2) Σ_k G(k) |ρ̂|² with FFTW unitary
/// conventions accounted for by the 1/N_grid² factor.
#pragma once

#include <cmath>
#include <vector>

#include "atomiverse/core/units.h"

namespace ATV {
namespace longrange {

/// @brief Expand coeffs Γ(2n) for the PPPM denominator (LAMMPS gf_b).
inline void pppm_gf_denom_coeffs(int order, std::vector<double>& gf_b) {
    gf_b.assign(static_cast<std::size_t>(order), 0.0);
    if (order < 1)
        return;
    gf_b[0] = 1.0;
    for (int m = 1; m < order; ++m) {
        for (int l = m; l > 0; --l)
            gf_b[static_cast<std::size_t>(l)] =
                4.0
                * (gf_b[static_cast<std::size_t>(l)] * (l - m) * (l - m - 0.5)
                   - gf_b[static_cast<std::size_t>(l - 1)] * (l - m - 1)
                         * (l - m - 1));
        gf_b[0] = 4.0 * (gf_b[0] * (0 - m) * (0 - m - 0.5));
    }
    long double ifact = 1;
    for (int k = 1; k < 2 * order; ++k)
        ifact *= k;
    const double gaminv = static_cast<double>(1.0L / ifact);
    for (int l = 0; l < order; ++l)
        gf_b[static_cast<std::size_t>(l)] *= gaminv;
}

inline double pppm_gf_denom(const std::vector<double>& gf_b, double x, double y,
                            double z) {
    double sx = 0, sy = 0, sz = 0;
    double xpow = 1, ypow = 1, zpow = 1;
    for (std::size_t l = 0; l < gf_b.size(); ++l) {
        sx += gf_b[l] * xpow;
        sy += gf_b[l] * ypow;
        sz += gf_b[l] * zpow;
        xpow *= x;
        ypow *= y;
        zpow *= z;
    }
    return sx * sy * sz;
}

inline double pppm_powsinxx(double arg, int twoorder) {
    if (std::abs(arg) < 1e-12)
        return 1.0;
    const double s = std::sin(arg) / arg;
    double p = 1.0;
    for (int i = 0; i < twoorder; ++i)
        p *= s;
    return p;
}

/// @brief Build full-grid Hockney–Eastwood G(k) for orthorhombic box.
///
/// Layout matches complex FFTW c2c: idx = k2 + K2*(k1 + K1*k0).
/// For r2c half-spectrum, call #pppm_build_green_r2c instead.
///
/// @param K          Mesh dimensions.
/// @param L          Box lengths [Å].
/// @param alpha      Ewald α (g_ewald) [1/Å].
/// @param order      Assignment order (stencil).
/// @param[out] green Length K0·K1·K2; zero mode = 0.
inline void pppm_build_green_c2c(const int K[3], const double L[3],
                                 double alpha, int order,
                                 std::vector<double>& green) {
    const int K0 = K[0], K1 = K[1], K2 = K[2];
    const int ntot = K0 * K1 * K2;
    green.assign(static_cast<std::size_t>(ntot), 0.0);

    std::vector<double> gf_b;
    pppm_gf_denom_coeffs(order, gf_b);

    const double twopi = 2.0 * ATV::units::pi;
    const double unitk[3] = {twopi / L[0], twopi / L[1], twopi / L[2]};
    constexpr double kEpsHoc = 1e-7;
    const double hoc = std::pow(-std::log(kEpsHoc), 0.25);
    const int nb[3] = {
        static_cast<int>((alpha * L[0] / (ATV::units::pi * K0)) * hoc) + 1,
        static_cast<int>((alpha * L[1] / (ATV::units::pi * K1)) * hoc) + 1,
        static_cast<int>((alpha * L[2] / (ATV::units::pi * K2)) * hoc) + 1};
    const int twoorder = 2 * order;
    const double four_pi = 4.0 * ATV::units::pi;

    for (int m0 = 0; m0 < K0; ++m0) {
        const int h0 = (m0 <= K0 / 2) ? m0 : m0 - K0;
        const double sn0 = std::sin(0.5 * unitk[0] * h0 * L[0] / K0);
        const double snx = sn0 * sn0;
        for (int m1 = 0; m1 < K1; ++m1) {
            const int h1 = (m1 <= K1 / 2) ? m1 : m1 - K1;
            const double sn1 = std::sin(0.5 * unitk[1] * h1 * L[1] / K1);
            const double sny = sn1 * sn1;
            for (int m2 = 0; m2 < K2; ++m2) {
                const int h2 = (m2 <= K2 / 2) ? m2 : m2 - K2;
                const int idx = m2 + K2 * (m1 + K1 * m0);
                if (h0 == 0 && h1 == 0 && h2 == 0) {
                    green[static_cast<std::size_t>(idx)] = 0.0;
                    continue;
                }
                const double sn2 = std::sin(0.5 * unitk[2] * h2 * L[2] / K2);
                const double snz = sn2 * sn2;
                const double kx = unitk[0] * h0;
                const double ky = unitk[1] * h1;
                const double kz = unitk[2] * h2;
                const double sqk = kx * kx + ky * ky + kz * kz;
                const double numerator = four_pi / sqk;
                const double denominator = pppm_gf_denom(gf_b, snx, sny, snz);

                double sum1 = 0.0;
                for (int nx = -nb[0]; nx <= nb[0]; ++nx) {
                    const double qx = unitk[0] * (h0 + K0 * nx);
                    const double sx =
                        std::exp(-0.25 * (qx / alpha) * (qx / alpha));
                    const double argx = 0.5 * qx * L[0] / K0;
                    const double wx = pppm_powsinxx(argx, twoorder);
                    for (int ny = -nb[1]; ny <= nb[1]; ++ny) {
                        const double qy = unitk[1] * (h1 + K1 * ny);
                        const double sy =
                            std::exp(-0.25 * (qy / alpha) * (qy / alpha));
                        const double argy = 0.5 * qy * L[1] / K1;
                        const double wy = pppm_powsinxx(argy, twoorder);
                        for (int nz = -nb[2]; nz <= nb[2]; ++nz) {
                            const double qz = unitk[2] * (h2 + K2 * nz);
                            const double sz =
                                std::exp(-0.25 * (qz / alpha) * (qz / alpha));
                            const double argz = 0.5 * qz * L[2] / K2;
                            const double wz = pppm_powsinxx(argz, twoorder);
                            const double dot1 = kx * qx + ky * qy + kz * qz;
                            const double dot2 = qx * qx + qy * qy + qz * qz;
                            if (dot2 < 1e-30)
                                continue;
                            sum1 += (dot1 / dot2) * sx * sy * sz * wx * wy * wz;
                        }
                    }
                }
                green[static_cast<std::size_t>(idx)] =
                    denominator > 1e-30 ? (numerator * sum1 / denominator)
                                        : 0.0;
            }
        }
    }
}

/// @brief AD-style green (LAMMPS `compute_gf_ad`): single-image Hockney–
///        Eastwood without the reciprocal-space alias sum. Matches analytic-
///        differentiation force paths and B-spline / PME-like assignment
///        better than the full ik alias sum on moderate meshes.
inline void pppm_build_green_ad(const int K[3], const double L[3], double alpha,
                                int order, std::vector<double>& green) {
    const int K0 = K[0], K1 = K[1], K2 = K[2];
    const int ntot = K0 * K1 * K2;
    green.assign(static_cast<std::size_t>(ntot), 0.0);

    std::vector<double> gf_b;
    pppm_gf_denom_coeffs(order, gf_b);

    const double twopi = 2.0 * ATV::units::pi;
    const double unitk[3] = {twopi / L[0], twopi / L[1], twopi / L[2]};
    const int twoorder = 2 * order;
    const double four_pi = 4.0 * ATV::units::pi;

    for (int m0 = 0; m0 < K0; ++m0) {
        const int h0 = (m0 <= K0 / 2) ? m0 : m0 - K0;
        const double kx = unitk[0] * h0;
        const double snx = std::sin(0.5 * unitk[0] * h0 * L[0] / K0);
        const double snx2 = snx * snx;
        const double sx = std::exp(-0.25 * (kx / alpha) * (kx / alpha));
        const double wx = pppm_powsinxx(0.5 * kx * L[0] / K0, twoorder);
        for (int m1 = 0; m1 < K1; ++m1) {
            const int h1 = (m1 <= K1 / 2) ? m1 : m1 - K1;
            const double ky = unitk[1] * h1;
            const double sny = std::sin(0.5 * unitk[1] * h1 * L[1] / K1);
            const double sny2 = sny * sny;
            const double sy = std::exp(-0.25 * (ky / alpha) * (ky / alpha));
            const double wy = pppm_powsinxx(0.5 * ky * L[1] / K1, twoorder);
            for (int m2 = 0; m2 < K2; ++m2) {
                const int h2 = (m2 <= K2 / 2) ? m2 : m2 - K2;
                const int idx = m2 + K2 * (m1 + K1 * m0);
                if (h0 == 0 && h1 == 0 && h2 == 0) {
                    green[static_cast<std::size_t>(idx)] = 0.0;
                    continue;
                }
                const double kz = unitk[2] * h2;
                const double snz = std::sin(0.5 * unitk[2] * h2 * L[2] / K2);
                const double snz2 = snz * snz;
                const double sz = std::exp(-0.25 * (kz / alpha) * (kz / alpha));
                const double wz = pppm_powsinxx(0.5 * kz * L[2] / K2, twoorder);
                const double sqk = kx * kx + ky * ky + kz * kz;
                const double numerator = four_pi / sqk;
                const double denominator =
                    pppm_gf_denom(gf_b, snx2, sny2, snz2);
                const double window = sx * sy * sz * wx * wy * wz;
                green[static_cast<std::size_t>(idx)] =
                    denominator > 1e-30 ? (numerator * window / denominator)
                                        : 0.0;
            }
        }
    }
}

} // namespace longrange
} // namespace ATV
