/// @file tests/core/pppm_lammps_reference.h
/// @brief LAMMPS ik PPPM as an independent test reference: charge
///        assignment, influence function and the ik field, on a small
///        periodic mesh with naive (separable) DFTs.
///
/// Literal ports of LAMMPS `src/KSPACE/pppm.cpp` / `pppm.h`, checkout
/// e8c4c278a8: `compute_rho_coeff`, `compute_rho1d`, `particle_map` +
/// `make_rho` (`shift`, `shiftone`, `nlower`, `nupper` from
/// `set_grid_local`), `compute_gf_denom`, `gf_denom`, `compute_gf_ik` and
/// `compute_gf_ik_triclinic` (one mode each), `fieldforce_ik`.
/// LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.
///
/// The weights come from LAMMPS's piecewise polynomial table, not from the
/// B-spline recursion the engines use, and the transforms are plain sums, so
/// a match pins the stencil placement and the weights, not shared code.
/// Normalization follows the Atomiverse convention (ρ̂ the unnormalized
/// transform of q·W, E = k_e/(2V) Σ_k G |ρ̂|², field = Σ_k i k (G/V) ρ̂ e^{ik·r}),
/// which the Madelung / Ewald tests pin separately. LAMMPS's 4π literal
/// (12.5663706) is a parameter so the influence can be compared both ways.
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "atomiverse/core/units.h"

namespace ATV::test::lammps_pppm {

inline constexpr int kOffset = 16384;               ///< LAMMPS OFFSET
inline constexpr double kEpsHoc = 1.0e-7;           ///< LAMMPS EPS_HOC
inline constexpr double kLammpsFourPi = 12.5663706; ///< compute_gf_ik numerator

/// @brief `compute_rho_coeff`: rho_coeff[l][k], l < order,
///        k ∈ [(1−order)/2, order/2], stored at l·order + k − (1−order)/2.
inline std::vector<double> rho_coeff(int order) {
    const int width = 2 * order + 1;
    std::vector<double> a(static_cast<std::size_t>(order * width), 0.0);
    auto A = [&](int l, int k) -> double& {
        return a[static_cast<std::size_t>(l * width + k + order)];
    };
    A(0, 0) = 1.0;
    for (int j = 1; j < order; j++) {
        for (int k = -j; k <= j; k += 2) {
            double s = 0.0;
            for (int l = 0; l < j; l++) {
                A(l + 1, k) = (A(l, k + 1) - A(l, k - 1)) / (l + 1);
                s += std::pow(0.5, (double)l + 1)
                     * (A(l, k - 1) + std::pow(-1.0, (double)l) * A(l, k + 1))
                     / (l + 1);
            }
            A(0, k) = s;
        }
    }
    std::vector<double> coeff(static_cast<std::size_t>(order * order), 0.0);
    int m = 0;
    for (int k = -(order - 1); k < order; k += 2) {
        for (int l = 0; l < order; l++)
            coeff[static_cast<std::size_t>(l * order + m)] = A(l, k);
        m++;
    }
    return coeff;
}

/// @brief `compute_rho1d` for one axis: out[k − nlower] = rho1d[k].
inline void rho1d(const std::vector<double>& coeff, int order, double dx,
                  double* out) {
    for (int m = 0; m < order; m++) {
        double r = 0.0;
        for (int l = order - 1; l >= 0; l--)
            r = coeff[static_cast<std::size_t>(l * order + m)] + r * dx;
        out[m] = r;
    }
}

/// @brief LAMMPS `nlower` (set_grid_local).
inline int nlower(int order) { return -(order - 1) / 2; }

/// @brief `particle_map` for one coordinate u = s·K: the stencil centre nx
///        and the `make_rho` offset dx.
inline int particle_map(double u, int order, double& dx) {
    const double shift = (order % 2) ? kOffset + 0.5 : kOffset;
    const double shiftone = (order % 2) ? 0.0 : 0.5;
    const int nx = static_cast<int>(u + shift) - kOffset;
    dx = nx + shiftone - u;
    return nx;
}

/// @brief `compute_gf_denom` (bigint factorial: orders ≤ 10).
inline std::vector<double> gf_b(int order) {
    std::vector<double> b(static_cast<std::size_t>(order), 0.0);
    int l = 0;
    b[0] = 1.0;
    for (int m = 1; m < order; m++) {
        for (l = m; l > 0; l--)
            b[l] = 4.0
                   * (b[l] * (l - m) * (l - m - 0.5)
                      - b[l - 1] * (l - m - 1) * (l - m - 1));
        b[0] = 4.0 * (b[0] * (l - m) * (l - m - 0.5));
    }
    std::int64_t ifact = 1;
    for (int k = 1; k < 2 * order; k++)
        ifact *= k;
    const double gaminv = 1.0 / ifact;
    for (l = 0; l < order; l++)
        b[l] *= gaminv;
    return b;
}

/// @brief `gf_denom` (pppm.h).
inline double gf_denom(const std::vector<double>& b, double x, double y,
                       double z) {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int l = static_cast<int>(b.size()) - 1; l >= 0; l--) {
        sx = b[l] + sx * x;
        sy = b[l] + sy * y;
        sz = b[l] + sz * z;
    }
    const double s = sx * sy * sz;
    return s * s;
}

inline double powsinxx(double x, int n) {
    if (x == 0.0)
        return 1.0;
    const double yy = std::sin(x) / x;
    double v = 1.0;
    for (int i = 0; i < n; ++i)
        v *= yy;
    return v;
}

/// @brief Row-major 3×3 product helpers: H^T v and H^{-T} v (LAMMPS
///        `lamda2xT` / `x2lamdaT` for an upper-triangular H).
inline void mat_t_vec(const double* m, const double* v, double* out) {
    for (int r = 0; r < 3; ++r)
        out[r] =
            m[0 * 3 + r] * v[0] + m[1 * 3 + r] * v[1] + m[2 * 3 + r] * v[2];
}

inline void inverse3(const double* m, double* inv) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7])
                       - m[1] * (m[3] * m[8] - m[5] * m[6])
                       + m[2] * (m[3] * m[7] - m[4] * m[6]);
    inv[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) / det;
}

/// @brief One Green's function value: `compute_gf_ik` for a diagonal cell,
///        `compute_gf_ik_triclinic` otherwise, at signed mode (kper, lper,
///        mper). @p cell is row-major with lattice vectors in columns.
inline double green_ik(int kper, int lper, int mper, const double* cell,
                       const int K[3], double g_ewald, int order,
                       double four_pi) {
    const double pi = ATV::units::pi;
    const std::vector<double> b = gf_b(order);
    const int twoorder = 2 * order;
    const bool ortho = cell[1] == 0.0 && cell[2] == 0.0 && cell[3] == 0.0
                       && cell[5] == 0.0 && cell[6] == 0.0 && cell[7] == 0.0;
    const double snx = std::pow(std::sin(pi * kper / K[0]), 2);
    const double sny = std::pow(std::sin(pi * lper / K[1]), 2);
    const double snz = std::pow(std::sin(pi * mper / K[2]), 2);
    if (ortho) {
        const double xprd = cell[0], yprd = cell[4], zprd = cell[8];
        const double unitkx = 2 * pi / xprd, unitky = 2 * pi / yprd,
                     unitkz = 2 * pi / zprd;
        const int nbx = static_cast<int>((g_ewald * xprd / (pi * K[0]))
                                         * std::pow(-std::log(kEpsHoc), 0.25));
        const int nby = static_cast<int>((g_ewald * yprd / (pi * K[1]))
                                         * std::pow(-std::log(kEpsHoc), 0.25));
        const int nbz = static_cast<int>((g_ewald * zprd / (pi * K[2]))
                                         * std::pow(-std::log(kEpsHoc), 0.25));
        const double sqk = std::pow(unitkx * kper, 2)
                           + std::pow(unitky * lper, 2)
                           + std::pow(unitkz * mper, 2);
        if (sqk == 0.0)
            return 0.0;
        const double numerator = four_pi / sqk;
        const double denominator = gf_denom(b, snx, sny, snz);
        double sum1 = 0.0;
        for (int nx = -nbx; nx <= nbx; nx++) {
            const double qx = unitkx * (kper + K[0] * nx);
            const double sx = std::exp(-0.25 * std::pow(qx / g_ewald, 2));
            const double wx = powsinxx(0.5 * qx * xprd / K[0], twoorder);
            for (int ny = -nby; ny <= nby; ny++) {
                const double qy = unitky * (lper + K[1] * ny);
                const double sy = std::exp(-0.25 * std::pow(qy / g_ewald, 2));
                const double wy = powsinxx(0.5 * qy * yprd / K[1], twoorder);
                for (int nz = -nbz; nz <= nbz; nz++) {
                    const double qz = unitkz * (mper + K[2] * nz);
                    const double sz =
                        std::exp(-0.25 * std::pow(qz / g_ewald, 2));
                    const double wz =
                        powsinxx(0.5 * qz * zprd / K[2], twoorder);
                    const double dot1 = unitkx * kper * qx + unitky * lper * qy
                                        + unitkz * mper * qz;
                    const double dot2 = qx * qx + qy * qy + qz * qz;
                    sum1 += (dot1 / dot2) * sx * sy * sz * wx * wy * wz;
                }
            }
        }
        return numerator * sum1 / denominator;
    }
    double inv[9];
    inverse3(cell, inv);
    double tmp[3] = {
        (g_ewald / (pi * K[0])) * std::pow(-std::log(kEpsHoc), 0.25),
        (g_ewald / (pi * K[1])) * std::pow(-std::log(kEpsHoc), 0.25),
        (g_ewald / (pi * K[2])) * std::pow(-std::log(kEpsHoc), 0.25)};
    // KSpace::lamda2xT on the upper-triangular H: the tilts enter as |tilt|.
    const double bound[3] = {
        cell[0] * tmp[0], std::fabs(cell[1]) * tmp[0] + cell[4] * tmp[1],
        std::fabs(cell[2]) * tmp[0] + std::fabs(cell[5]) * tmp[1]
            + cell[8] * tmp[2]};
    const int nbx = static_cast<int>(bound[0]);
    const int nby = static_cast<int>(bound[1]);
    const int nbz = static_cast<int>(bound[2]);
    const double unit_in[3] = {2 * pi * kper, 2 * pi * lper, 2 * pi * mper};
    double unitk[3];
    mat_t_vec(inv, unit_in, unitk);
    const double sqk =
        unitk[0] * unitk[0] + unitk[1] * unitk[1] + unitk[2] * unitk[2];
    if (sqk == 0.0)
        return 0.0;
    const double numerator = four_pi / sqk;
    const double denominator = gf_denom(b, snx, sny, snz);
    double sum1 = 0.0;
    for (int nx = -nbx; nx <= nbx; nx++) {
        const double wx = powsinxx(pi * kper / K[0] + pi * nx, twoorder);
        for (int ny = -nby; ny <= nby; ny++) {
            const double wy = powsinxx(pi * lper / K[1] + pi * ny, twoorder);
            for (int nz = -nbz; nz <= nbz; nz++) {
                const double wz =
                    powsinxx(pi * mper / K[2] + pi * nz, twoorder);
                const double b_in[3] = {2 * pi * K[0] * nx, 2 * pi * K[1] * ny,
                                        2 * pi * K[2] * nz};
                double bb[3];
                mat_t_vec(inv, b_in, bb);
                const double qx = unitk[0] + bb[0];
                const double qy = unitk[1] + bb[1];
                const double qz = unitk[2] + bb[2];
                const double sx = std::exp(-0.25 * std::pow(qx / g_ewald, 2));
                const double sy = std::exp(-0.25 * std::pow(qy / g_ewald, 2));
                const double sz = std::exp(-0.25 * std::pow(qz / g_ewald, 2));
                const double dot1 =
                    unitk[0] * qx + unitk[1] * qy + unitk[2] * qz;
                const double dot2 = qx * qx + qy * qy + qz * qz;
                sum1 += (dot1 / dot2) * sx * sy * sz * wx * wy * wz;
            }
        }
    }
    return numerator * sum1 / denominator;
}

/// @brief LAMMPS mode index `i - K*(2*i/K)`.
inline int signed_mode(int m, int K) { return m - K * ((2 * m) / K); }

/// @brief Reciprocal energy [eV], ik forces [eV/Å] and virial [eV] of
///        point charges.
struct Result {
    double energy = 0.0;
    std::vector<double> force; ///< 3·n, atom-major.
    /// `PPPM::compute` virial: Σ_k (k_e/2V) G|ρ̂|² vg_k, vg from `setup` /
    /// `setup_triclinic` (vterm = −2(1/k² + 1/(4α²))); xx, yy, zz, xy, xz, yz.
    double virial[6] = {};
};

/// @brief LAMMPS ik PPPM on the full periodic mesh.
///
/// @param cell  Row-major cell [Å], lattice vectors in columns.
/// @param frac  Fractional coordinates, 3·n.
/// @param q     Charges [e], n.
/// @param K     Mesh size per axis.
/// @param stencil_shift Grid units subtracted from every u = s·K before the
///        map: 0 is LAMMPS; order/2 reproduces the one-sided smooth-PME
///        stencil, the placement the engines used before the port.
inline Result solve(const double* cell, const std::vector<double>& frac,
                    const std::vector<double>& q, const int K[3],
                    double g_ewald, int order, double stencil_shift = 0.0) {
    const double pi = ATV::units::pi;
    const int n = static_cast<int>(q.size());
    const int N = K[0] * K[1] * K[2];
    const std::vector<double> coeff = rho_coeff(order);
    const int lo = nlower(order);
    auto idx = [&](int a, int b, int c) {
        auto w = [](int g, int k) { return ((g % k) + k) % k; };
        return (w(a, K[0]) * K[1] + w(b, K[1])) * K[2] + w(c, K[2]);
    };
    // make_rho
    std::vector<int> start(static_cast<std::size_t>(3 * n));
    std::vector<double> w(static_cast<std::size_t>(3 * n * order));
    std::vector<double> rho(static_cast<std::size_t>(N), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            double u = frac[3 * i + d] * K[d] - stencil_shift;
            u -= std::floor(u / K[d]) * K[d];
            double dx = 0.0;
            const int nx = particle_map(u, order, dx);
            rho1d(coeff, order, dx,
                  &w[static_cast<std::size_t>((3 * i + d) * order)]);
            start[3 * i + d] = nx + lo;
        }
        for (int a = 0; a < order; ++a)
            for (int b = 0; b < order; ++b)
                for (int c = 0; c < order; ++c)
                    rho[idx(start[3 * i] + a, start[3 * i + 1] + b,
                            start[3 * i + 2] + c)] +=
                        q[i] * w[(3 * i + 0) * order + a]
                        * w[(3 * i + 1) * order + b]
                        * w[(3 * i + 2) * order + c];
    }
    // Separable DFT along each axis; sign −1 forward, +1 backward.
    auto dft = [&](std::vector<std::complex<double>>& v, double sign) {
        for (int axis = 0; axis < 3; ++axis) {
            const int L = K[axis];
            std::vector<std::complex<double>> line(L), out(L);
            for (int a = 0; a < K[0]; ++a)
                for (int b = 0; b < K[1]; ++b)
                    for (int c = 0; c < K[2]; ++c) {
                        const int pos[3] = {a, b, c};
                        if (pos[axis] != 0)
                            continue;
                        for (int t = 0; t < L; ++t) {
                            int p[3] = {a, b, c};
                            p[axis] = t;
                            line[t] = v[(p[0] * K[1] + p[1]) * K[2] + p[2]];
                        }
                        for (int m = 0; m < L; ++m) {
                            std::complex<double> s = 0.0;
                            for (int t = 0; t < L; ++t)
                                s += line[t]
                                     * std::polar(1.0, sign * 2 * pi * double(m)
                                                           * t / L);
                            out[m] = s;
                        }
                        for (int t = 0; t < L; ++t) {
                            int p[3] = {a, b, c};
                            p[axis] = t;
                            v[(p[0] * K[1] + p[1]) * K[2] + p[2]] = out[t];
                        }
                    }
        }
    };
    std::vector<std::complex<double>> rhat(rho.begin(), rho.end());
    dft(rhat, -1.0);
    double inv[9];
    inverse3(cell, inv);
    const double volume =
        std::abs(cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
                 - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
                 + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
    Result r;
    r.force.assign(static_cast<std::size_t>(3 * n), 0.0);
    std::vector<std::complex<double>> field[3];
    for (auto& f : field)
        f.assign(static_cast<std::size_t>(N), 0.0);
    double esum = 0.0;
    double vsum[6] = {};
    for (int m0 = 0; m0 < K[0]; ++m0)
        for (int m1 = 0; m1 < K[1]; ++m1)
            for (int m2 = 0; m2 < K[2]; ++m2) {
                const int h[3] = {signed_mode(m0, K[0]), signed_mode(m1, K[1]),
                                  signed_mode(m2, K[2])};
                const double G = green_ik(h[0], h[1], h[2], cell, K, g_ewald,
                                          order, 4.0 * pi);
                const int u = (m0 * K[1] + m1) * K[2] + m2;
                esum += G * std::norm(rhat[u]);
                const double hd[3] = {double(h[0]), double(h[1]), double(h[2])};
                double k[3];
                mat_t_vec(inv, hd, k);
                for (double& c : k)
                    c *= 2 * pi;
                const double sqk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                if (sqk != 0.0) {
                    const double vterm =
                        -2.0 * (1.0 / sqk + 0.25 / (g_ewald * g_ewald));
                    const double vg[6] = {
                        1.0 + vterm * k[0] * k[0], 1.0 + vterm * k[1] * k[1],
                        1.0 + vterm * k[2] * k[2], vterm * k[0] * k[1],
                        vterm * k[0] * k[2],       vterm * k[1] * k[2]};
                    for (int c = 0; c < 6; ++c)
                        vsum[c] += G * std::norm(rhat[u]) * vg[c];
                }
                for (double& c : k)
                    c /= 2 * pi;
                for (int a = 0; a < 3; ++a)
                    field[a][u] = std::complex<double>(0.0, 2 * pi * k[a])
                                  * (G / volume) * rhat[u];
            }
    r.energy = ATV::units::UnitSystem::lj().qqrd2e() / (2.0 * volume) * esum;
    for (int c = 0; c < 6; ++c)
        r.virial[c] =
            ATV::units::UnitSystem::lj().qqrd2e() / (2.0 * volume) * vsum[c];
    for (auto& f : field)
        dft(f, +1.0);
    // fieldforce_ik: F = −k_e q Σ W · Re(field).
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < order; ++a)
            for (int b = 0; b < order; ++b)
                for (int c = 0; c < order; ++c) {
                    const int g = idx(start[3 * i] + a, start[3 * i + 1] + b,
                                      start[3 * i + 2] + c);
                    const double wt = w[(3 * i + 0) * order + a]
                                      * w[(3 * i + 1) * order + b]
                                      * w[(3 * i + 2) * order + c];
                    for (int d = 0; d < 3; ++d)
                        r.force[3 * i + d] -=
                            ATV::units::UnitSystem::lj().qqrd2e() * q[i] * wt
                            * field[d][g].real();
                }
    return r;
}

} // namespace ATV::test::lammps_pppm
