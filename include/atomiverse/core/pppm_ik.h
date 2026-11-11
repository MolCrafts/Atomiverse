/// @file core/pppm_ik.h
/// @brief Hockney–Eastwood PPPM scalar laws shared by both backends: the
///        centred charge-assignment stencil, the ik influence function and
///        the mesh-size rule.
///
/// Ported from LAMMPS `src/KSPACE/pppm.cpp` and `src/KSPACE/pppm.h`,
/// checkout e8c4c278a8: `set_grid_local` (`shift`, `shiftone`, `nlower`,
/// `nupper`), `particle_map` / `make_rho` (grid point and offset of a
/// charge), `compute_gf_denom` / `gf_denom`, `compute_gf_ik` /
/// `compute_gf_ik_triclinic` and `factorable`.
/// LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.
///
/// **Stencil.** A charge at grid coordinate u = s·K (s fractional) is spread
/// on the `order` points nx + nlower … nx + nupper, nlower = −(order−1)/2,
/// nupper = order/2. For even orders nx is the grid point below the charge,
/// for odd orders the nearest one. The weight of point nx + j is the centred
/// cardinal B-spline W_order(nx + j − u), i.e. LAMMPS `rho1d[j]`; it equals
/// `longrange::bspline(frac)[j − nlower]` with frac = 0.5 − dx, dx the LAMMPS
/// `make_rho` offset. This is not the one-sided smooth-PME stencil
/// (`cpu/longrange/pme_spline.h`), which places an odd-order spline half a
/// cell away.
///
/// **Influence.** The scalar G(k) is the ik Green's function: assignment
/// window [sinc]^(2·order), analytic denominator, and the alias sum
/// truncated where the Gaussian falls below 1e-7 (`EPS_HOC`), with the
/// LAMMPS alias bound (`lamda2xT` of the per-axis bound: the restricted
/// triclinic H^T with |tilts|, which is the orthogonal bound for a diagonal
/// cell). 4π is exact here; LAMMPS writes 12.5663706 (1.14e-9 low).
///
/// k = 2π H^{-T} h, with lattice vectors stored as columns of H and
/// `inv` the row-major inverse (s = inv · r). Forces that consume this
/// G are ik forces: Ê(k) = i k G(k) ρ̂(k), interpolated with the same
/// B-spline θ that spread the charge.
///
/// **Half spectrum.** LAMMPS transforms the full complex mesh and keeps the
/// real part of each inverse ik transform (`poisson_ik`); both engines use an
/// r2c/c2r pair. `pppm_half_mode` folds each stored mode m together with its
/// conjugate partner −m so the two are the same sum: at a Nyquist index
/// (h = −K/2 of an even mesh, which is its own negative) the partner's
/// wavevector is not −k, and the ik derivative along that index cancels in
/// LAMMPS's real part. Nothing is zeroed by hand; the fold gives it.
#pragma once

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "atomiverse/core/units.h"

#if defined(__CUDACC__)
#define ATV_PPPM_HD __host__ __device__
#else
#define ATV_PPPM_HD
#endif

namespace ATV::longrange {

/// LAMMPS `OFFSET`: keeps the truncating cast of a slightly negative grid
/// coordinate on the floor (`int(-0.75) = 0` otherwise).
inline constexpr int kPppmOffset = 16384;

/// @brief First stencil offset, LAMMPS `nlower = -(order-1)/2`.
/// @param order Charge-assignment order (count), ≥ 2.
/// @return nlower (grid points, ≤ 0).
ATV_PPPM_HD inline int pppm_stencil_lower(int order) {
    return -(order - 1) / 2;
}

/// @brief Last stencil offset, LAMMPS `nupper = order/2`.
/// @param order Charge-assignment order (count), ≥ 2.
/// @return nupper (grid points, ≥ 1).
ATV_PPPM_HD inline int pppm_stencil_upper(int order) { return order / 2; }

/// @brief Fraction added before the floor, LAMMPS `shift - OFFSET`.
/// @param order Charge-assignment order (count).
/// @return 0.5 for odd orders (nearest grid point), 0 for even orders.
ATV_PPPM_HD inline double pppm_stencil_shift(int order) {
    return (order % 2) ? 0.5 : 0.0;
}

/// @brief Grid point of a charge and its B-spline offset (LAMMPS
///        `particle_map` + `make_rho`).
///
/// nx = int(u + shift + OFFSET) − OFFSET; the stencil is nx + nlower …
/// nx + nupper. The returned `frac` = u + shift − nx ∈ [0, 1) is
/// 0.5 − dx for the LAMMPS offset dx = nx + shiftone − u, and
/// `bspline(frac, order, θ, …)` gives θ[j − nlower] = rho1d[j].
///
/// @param u     Grid coordinate s·K (grid units), > −kPppmOffset.
/// @param order Charge-assignment order (count).
/// @param[out] frac B-spline argument (dimensionless) in [0, 1).
/// @return nx (grid index; may equal K at the upper edge).
ATV_PPPM_HD inline int pppm_particle_map(double u, int order, double& frac) {
    const double shifted = u + pppm_stencil_shift(order);
    const int nx = static_cast<int>(shifted + kPppmOffset) - kPppmOffset;
    frac = shifted - static_cast<double>(nx);
    return nx;
}

/// @brief Smallest mesh size ≥ max(n, 4) with prime factors 2, 3 and 5 only
///        (LAMMPS `PPPM::factorable`; 7 is a smooth-PME size, not a PPPM one).
/// @param n Requested points along one axis (count).
/// @return The PPPM mesh size (count).
/// @throws std::overflow_error when no such size fits an int.
inline int pppm_fft_size(int n) {
    auto ok = [](int m) {
        for (int p : {2, 3, 5})
            while (m % p == 0)
                m /= p;
        return m == 1;
    };
    if (n < 4)
        n = 4;
    while (!ok(n)) {
        if (n == std::numeric_limits<int>::max())
            throw std::overflow_error(
                "KSpacePPPM: FFT dimension exceeds int indexing");
        ++n;
    }
    return n;
}

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

/// @brief PPPM influence denominator (LAMMPS `gf_denom`).
/// @param gf_b  Coefficients from pppm_gf_b_fill, `order` slots.
/// @param order Charge-assignment order (count).
/// @param x,y,z sin²(π h_d / K_d) per axis (dimensionless).
/// @return (Σ_l b_l x^l · Σ_l b_l y^l · Σ_l b_l z^l)² (dimensionless).
ATV_PPPM_HD inline double pppm_gf_denom_scalar(const double* gf_b, int order,
                                               double x, double y, double z) {
    // LAMMPS `PPPM::gf_denom` (pppm.h): Horner in sin², then (Sx Sy Sz)².
    // The polynomial is Σ W²; the ik influence divides by |Σ W²|².
    double sx = 0, sy = 0, sz = 0;
    for (int l = order - 1; l >= 0; --l) {
        sx = gf_b[l] + sx * x;
        sy = gf_b[l] + sy * y;
        sz = gf_b[l] + sz * z;
    }
    const double s = sx * sy * sz;
    return s * s;
}

/// @brief The LAMMPS restricted-triclinic box of a cell (`Domain::h`).
///
/// h = (lx, ly, lz, yz, xz, xy) with a = (lx, 0, 0), b = (xy, ly, 0),
/// c = (xz, yz, lz) — the Gram–Schmidt (QR) factor of the column lattice
/// vectors, so it does not depend on how the cell is oriented in space. An
/// upper-triangular H with a positive diagonal is read directly.
/// @param cell Row-major cell H [Å], lattice vectors in columns.
/// @param[out] h (lx, ly, lz, yz, xz, xy) [Å].
ATV_PPPM_HD inline void pppm_lammps_box(const double* cell, double h[6]) {
    if (cell[3] == 0.0 && cell[6] == 0.0 && cell[7] == 0.0 && cell[0] > 0.0
        && cell[4] > 0.0 && cell[8] > 0.0) {
        h[0] = cell[0];
        h[1] = cell[4];
        h[2] = cell[8];
        h[3] = cell[5];
        h[4] = cell[2];
        h[5] = cell[1];
        return;
    }
    const double a[3] = {cell[0], cell[3], cell[6]};
    const double b[3] = {cell[1], cell[4], cell[7]};
    const double c[3] = {cell[2], cell[5], cell[8]};
    const double aa = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    const double ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const double ac = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
    const double bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    const double bc = b[0] * c[0] + b[1] * c[1] + b[2] * c[2];
    const double cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    const double lx = ::sqrt(aa);
    const double xy = lx > 0.0 ? ab / lx : 0.0;
    const double ly2 = bb - xy * xy;
    const double ly = ly2 > 0.0 ? ::sqrt(ly2) : 0.0;
    const double xz = lx > 0.0 ? ac / lx : 0.0;
    const double yz = ly > 0.0 ? (bc - xy * xz) / ly : 0.0;
    const double lz2 = cc - xz * xz - yz * yz;
    h[0] = lx;
    h[1] = ly;
    h[2] = lz2 > 0.0 ? ::sqrt(lz2) : 0.0;
    h[3] = yz;
    h[4] = xz;
    h[5] = xy;
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
/// @param h0,h1,h2 Signed mode indices (pppm_signed_mode).
/// @param cell Row-major cell H [Å], lattice vectors in columns; sets the
///        alias bound only (LAMMPS `lamda2xT`, the orthogonal bound for a
///        diagonal H).
/// @param K     Mesh size per axis (count).
/// @param alpha Ewald α [1/Å].
/// @param order Charge-assignment order (count).
/// @param gf_b  Denominator coefficients from pppm_gf_b_fill, `order` slots.
/// @param inv Row-major inverse cell [1/Å].
/// @return G(k) [Å²]; 0 for the zero mode.
ATV_PPPM_HD inline double pppm_green_ik(int h0, int h1, int h2,
                                        const double* cell, const int K[3],
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
    // LAMMPS compute_gf_ik_triclinic: tmp_d = α/(π K_d)·hoc, nb =
    // lamda2xT(tmp), whose tilts enter as |tilt|. For a diagonal H this is
    // compute_gf_ik's α L_d/(π K_d)·hoc.
    double tmp[3];
    for (int d = 0; d < 3; ++d)
        tmp[d] = (alpha / (pi * static_cast<double>(K[d]))) * hoc;
    double lbox[6];
    pppm_lammps_box(cell, lbox);
    const double bound[3] = {
        lbox[0] * tmp[0], ::fabs(lbox[5]) * tmp[0] + lbox[1] * tmp[1],
        ::fabs(lbox[4]) * tmp[0] + ::fabs(lbox[3]) * tmp[1] + lbox[2] * tmp[2]};
    int nb[3];
    for (int d = 0; d < 3; ++d)
        nb[d] = static_cast<int>(bound[d]);
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

/// @brief One stored r2c mode with its conjugate partner folded in.
///
/// LAMMPS sums the energy and the virial over every mode of the full mesh,
/// each with its own G and k, and keeps Re of each inverse ik transform. For
/// a stored mode m (m2 ∈ [0, K2/2]) with partner p = −m:
///  * `green` — the energy weight: (G(m) + G(p))/2 when p is not stored
///    (0 < m2 < K2/2, counted twice), G(m) on the m2 = 0 and m2 = K2/2
///    planes, whose partners are stored themselves;
///  * `ik[a]` — (k_a(m) G(m) − k_a(p) G(p))/2, so i·ik·ρ̂(m) is the Hermitian
///    part of LAMMPS's field i k G ρ̂ and a c2r transform of it is LAMMPS's
///    real part. Away from Nyquist indices p = −h and this is k G;
///  * `virial[c]` — G(δ_ab + vt m_a m_b), m = k/(2π),
///    vt = −2(1/|m|² + π²/α²) (LAMMPS `setup` vg), averaged over m and p
///    like `green`. Order xx, yy, zz, xy, xz, yz.
struct PppmHalfMode {
    double green = 0.0;    ///< Energy weight [Å²].
    double ik[3] = {};     ///< ik coefficient [Å].
    double virial[6] = {}; ///< Virial coefficient [Å²].
};

/// @brief LAMMPS-equivalent half-spectrum coefficients of mode (m0, m1, m2).
/// @param m0,m1,m2 Stored mode indices, m_d ∈ [0, K_d), m2 ≤ K2/2.
/// @param cell,K,alpha,order,gf_b,inv As for pppm_green_ik.
/// @param want_virial Fill `virial` (otherwise left zero).
ATV_PPPM_HD inline PppmHalfMode
pppm_half_mode(int m0, int m1, int m2, const double* cell, const int K[3],
               double alpha, int order, const double* gf_b, const double* inv,
               bool want_virial) {
    const int m[3] = {m0, m1, m2};
    int h[3], p[3];
    bool negated = true;
    for (int d = 0; d < 3; ++d) {
        h[d] = pppm_signed_mode(m[d], K[d]);
        p[d] = pppm_signed_mode((K[d] - m[d]) % K[d], K[d]);
        negated = negated && (p[d] == -h[d]);
    }
    const double G =
        pppm_green_ik(h[0], h[1], h[2], cell, K, alpha, order, gf_b, inv);
    // G is even under h → −h; only a Nyquist partner needs its own sum.
    const double Gp = negated ? G
                              : pppm_green_ik(p[0], p[1], p[2], cell, K, alpha,
                                              order, gf_b, inv);
    double k[3], kp[3];
    pppm_wavevector(h[0], h[1], h[2], inv, k);
    pppm_wavevector(p[0], p[1], p[2], inv, kp);
    const bool interior = m2 != 0 && !(K[2] % 2 == 0 && m2 == K[2] / 2);
    PppmHalfMode out;
    out.green = interior ? 0.5 * (G + Gp) : G;
    for (int a = 0; a < 3; ++a)
        out.ik[a] = 0.5 * (k[a] * G - kp[a] * Gp);
    if (want_virial) {
        const double pi = ATV::units::pi;
        const double pi2_alpha2 = pi * pi / (alpha * alpha);
        const double inv_twopi = 0.5 / pi;
        const double* kv[2] = {k, kp};
        const double gv[2] = {G, Gp};
        const double wv[2] = {interior ? 0.5 : 1.0, interior ? 0.5 : 0.0};
        for (int t = 0; t < 2; ++t) {
            const double g = gv[t] * wv[t];
            if (g == 0.0)
                continue;
            const double mx = kv[t][0] * inv_twopi;
            const double my = kv[t][1] * inv_twopi;
            const double mz = kv[t][2] * inv_twopi;
            const double msq = mx * mx + my * my + mz * mz;
            const double vt = -2.0 * (1.0 / msq + pi2_alpha2);
            out.virial[0] += g * (1.0 + vt * mx * mx);
            out.virial[1] += g * (1.0 + vt * my * my);
            out.virial[2] += g * (1.0 + vt * mz * mz);
            out.virial[3] += g * vt * mx * my;
            out.virial[4] += g * vt * mx * mz;
            out.virial[5] += g * vt * my * mz;
        }
    }
    return out;
}

/// @brief Fill the folded half-spectrum tables (pppm_half_mode) of a mesh.
/// Layout idx = m2 + n2·(m1 + K1·m0), n2 = K2/2 + 1; `ik` and `virial` are
/// component planes, ik[a·n + idx] and virial[c·n + idx], n = K0·K1·n2.
/// @param[out] green  n values [Å²].
/// @param[out] ik     3·n values [Å], or nullptr.
/// @param[out] virial 6·n values [Å²], or nullptr.
inline void pppm_fill_half_modes(const int K[3], const double* cell,
                                 const double* inv, double alpha, int order,
                                 double* green, double* ik, double* virial) {
    double gf_b[12] = {};
    pppm_gf_b_fill(order, gf_b);
    const int n2 = K[2] / 2 + 1;
    const long long n = static_cast<long long>(K[0]) * K[1] * n2;
    for (int m0 = 0; m0 < K[0]; ++m0)
        for (int m1 = 0; m1 < K[1]; ++m1)
            for (int m2 = 0; m2 < n2; ++m2) {
                const long long idx =
                    m2 + static_cast<long long>(n2) * (m1 + K[1] * m0);
                const PppmHalfMode hm =
                    pppm_half_mode(m0, m1, m2, cell, K, alpha, order, gf_b, inv,
                                   virial != nullptr);
                green[idx] = hm.green;
                if (ik)
                    for (int a = 0; a < 3; ++a)
                        ik[a * n + idx] = hm.ik[a];
                if (virial)
                    for (int c = 0; c < 6; ++c)
                        virial[c * n + idx] = hm.virial[c];
            }
}

} // namespace ATV::longrange

#undef ATV_PPPM_HD
