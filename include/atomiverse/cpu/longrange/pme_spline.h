/// @file cpu/longrange/pme_spline.h
/// @brief Backend-agnostic PME B-spline / modulus / FFT-size math and charge
///        spreading (no Frame / Component / MPI / FFT engine).
///
/// Smooth PME of Essmann et al., *J. Chem. Phys.* **103**, 8577 (1995).
/// DOI: 10.1063/1.470117 — cardinal B-spline weights (`fill_bspline` appendix),
/// Euler spline moduli, FFT-friendly grid sizing, and Q-grid charge spreading.
/// Units: positions [Å], charges [e], fractional coords dimensionless.
#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace longrange {

/// @brief Cardinal B-spline weights and derivatives on the n-point stencil.
///
/// For fractional offset @p frac ∈ [0, 1), fills @p theta[k] = M_n(frac+k) and
/// @p dtheta[k] = dM_n/d(frac) at frac+k, k = 0..n−1. Canonical Essmann/Amber
/// `fill_bspline` (J. Chem. Phys. 103, 8577 (1995), appendix).
///
/// @param frac   Fractional offset in [0, 1).
/// @param n      B-spline order (≥ 2; PME uses ≥ 3).
/// @param[out] theta  Length @p n.
/// @param[out] dtheta Length @p n.
inline void bspline(double frac, int n, double* theta, double* dtheta) {
    theta[n - 1] = 0.0;
    theta[1] = frac;
    theta[0] = 1.0 - frac;
    // Standard B-spline up-recursion to order n-1.
    for (int k = 3; k < n; ++k) {
        const double div = 1.0 / (k - 1);
        theta[k - 1] = div * frac * theta[k - 2];
        for (int j = 1; j < k - 1; ++j)
            theta[k - 1 - j] = div
                               * ((frac + j) * theta[k - 2 - j]
                                  + (k - j - frac) * theta[k - 1 - j]);
        theta[0] = div * (1.0 - frac) * theta[0];
    }
    // Derivative taken from the order n-1 values.
    dtheta[0] = -theta[0];
    for (int j = 1; j < n; ++j)
        dtheta[j] = theta[j - 1] - theta[j];
    // Final up-recursion: order n-1 → n.
    const double div = 1.0 / (n - 1);
    theta[n - 1] = div * frac * theta[n - 2];
    for (int j = 1; j < n - 1; ++j)
        theta[n - 1 - j] = div
                           * ((frac + j) * theta[n - 2 - j]
                              + (n - j - frac) * theta[n - 1 - j]);
    theta[0] = div * (1.0 - frac) * theta[0];
}

/// @brief Euler spline modulus |b_d(m)|² for grid size @p K and order @p n.
///
/// Exact zeros at even-order K/2 are repaired by neighbour averaging.
///
/// @param K Grid dimension (points along one axis).
/// @param n B-spline order.
/// @return Length-@p K moduli (dimensionless).
inline std::vector<double> spline_moduli(int K, int n) {
    // Spline values at integers: data[k] = M_n(k), k=0..n-1 (frac=0).
    std::vector<double> data(static_cast<std::size_t>(n), 0.0),
        ddummy(static_cast<std::size_t>(n), 0.0);
    bspline(0.0, n, data.data(), ddummy.data());
    // fill_bspline(0) returns [M_n(1), M_n(2), …, M_n(n-1), 0], so the spline
    // value M_n(k+1) used in the Euler modulus is data[k].
    std::vector<double> mod(static_cast<std::size_t>(K), 0.0);
    for (int i = 0; i < K; ++i) {
        double sr = 0, si = 0;
        for (int k = 0; k < n - 1; ++k) {
            const double arg = 2.0 * ATV::units::pi * i * k / K;
            sr += data[static_cast<std::size_t>(k)] * std::cos(arg);
            si += data[static_cast<std::size_t>(k)] * std::sin(arg);
        }
        const double denom = sr * sr + si * si;
        mod[static_cast<std::size_t>(i)] = denom > 1e-12 ? 1.0 / denom : 0.0;
    }
    // Fix exact zeros (even orders at i=K/2) by neighbour averaging.
    for (int i = 0; i < K; ++i) {
        if (mod[static_cast<std::size_t>(i)] == 0.0) {
            const double lo = mod[static_cast<std::size_t>((i - 1 + K) % K)];
            const double hi = mod[static_cast<std::size_t>((i + 1) % K)];
            mod[static_cast<std::size_t>(i)] = 0.5 * (lo + hi);
        }
    }
    return mod;
}

/// @brief Smallest m ≥ max(n, 4) that factorises into {2, 3, 5, 7}.
/// @param n Requested grid size (points).
/// @return FFT-friendly size ≥ 4.
inline int good_fft_size(int n) {
    auto ok = [](int m) {
        for (int p : {2, 3, 5, 7})
            while (m % p == 0)
                m /= p;
        return m == 1;
    };
    if (n < 4)
        n = 4;
    while (!ok(n))
        ++n;
    return n;
}

/// @brief Fill per-atom B-spline tables (base index + θ / dθ) from positions.
///
/// @param x,y,z     Atom coordinates [Å], length @p n_atoms.
/// @param n_atoms   Atom count.
/// @param inv_cell  Inverse cell matrix [1/Å] (Mat3 indexing).
/// @param K         Grid dimensions {K0, K1, K2}.
/// @param order     B-spline order n.
/// @param[out] base Length 3·n_atoms; base[3·i+d] = floor(u_d) in [0, K_d).
/// @param[out] theta  Length 3·order·n_atoms; θ for atom i, dim d at
///                    &theta[(3·i+d)·order].
/// @param[out] dtheta Same layout as @p theta (dM/d frac).
inline void pme_fill_atom_splines(const StorageReal* x, const StorageReal* y,
                                  const StorageReal* z, int n_atoms,
                                  const Mat3& inv_cell, const int K[3],
                                  int order, int* base, double* theta,
                                  double* dtheta) {
    const int n = order;
    for (int i = 0; i < n_atoms; ++i) {
        double s[3];
        for (int d = 0; d < 3; ++d)
            s[d] = inv_cell(0, d) * x[i] + inv_cell(1, d) * y[i]
                   + inv_cell(2, d) * z[i];
        for (int d = 0; d < 3; ++d) {
            double u = s[d] * K[d];
            u -= std::floor(u / K[d]) * K[d]; // wrap into [0,K)
            const int u0 = static_cast<int>(std::floor(u));
            const double frac = u - u0;
            bspline(frac, n, &theta[(3 * i + d) * n], &dtheta[(3 * i + d) * n]);
            base[3 * i + d] = u0;
        }
    }
}

/// @brief Wrap grid index into [0, K) without double-modulo.
/// @param g Candidate grid index, at most one period outside [0, K).
/// @param K Grid extent along this axis.
/// @return The index folded into [0, K).
inline int pme_wrap(int g, int K) {
    // base ∈ [0,K), offset ∈ [-(n-1), n-1] → at most one wrap either side
    // for typical order ≤ 8; fall back to while for safety.
    if (g >= K)
        g -= K;
    if (g < 0)
        g += K;
    if (g >= K)
        g -= K;
    if (g < 0)
        g += K;
    return g;
}

/// @brief Spread charges onto a complex PME grid (Q).
///
/// Grid layout: idx = k2 + K2·(k1 + K1·k0). Caller zeroes @p grid first.
/// @p base / @p theta from #pme_fill_atom_splines.
///
/// @param q       Charges [e], length @p n_atoms.
/// @param n_atoms Atom count.
/// @param order   B-spline order.
/// @param K       Grid dimensions {K0, K1, K2}.
/// @param base    Per-atom base indices (length 3·n_atoms).
/// @param theta   Per-atom B-spline weights (length 3·order·n_atoms).
/// @param[in,out] grid Complex charge grid, length K0·K1·K2.
inline void pme_spread_charges(const StorageReal* q, int n_atoms, int order,
                               const int K[3], const int* base,
                               const double* theta,
                               std::complex<double>* grid) {
    const int K0 = K[0], K1 = K[1], K2 = K[2];
    const int n = order;
    const int nm1 = n - 1;
    for (int i = 0; i < n_atoms; ++i) {
        const double qi = q[i];
        if (qi == 0.0)
            continue;
        const double* thx = &theta[(3 * i + 0) * n];
        const double* thy = &theta[(3 * i + 1) * n];
        const double* thz = &theta[(3 * i + 2) * n];
        const int b0 = base[3 * i + 0] - nm1;
        const int b1 = base[3 * i + 1] - nm1;
        const int b2 = base[3 * i + 2] - nm1;
        // theta[k] = M_n(frac + n-1-k) pairs with grid point u0 - (n-1) + k.
        for (int a = 0; a < n; ++a) {
            const int g0 = pme_wrap(b0 + a, K0);
            const double wa = qi * thx[a];
            for (int b = 0; b < n; ++b) {
                const int g1 = pme_wrap(b1 + b, K1);
                const double wab = wa * thy[b];
                const int plane = K2 * (g1 + K1 * g0);
                for (int c = 0; c < n; ++c) {
                    const int g2 = pme_wrap(b2 + c, K2);
                    // Charge is real — write only .real to cut complex ops.
                    grid[plane + g2].real(grid[plane + g2].real()
                                          + wab * thz[c]);
                }
            }
        }
    }
}

/// @brief Spread onto a real grid (for r2c paths / faster charge assignment).
inline void pme_spread_charges_real(const StorageReal* q, int n_atoms,
                                    int order, const int K[3], const int* base,
                                    const double* theta, double* grid) {
    const int K0 = K[0], K1 = K[1], K2 = K[2];
    const int n = order;
    const int nm1 = n - 1;
    for (int i = 0; i < n_atoms; ++i) {
        const double qi = q[i];
        if (qi == 0.0)
            continue;
        const double* thx = &theta[(3 * i + 0) * n];
        const double* thy = &theta[(3 * i + 1) * n];
        const double* thz = &theta[(3 * i + 2) * n];
        const int b0 = base[3 * i + 0] - nm1;
        const int b1 = base[3 * i + 1] - nm1;
        const int b2 = base[3 * i + 2] - nm1;
        for (int a = 0; a < n; ++a) {
            const int g0 = pme_wrap(b0 + a, K0);
            const double wa = qi * thx[a];
            for (int b = 0; b < n; ++b) {
                const int g1 = pme_wrap(b1 + b, K1);
                const double wab = wa * thy[b];
                const int plane = K2 * (g1 + K1 * g0);
                for (int c = 0; c < n; ++c) {
                    const int g2 = pme_wrap(b2 + c, K2);
                    grid[plane + g2] += wab * thz[c];
                }
            }
        }
    }
}

} // namespace longrange
} // namespace ATV
