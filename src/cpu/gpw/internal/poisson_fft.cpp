/// @file src/cpu/gpw/internal/poisson_fft.cpp
/// @brief Implementation of the FFT periodic Poisson solver (see header).
///
/// Convention: forward FFTW r2c gives the unnormalized DFT
///   rho_hat(G) = Σ_r ρ(r) e^{-iG·r}.
/// We store V_hat(G) = 4π/|G|² · rho_hat(G) (V_hat(0)=0) and inverse-transform;
/// FFTW c2r returns N·V(r), so the result is divided by N = nx*ny*nz. The
/// physical |G|² uses G_d = 2π m_d / L_d, so no grid-spacing factor is needed.

#include "atomiverse/cpu/gpw/internal/poisson_fft.h"

#include <cmath>

#include "atomiverse/cpu/gpw/internal/fft_cache.h"

namespace ATV {
namespace cpu {
namespace gpw {

PoissonResult solve_poisson(const UniformGrid& grid,
                            const std::vector<double>& rho_r) {
    const int n0 = grid.n[0];
    const int n1 = grid.n[1];
    const int n2 = grid.n[2];
    const long ntot = static_cast<long>(n0) * n1 * n2;
    const int n2c = n2 / 2 + 1; // r2c last-dim (Hermitian) length

    const double two_pi = 2.0 * M_PI;
    const double gx_unit = two_pi / grid.L[0];
    const double gy_unit = two_pi / grid.L[1];
    const double gz_unit = two_pi / grid.L[2];
    const double four_pi = 4.0 * M_PI;

    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    double* rbuf = fft.r();
    fftw_complex* cbuf = fft.c();

    for (long i = 0; i < ntot; ++i)
        rbuf[i] = rho_r[static_cast<std::size_t>(i)];

    fft.execute_r2c(); // rbuf -> cbuf (= rho_hat)

    // Apply 4π/|G|² in reciprocal space; zero the G=0 component.
    for (int i0 = 0; i0 < n0; ++i0) {
        const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
        const double gx = m0 * gx_unit;
        for (int i1 = 0; i1 < n1; ++i1) {
            const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
            const double gy = m1 * gy_unit;
            for (int i2 = 0; i2 < n2c; ++i2) {
                const int m2 = i2; // r2c keeps only non-negative last-dim freqs
                const double gz = m2 * gz_unit;
                const long idx = (static_cast<long>(i0) * n1 + i1) * n2c + i2;
                const double g2 = gx * gx + gy * gy + gz * gz;
                if (g2 == 0.0) {
                    cbuf[idx][0] = 0.0;
                    cbuf[idx][1] = 0.0;
                } else {
                    const double f = four_pi / g2;
                    cbuf[idx][0] *= f;
                    cbuf[idx][1] *= f;
                }
            }
        }
    }

    fft.execute_c2r(); // cbuf -> rbuf (= N * V(r))

    PoissonResult out;
    out.v_r.resize(static_cast<std::size_t>(ntot));
    const double inv_n = 1.0 / static_cast<double>(ntot);
    for (long i = 0; i < ntot; ++i)
        out.v_r[static_cast<std::size_t>(i)] = rbuf[i] * inv_n;

    // Hartree energy 0.5 Σ_g w ρ_g V_g.
    const double w = grid.weight();
    double e = 0.0;
    for (long i = 0; i < ntot; ++i)
        e += rho_r[static_cast<std::size_t>(i)] * out.v_r[static_cast<std::size_t>(i)];
    out.e_hartree = 0.5 * w * e;

    return out;
}

PoissonResult solve_poisson_q(const UniformGrid& grid,
                              const std::vector<double>& rho_r,
                              const double q_frac[3]) {
    // q = 0 → identical to solve_poisson (fast path / bit-identical kernel).
    if (q_frac[0] == 0.0 && q_frac[1] == 0.0 && q_frac[2] == 0.0)
        return solve_poisson(grid, rho_r);

    const int n0 = grid.n[0];
    const int n1 = grid.n[1];
    const int n2 = grid.n[2];
    const long ntot = static_cast<long>(n0) * n1 * n2;
    const int n2c = n2 / 2 + 1;

    const double two_pi = 2.0 * M_PI;
    const double gx_unit = two_pi / grid.L[0];
    const double gy_unit = two_pi / grid.L[1];
    const double gz_unit = two_pi / grid.L[2];
    const double four_pi = 4.0 * M_PI;
    // Crystal momentum q = 2π κ / L (same convention as k-point Bloch phases).
    const double qx = q_frac[0] * two_pi / grid.L[0];
    const double qy = q_frac[1] * two_pi / grid.L[1];
    const double qz = q_frac[2] * two_pi / grid.L[2];

    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    double* rbuf = fft.r();
    fftw_complex* cbuf = fft.c();
    for (long i = 0; i < ntot; ++i)
        rbuf[i] = rho_r[static_cast<std::size_t>(i)];

    fft.execute_r2c();

    // V(G) = 4π / |q+G|² · ρ(G). Never drop G=0 when q ≠ 0.
    for (int i0 = 0; i0 < n0; ++i0) {
        const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
        const double gx = m0 * gx_unit;
        for (int i1 = 0; i1 < n1; ++i1) {
            const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
            const double gy = m1 * gy_unit;
            for (int i2 = 0; i2 < n2c; ++i2) {
                const int m2 = i2;
                const double gz = m2 * gz_unit;
                const long idx = (static_cast<long>(i0) * n1 + i1) * n2c + i2;
                const double gqx = gx + qx;
                const double gqy = gy + qy;
                const double gqz = gz + qz;
                const double g2 = gqx * gqx + gqy * gqy + gqz * gqz;
                if (g2 < 1e-30) {
                    cbuf[idx][0] = 0.0;
                    cbuf[idx][1] = 0.0;
                } else {
                    const double f = four_pi / g2;
                    cbuf[idx][0] *= f;
                    cbuf[idx][1] *= f;
                }
            }
        }
    }

    fft.execute_c2r();

    PoissonResult out;
    out.v_r.resize(static_cast<std::size_t>(ntot));
    const double inv_n = 1.0 / static_cast<double>(ntot);
    for (long i = 0; i < ntot; ++i)
        out.v_r[static_cast<std::size_t>(i)] = rbuf[i] * inv_n;

    const double w = grid.weight();
    double e = 0.0;
    for (long i = 0; i < ntot; ++i)
        e += rho_r[static_cast<std::size_t>(i)] * out.v_r[static_cast<std::size_t>(i)];
    out.e_hartree = 0.5 * w * e;

    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
