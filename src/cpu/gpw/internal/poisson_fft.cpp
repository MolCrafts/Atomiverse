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

#include <fftw3.h>

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
    const long ncplx = static_cast<long>(n0) * n1 * n2c;

    const double two_pi = 2.0 * M_PI;
    const double gx_unit = two_pi / grid.L[0];
    const double gy_unit = two_pi / grid.L[1];
    const double gz_unit = two_pi / grid.L[2];
    const double four_pi = 4.0 * M_PI;

    // Allocate FFTW buffers. r2c reads a real array of ntot, writes ncplx complex.
    double* rbuf = static_cast<double*>(fftw_malloc(sizeof(double) * ntot));
    fftw_complex* cbuf =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * ncplx));

    for (long i = 0; i < ntot; ++i)
        rbuf[i] = rho_r[i];

    fftw_plan fwd = fftw_plan_dft_r2c_3d(n0, n1, n2, rbuf, cbuf, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_c2r_3d(n0, n1, n2, cbuf, rbuf, FFTW_ESTIMATE);

    fftw_execute(fwd); // rbuf -> cbuf (= rho_hat)

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

    fftw_execute(bwd); // cbuf -> rbuf (= N * V(r))

    PoissonResult out;
    out.v_r.resize(static_cast<std::size_t>(ntot));
    const double inv_n = 1.0 / static_cast<double>(ntot);
    for (long i = 0; i < ntot; ++i)
        out.v_r[i] = rbuf[i] * inv_n;

    // Hartree energy 0.5 Σ_g w ρ_g V_g.
    const double w = grid.weight();
    double e = 0.0;
    for (long i = 0; i < ntot; ++i)
        e += rho_r[i] * out.v_r[i];
    out.e_hartree = 0.5 * w * e;

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(rbuf);
    fftw_free(cbuf);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
