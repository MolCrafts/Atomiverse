/// @file src/cpu/gpw/internal/mt_poisson.cpp
/// @brief Implementation of the isolated (truncated-Coulomb) Poisson solver.
///
/// Same FFTW r2c/c2r pipeline and G-vector / 1/N normalization as the periodic
/// solver (`poisson_fft.cpp`); only the reciprocal-space kernel changes to the
/// spherically-truncated Coulomb form (Rozzi et al., PRB 73, 205119 (2006)):
///   v(G) = (4π/|G|²)[1 - cos(|G| R_c)]   for G≠0,    v(0) = 2π R_c²,
/// with R_c = min(L)/2. For a charge contained within R_c this reproduces the
/// open-boundary 1/r potential, removing the periodic-image interaction.

#include "atomiverse/cpu/gpw/internal/mt_poisson.h"

#include <algorithm>
#include <cmath>

#include "atomiverse/cpu/gpw/internal/fft_cache.h"

namespace ATV {
namespace cpu {
namespace gpw {

PoissonResult solve_poisson_mt(const UniformGrid& grid,
                               const std::vector<double>& rho_r) {
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

    // Truncation radius: largest sphere inscribed in the orthorhombic cell.
    const double rc = 0.5 * std::min({grid.L[0], grid.L[1], grid.L[2]});
    const double v_g0 = two_pi * rc * rc; // truncated-Coulomb G=0 kernel

    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    double* rbuf = fft.r();
    fftw_complex* cbuf = fft.c();

    for (long i = 0; i < ntot; ++i)
        rbuf[i] = rho_r[static_cast<std::size_t>(i)];

    fft.execute_r2c(); // rbuf -> cbuf (= rho_hat)

    for (int i0 = 0; i0 < n0; ++i0) {
        const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
        const double gx = m0 * gx_unit;
        for (int i1 = 0; i1 < n1; ++i1) {
            const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
            const double gy = m1 * gy_unit;
            for (int i2 = 0; i2 < n2c; ++i2) {
                const double gz = i2 * gz_unit;
                const long idx = (static_cast<long>(i0) * n1 + i1) * n2c + i2;
                const double g2 = gx * gx + gy * gy + gz * gz;
                double f;
                if (g2 == 0.0) {
                    f = v_g0; // 2π R_c²
                } else {
                    const double g = std::sqrt(g2);
                    f = (four_pi / g2) * (1.0 - std::cos(g * rc));
                }
                cbuf[idx][0] *= f;
                cbuf[idx][1] *= f;
            }
        }
    }

    fft.execute_c2r(); // cbuf -> rbuf (= N * V(r))

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
