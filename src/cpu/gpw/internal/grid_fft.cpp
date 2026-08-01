/// @file src/cpu/gpw/internal/grid_fft.cpp
/// @brief Implementation of spectral gradient / divergence (see header).
///
/// Convention mirrors poisson_fft: FFTW r2c forward (unnormalized), apply i*G_d,
/// c2r backward returns N*result, divide by N = nx*ny*nz. G_d = 2 pi m_d / L_d
/// with signed FFT frequencies. The Nyquist mode (2|m_d| == n_d) of an
/// odd-order derivative is zeroed so the inverse transform is real.

#include "atomiverse/cpu/gpw/internal/grid_fft.h"

#include <cmath>

#include "atomiverse/cpu/gpw/internal/fft_cache.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

// Per-dimension reciprocal-lattice spacing.
struct GInfo {
    int n0, n1, n2, n2c;
    long ntot, ncplx;
    double gu[3];
};

GInfo make_ginfo(const UniformGrid& grid) {
    GInfo g;
    g.n0 = grid.n[0];
    g.n1 = grid.n[1];
    g.n2 = grid.n[2];
    g.n2c = g.n2 / 2 + 1;
    g.ntot = static_cast<long>(g.n0) * g.n1 * g.n2;
    g.ncplx = static_cast<long>(g.n0) * g.n1 * g.n2c;
    g.gu[0] = 2.0 * M_PI / grid.L[0];
    g.gu[1] = 2.0 * M_PI / grid.L[1];
    g.gu[2] = 2.0 * M_PI / grid.L[2];
    return g;
}

// Signed wavevector component along dim d for complex index idx, with the
// Nyquist mode zeroed (odd-derivative convention).
inline double gcomp(const GInfo& g, long idx, int d) {
    const int i2 = static_cast<int>(idx % g.n2c);
    const int i1 = static_cast<int>((idx / g.n2c) % g.n1);
    const int i0 = static_cast<int>(idx / (static_cast<long>(g.n2c) * g.n1));
    int n, i;
    if (d == 0) {
        n = g.n0;
        i = i0;
    } else if (d == 1) {
        n = g.n1;
        i = i1;
    } else {
        n = g.n2;
        i = i2;
    }
    const int m = (i <= n / 2) ? i : i - n; // r2c last dim has only m>=0
    if (2 * (m < 0 ? -m : m) == n)
        return 0.0; // Nyquist -> zero
    return m * g.gu[d];
}

} // namespace

std::array<std::vector<double>, 3> gradient(const UniformGrid& grid,
                                            const std::vector<double>& f) {
    const GInfo g = make_ginfo(grid);
    R2cFft& fft = R2cFft::for_shape(g.n0, g.n1, g.n2);
    double* rbuf = fft.r();
    fftw_complex* fhat = fft.c();
    fftw_complex* cwork = fft.c2();
    double* rout = fft.r2();

    for (long i = 0; i < g.ntot; ++i)
        rbuf[i] = f[static_cast<std::size_t>(i)];
    fft.execute_r2c(); // rbuf -> fhat

    std::array<std::vector<double>, 3> out;
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
    for (int d = 0; d < 3; ++d) {
        for (long idx = 0; idx < g.ncplx; ++idx) {
            const double gd = gcomp(g, idx, d);
            // i*gd * (re + i*im) = -gd*im + i*gd*re
            cwork[idx][0] = -gd * fhat[idx][1];
            cwork[idx][1] = gd * fhat[idx][0];
        }
        fft.execute_c2r_alt(); // cwork -> rout (= N * d f / d x_d)
        out[d].resize(static_cast<std::size_t>(g.ntot));
        for (long i = 0; i < g.ntot; ++i)
            out[d][static_cast<std::size_t>(i)] = rout[i] * inv_n;
    }

    return out;
}

std::vector<double> divergence(const UniformGrid& grid,
                               const std::array<std::vector<double>, 3>& v) {
    const GInfo g = make_ginfo(grid);
    R2cFft& fft = R2cFft::for_shape(g.n0, g.n1, g.n2);
    double* rbuf = fft.r();
    fftw_complex* fhat = fft.c();
    fftw_complex* acc = fft.c2();
    double* rout = fft.r2();

    for (long idx = 0; idx < g.ncplx; ++idx) {
        acc[idx][0] = 0.0;
        acc[idx][1] = 0.0;
    }

    for (int d = 0; d < 3; ++d) {
        for (long i = 0; i < g.ntot; ++i)
            rbuf[i] = v[d][static_cast<std::size_t>(i)];
        fft.execute_r2c(); // rbuf -> fhat
        for (long idx = 0; idx < g.ncplx; ++idx) {
            const double gd = gcomp(g, idx, d);
            // accumulate i*gd * fhat
            acc[idx][0] += -gd * fhat[idx][1];
            acc[idx][1] += gd * fhat[idx][0];
        }
    }
    fft.execute_c2r_alt(); // acc -> rout (= N * div v)

    std::vector<double> out(static_cast<std::size_t>(g.ntot));
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
    for (long i = 0; i < g.ntot; ++i)
        out[static_cast<std::size_t>(i)] = rout[i] * inv_n;

    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
