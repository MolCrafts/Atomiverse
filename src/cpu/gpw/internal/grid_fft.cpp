/// @file src/cpu/gpw/internal/grid_fft.cpp
/// @brief Implementation of spectral gradient / divergence (see header).
///
/// Convention mirrors poisson_fft: FFTW r2c forward (unnormalized), apply i*G_d,
/// c2r backward returns N*result, divide by N = nx*ny*nz. G_d = 2 pi m_d / L_d
/// with signed FFT frequencies. The Nyquist mode (2|m_d| == n_d) of an
/// odd-order derivative is zeroed so the inverse transform is real.

#include "atomiverse/cpu/gpw/internal/grid_fft.h"

#include <cmath>

#include <fftw3.h>

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
    double* rbuf = static_cast<double*>(fftw_malloc(sizeof(double) * g.ntot));
    fftw_complex* fhat =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * g.ncplx));
    fftw_complex* cwork =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * g.ncplx));
    double* rout = static_cast<double*>(fftw_malloc(sizeof(double) * g.ntot));

    for (long i = 0; i < g.ntot; ++i)
        rbuf[i] = f[i];
    fftw_plan fwd =
        fftw_plan_dft_r2c_3d(g.n0, g.n1, g.n2, rbuf, fhat, FFTW_ESTIMATE);
    fftw_plan bwd =
        fftw_plan_dft_c2r_3d(g.n0, g.n1, g.n2, cwork, rout, FFTW_ESTIMATE);
    fftw_execute(fwd);

    std::array<std::vector<double>, 3> out;
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
    for (int d = 0; d < 3; ++d) {
        for (long idx = 0; idx < g.ncplx; ++idx) {
            const double gd = gcomp(g, idx, d);
            // i*gd * (re + i*im) = -gd*im + i*gd*re
            cwork[idx][0] = -gd * fhat[idx][1];
            cwork[idx][1] = gd * fhat[idx][0];
        }
        fftw_execute(bwd); // cwork -> rout (= N * d f / d x_d)
        out[d].resize(static_cast<std::size_t>(g.ntot));
        for (long i = 0; i < g.ntot; ++i)
            out[d][i] = rout[i] * inv_n;
    }

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(rbuf);
    fftw_free(fhat);
    fftw_free(cwork);
    fftw_free(rout);
    return out;
}

std::vector<double> divergence(const UniformGrid& grid,
                               const std::array<std::vector<double>, 3>& v) {
    const GInfo g = make_ginfo(grid);
    double* rbuf = static_cast<double*>(fftw_malloc(sizeof(double) * g.ntot));
    fftw_complex* fhat =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * g.ncplx));
    fftw_complex* acc =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * g.ncplx));
    double* rout = static_cast<double*>(fftw_malloc(sizeof(double) * g.ntot));

    for (long idx = 0; idx < g.ncplx; ++idx) {
        acc[idx][0] = 0.0;
        acc[idx][1] = 0.0;
    }

    fftw_plan fwd =
        fftw_plan_dft_r2c_3d(g.n0, g.n1, g.n2, rbuf, fhat, FFTW_ESTIMATE);
    fftw_plan bwd =
        fftw_plan_dft_c2r_3d(g.n0, g.n1, g.n2, acc, rout, FFTW_ESTIMATE);

    for (int d = 0; d < 3; ++d) {
        for (long i = 0; i < g.ntot; ++i)
            rbuf[i] = v[d][i];
        fftw_execute(fwd); // rbuf -> fhat
        for (long idx = 0; idx < g.ncplx; ++idx) {
            const double gd = gcomp(g, idx, d);
            // accumulate i*gd * fhat
            acc[idx][0] += -gd * fhat[idx][1];
            acc[idx][1] += gd * fhat[idx][0];
        }
    }
    fftw_execute(bwd); // acc -> rout (= N * div v)

    std::vector<double> out(static_cast<std::size_t>(g.ntot));
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
    for (long i = 0; i < g.ntot; ++i)
        out[i] = rout[i] * inv_n;

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(rbuf);
    fftw_free(fhat);
    fftw_free(acc);
    fftw_free(rout);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
