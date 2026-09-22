/// @file src/cpu/gpw/internal/grid_fft.cpp
/// @brief Implementation of spectral gradient / divergence (see header).
///
/// Convention mirrors poisson_fft: FFTW r2c forward (unnormalized), apply i*G_d,
/// c2r backward returns N*result, divide by N = nx*ny*nz. G_d = 2 pi m_d / L_d
/// with signed FFT frequencies. The Nyquist mode (2|m_d| == n_d) of an
/// odd-order derivative is zeroed so the inverse transform is real.
///
/// GGA residual: one forward + fused howmany=3 multi-out inverse for gradient;
/// howmany=3 multi-out forward + one inverse for divergence. G-tables are
/// cached per (shape, L). Input packing uses memcpy (not OMP scatter).

#include "atomiverse/cpu/gpw/internal/grid_fft.h"

#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "atomiverse/cpu/gpw/internal/fft_cache.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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

// Signed wavevector tables G_d[idx] = m_d * (2 pi / L_d) with Nyquist zeroed.
// Keyed by shape + box lengths so GGA SCF reuses the same tables every step.
struct GTables {
    std::vector<double> gx, gy, gz;
};

const GTables& g_tables(const GInfo& g) {
    using Key = std::tuple<int, int, int, double, double, double>;
    thread_local std::map<Key, GTables> cache;
    const Key key{g.n0, g.n1, g.n2, g.gu[0], g.gu[1], g.gu[2]};
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    GTables t;
    t.gx.resize(static_cast<std::size_t>(g.ncplx));
    t.gy.resize(static_cast<std::size_t>(g.ncplx));
    t.gz.resize(static_cast<std::size_t>(g.ncplx));
    const long stride_i0 = static_cast<long>(g.n2c) * g.n1;
    for (int i0 = 0; i0 < g.n0; ++i0) {
        int m0 = (i0 <= g.n0 / 2) ? i0 : i0 - g.n0;
        if (2 * (m0 < 0 ? -m0 : m0) == g.n0)
            m0 = 0; // Nyquist
        const double g0 = m0 * g.gu[0];
        for (int i1 = 0; i1 < g.n1; ++i1) {
            int m1 = (i1 <= g.n1 / 2) ? i1 : i1 - g.n1;
            if (2 * (m1 < 0 ? -m1 : m1) == g.n1)
                m1 = 0;
            const double g1 = m1 * g.gu[1];
            const long base = static_cast<long>(i0) * stride_i0
                              + static_cast<long>(i1) * g.n2c;
            for (int i2 = 0; i2 < g.n2c; ++i2) {
                int m2 = i2; // r2c last dim has only m>=0
                if (2 * m2 == g.n2)
                    m2 = 0;
                const double g2 = m2 * g.gu[2];
                const long idx = base + i2;
                t.gx[static_cast<std::size_t>(idx)] = g0;
                t.gy[static_cast<std::size_t>(idx)] = g1;
                t.gz[static_cast<std::size_t>(idx)] = g2;
            }
        }
    }
    auto [ins, ok] = cache.emplace(key, std::move(t));
    (void)ok;
    return ins->second;
}

} // namespace

std::array<std::vector<double>, 3> gradient(const UniformGrid& grid,
                                            const std::vector<double>& f) {
    const GInfo g = make_ginfo(grid);
    R2cFft& fft = R2cFft::for_shape(g.n0, g.n1, g.n2);
    double* rbuf = fft.r();
    fftw_complex* fhat = fft.c();
    fftw_complex* c_many = fft.c_many();
    double* r_many = fft.r_many();
    const GTables& gt = g_tables(g);

    // Contiguous pack — avoid OMP-parallel element scatter on the hot path.
    if (static_cast<long>(f.size()) != g.ntot)
        throw std::invalid_argument(
            "gradient: f length must equal grid.npts()");
    std::memcpy(rbuf, f.data(),
                static_cast<std::size_t>(g.ntot) * sizeof(double));
    fft.execute_r2c(); // rbuf -> fhat

    // Fuse spectral multiply for all 3 Cartesian derivatives into c_many.
#pragma omp parallel for schedule(static)
    for (long idx = 0; idx < g.ncplx; ++idx) {
        const double re = fhat[idx][0];
        const double im = fhat[idx][1];
        const double gx = gt.gx[static_cast<std::size_t>(idx)];
        const double gy = gt.gy[static_cast<std::size_t>(idx)];
        const double gz = gt.gz[static_cast<std::size_t>(idx)];
        // i*g * (re + i*im) = -g*im + i*g*re
        const long bx = idx;
        const long by = idx + g.ncplx;
        const long bz = idx + 2 * g.ncplx;
        c_many[bx][0] = -gx * im;
        c_many[bx][1] = gx * re;
        c_many[by][0] = -gy * im;
        c_many[by][1] = gy * re;
        c_many[bz][0] = -gz * im;
        c_many[bz][1] = gz * re;
    }
    fft.execute_many_c2r(); // c_many -> r_many (= N * grad_d)

    std::array<std::vector<double>, 3> out;
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
    for (int d = 0; d < 3; ++d) {
        out[d].resize(static_cast<std::size_t>(g.ntot));
        const double* src = r_many + static_cast<long>(d) * g.ntot;
        double* dst = out[d].data();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < g.ntot; ++i)
            dst[i] = src[i] * inv_n;
    }
    return out;
}

std::vector<double> divergence(const UniformGrid& grid,
                               const std::array<std::vector<double>, 3>& v) {
    const GInfo g = make_ginfo(grid);
    R2cFft& fft = R2cFft::for_shape(g.n0, g.n1, g.n2);
    double* r_many = fft.r_many();
    fftw_complex* c_many = fft.c_many();
    fftw_complex* acc = fft.c2();
    double* rout = fft.r2();
    const GTables& gt = g_tables(g);

    for (int d = 0; d < 3; ++d) {
        if (static_cast<long>(v[d].size()) != g.ntot)
            throw std::invalid_argument(
                "divergence: v component length must equal grid.npts()");
        std::memcpy(r_many + static_cast<long>(d) * g.ntot, v[d].data(),
                    static_cast<std::size_t>(g.ntot) * sizeof(double));
    }
    fft.execute_many_r2c(); // r_many -> c_many

    // acc = Σ_d i G_d · vhat_d
#pragma omp parallel for schedule(static)
    for (long idx = 0; idx < g.ncplx; ++idx) {
        const double gx = gt.gx[static_cast<std::size_t>(idx)];
        const double gy = gt.gy[static_cast<std::size_t>(idx)];
        const double gz = gt.gz[static_cast<std::size_t>(idx)];
        const long bx = idx;
        const long by = idx + g.ncplx;
        const long bz = idx + 2 * g.ncplx;
        // accumulate i*g * fhat = (-g*im, g*re)
        const double re =
            -gx * c_many[bx][1] - gy * c_many[by][1] - gz * c_many[bz][1];
        const double im =
            gx * c_many[bx][0] + gy * c_many[by][0] + gz * c_many[bz][0];
        acc[idx][0] = re;
        acc[idx][1] = im;
    }
    fft.execute_c2r_alt(); // acc -> rout (= N * div v)

    std::vector<double> out(static_cast<std::size_t>(g.ntot));
    const double inv_n = 1.0 / static_cast<double>(g.ntot);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < g.ntot; ++i)
        out[static_cast<std::size_t>(i)] = rout[i] * inv_n;

    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
