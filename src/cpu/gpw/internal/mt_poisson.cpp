/// @file src/cpu/gpw/internal/mt_poisson.cpp
/// @brief Martyna–Tuckerman isolated Poisson kernel and cached FFT application.
///
/// Adapted from CP2K v2026.2 src/pw/mt_util.F (blob
/// 148c8f3b92ba5ec469c2d31e72572343cc8b517c) and pw_poisson_types.F.
/// Copyright 2000-2026 CP2K developers group <https://cp2k.org>.
/// CP2K source license: GPL-2.0-or-later; this project uses GPL-2.0-only.
/// Martyna & Tuckerman, J. Chem. Phys. 110, 2810 (1999).
///
/// Fourier transform erf(alpha*r)/r on an auxiliary grid, then add the
/// analytic short-range complement. No spherical interaction cutoff is used.

#include "atomiverse/cpu/gpw/internal/mt_poisson.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/gpw/internal/fft_cache.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

// MT kernel cache: v(G) depends only on grid shape, box and fixed alpha*L.
// SCF reuses the same isolated grid every step; rebuild only on shape change.
struct MtKernelCache {
    int n0 = 0, n1 = 0, n2 = 0;
    double L0 = 0, L1 = 0, L2 = 0;
    std::vector<double> f; ///< Hermitian length n0*n1*(n2/2+1)
    bool matches(int a, int b, int c, double l0, double l1, double l2) const {
        return n0 == a && n1 == b && n2 == c && L0 == l0 && L1 == l1
               && L2 == l2;
    }
    void rebuild(int a, int b, int c, double l0, double l1, double l2) {
        n0 = a;
        n1 = b;
        n2 = c;
        L0 = l0;
        L1 = l1;
        L2 = l2;
        const int n2c = c / 2 + 1;
        const long nc = static_cast<long>(a) * b * n2c;
        f.resize(static_cast<std::size_t>(nc));
        // CP2K's default alpha*min(L)=7. Sample the smooth long-range part
        // at a finer cutoff, then retain the original reciprocal-space modes.
        const double alpha = 7.0 / std::min({l0, l1, l2});
        const double alpha2 = alpha * alpha;
        const int aa = 2 * a, bb = 2 * b, cc = 2 * c;
        const int cc2 = cc / 2 + 1;
        auto& auxiliary = R2cFft::for_shape(aa, bb, cc);
        double* values = auxiliary.r();
        const double volume = l0 * l1 * l2;
        const double dv = volume / (static_cast<double>(aa) * bb * cc);
#pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < aa; ++i) {
            for (int j = 0; j < bb; ++j) {
                const double x = std::min(i, aa - i) * l0 / aa;
                const double y = std::min(j, bb - j) * l1 / bb;
                for (int k = 0; k < cc; ++k) {
                    const double z = std::min(k, cc - k) * l2 / cc;
                    const double r = std::sqrt(x * x + y * y + z * z);
                    values[(static_cast<long>(i) * bb + j) * cc + k] =
                        r > 0 ? std::erf(alpha * r) / r
                              : 2 * alpha / std::sqrt(M_PI);
                }
            }
        }
        auxiliary.execute_r2c();
        const fftw_complex* transformed = auxiliary.c();
#pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                const int mi = i <= a / 2 ? i : i - a;
                const int mj = j <= b / 2 ? j : j - b;
                const int ai = mi < 0 ? aa + mi : mi;
                const int aj = mj < 0 ? bb + mj : mj;
                const double gx = 2 * M_PI * mi / l0;
                const double gy = 2 * M_PI * mj / l1;
                for (int k = 0; k < n2c; ++k) {
                    const double gz = 2 * M_PI * k / l2;
                    const double g2 = gx * gx + gy * gy + gz * gz;
                    const double complement =
                        g2 > 0 ? -4 * M_PI / g2 * std::expm1(-g2 / (4 * alpha2))
                               : M_PI / alpha2;
                    const long idx = (static_cast<long>(i) * b + j) * n2c + k;
                    const long aux =
                        (static_cast<long>(ai) * bb + aj) * cc2 + k;
                    f[static_cast<std::size_t>(idx)] =
                        transformed[aux][0] * dv + complement;
                }
            }
        }
    }
};

MtKernelCache& mt_kernel_cache() {
    thread_local MtKernelCache cache;
    return cache;
}

} // namespace

PoissonResult solve_poisson_mt(const UniformGrid& grid,
                               const std::vector<double>& rho_r) {
    const int n0 = grid.n[0];
    const int n1 = grid.n[1];
    const int n2 = grid.n[2];
    const long ntot = static_cast<long>(n0) * n1 * n2;
    const int n2c = n2 / 2 + 1;
    const long nc = static_cast<long>(n0) * n1 * n2c;

    MtKernelCache& ker = mt_kernel_cache();
    if (!ker.matches(n0, n1, n2, grid.L[0], grid.L[1], grid.L[2]))
        ker.rebuild(n0, n1, n2, grid.L[0], grid.L[1], grid.L[2]);
    const double* fker = ker.f.data();

    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    double* rbuf = fft.r();
    fftw_complex* cbuf = fft.c();

#pragma omp parallel for schedule(static) if (ntot > 4096)
    for (long i = 0; i < ntot; ++i)
        rbuf[i] = rho_r[static_cast<std::size_t>(i)];

    fft.execute_r2c(); // rbuf -> cbuf (= rho_hat)

    // Apply the cached Martyna–Tuckerman kernel.
#pragma omp parallel for schedule(static) if (nc > 4096)
    for (long idx = 0; idx < nc; ++idx) {
        const double f = fker[static_cast<std::size_t>(idx)];
        cbuf[idx][0] *= f;
        cbuf[idx][1] *= f;
    }

    fft.execute_c2r(); // cbuf -> rbuf (= N * V(r))

    PoissonResult out;
    out.v_r.resize(static_cast<std::size_t>(ntot));
    const double inv_n = 1.0 / static_cast<double>(ntot);
#pragma omp parallel for schedule(static) if (ntot > 4096)
    for (long i = 0; i < ntot; ++i)
        out.v_r[static_cast<std::size_t>(i)] = rbuf[i] * inv_n;

    // Hartree energy 0.5 Σ w ρ V — fixed chunk fold (thread-reproducible).
    const double w = grid.weight();
    constexpr long kEChunks = 256;
    const device::ChunkedRange range(ntot, kEChunks);
    std::vector<double> part(static_cast<std::size_t>(range.chunks()), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < range.chunks(); ++c) {
        double s = 0.0;
        for (long i = range.begin(c); i < range.end(c); ++i)
            s += rho_r[static_cast<std::size_t>(i)]
                 * out.v_r[static_cast<std::size_t>(i)];
        part[static_cast<std::size_t>(c)] = s;
    }
    double e = 0.0;
    for (double p : part)
        e += p;
    out.e_hartree = 0.5 * w * e;

    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
