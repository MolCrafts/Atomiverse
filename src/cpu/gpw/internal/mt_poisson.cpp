/// @file src/cpu/gpw/internal/mt_poisson.cpp
/// @brief Implementation of the isolated (truncated-Coulomb) Poisson solver.
///
/// Same FFTW r2c/c2r pipeline and G-vector / 1/N normalization as the periodic
/// solver (`poisson_fft.cpp`); only the reciprocal-space kernel changes to the
/// spherically-truncated Coulomb form (Rozzi et al., PRB 73, 205119 (2006)):
///   v(G) = (4π/|G|²)[1 - cos(|G| R_c)]   for G≠0,    v(0) = 2π R_c²,
/// with R_c = min(L)/2. For a charge contained within R_c this reproduces the
/// open-boundary 1/r potential, removing the periodic-image interaction.
///
/// Geometry-fixed kernel table is cached (shape + L) so SCF iters skip the
/// √/cos rebuild; apply / H2D-scale loops are OpenMP.

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

// Grow-only MT kernel cache: v(G) depends only on grid shape + box + R_c.
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
        const long nc = static_cast<long>(a) * static_cast<long>(b)
                        * static_cast<long>(n2c);
        f.resize(static_cast<std::size_t>(nc));
        const double two_pi = 2.0 * M_PI;
        const double gx_unit = two_pi / l0;
        const double gy_unit = two_pi / l1;
        const double gz_unit = two_pi / l2;
        const double four_pi = 4.0 * M_PI;
        const double rc = 0.5 * std::min({l0, l1, l2});
        const double v_g0 = two_pi * rc * rc;
#pragma omp parallel for schedule(static) collapse(2)
        for (int i0 = 0; i0 < a; ++i0) {
            for (int i1 = 0; i1 < b; ++i1) {
                const int m0 = (i0 <= a / 2) ? i0 : i0 - a;
                const int m1 = (i1 <= b / 2) ? i1 : i1 - b;
                const double gx = m0 * gx_unit;
                const double gy = m1 * gy_unit;
                const double gxy2 = gx * gx + gy * gy;
                double* row = &f[static_cast<std::size_t>(
                    (static_cast<long>(i0) * b + i1) * n2c)];
                for (int i2 = 0; i2 < n2c; ++i2) {
                    const double gz = i2 * gz_unit;
                    const double g2 = gxy2 + gz * gz;
                    if (g2 == 0.0)
                        row[i2] = v_g0;
                    else {
                        const double g = std::sqrt(g2);
                        row[i2] = (four_pi / g2) * (1.0 - std::cos(g * rc));
                    }
                }
            }
        }
    }
};

MtKernelCache& mt_kernel_cache() {
    static MtKernelCache cache;
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

    // Apply cached truncated-Coulomb kernel (bit-identical to on-the-fly f(G)).
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
