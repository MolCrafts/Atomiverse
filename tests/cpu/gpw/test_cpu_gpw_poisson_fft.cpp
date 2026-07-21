/// @file tests/cpu/gpw/test_cpu_gpw_poisson_fft.cpp
/// @brief GPW periodic FFT Poisson solver: band-limited round trips.
///
/// A single plane wave V(r)=cos(G·r) is exactly represented on the grid, with
/// ρ = -∇²V/4π = (|G|²/4π) V. solve_poisson(ρ) must return V (both zero-mean, so
/// the zeroed G=0 component is irrelevant). This isolates the G-vector indexing
/// and FFTW normalization from any discretization error.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"

using ATV::cpu::gpw::PoissonResult;
using ATV::cpu::gpw::solve_poisson;
using ATV::cpu::gpw::UniformGrid;

namespace {

UniformGrid cubic_grid(int n, double L) {
    UniformGrid g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = n;
        g.L[d] = L;
        g.origin[d] = 0.0;
    }
    return g;
}

// G-vector (physical) for integer wavevector k on an orthorhombic cell.
struct Gvec {
    double g[3];
    double g2;
};
Gvec gvec(const UniformGrid& grid, const int k[3]) {
    Gvec out;
    out.g2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        out.g[d] = 2.0 * M_PI * k[d] / grid.L[d];
        out.g2 += out.g[d] * out.g[d];
    }
    return out;
}

} // namespace

// Single plane wave: V(r)=cos(G.r), rho=(|G|^2/4pi)V -> solver returns V.
TEST(GpwPoissonFFT, SinglePlaneWaveRoundTrip) {
    UniformGrid grid = cubic_grid(16, 10.0);
    const int k[3] = {1, 0, 0};
    const Gvec G = gvec(grid, k);

    const long ng = grid.npts();
    std::vector<double> v_exact(static_cast<std::size_t>(ng));
    std::vector<double> rho(static_cast<std::size_t>(ng));
    const double pref = G.g2 / (4.0 * M_PI);
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double phase = G.g[0] * r[0] + G.g[1] * r[1] + G.g[2] * r[2];
        v_exact[i] = std::cos(phase);
        rho[i] = pref * v_exact[i];
    }

    PoissonResult res = solve_poisson(grid, rho);
    for (long i = 0; i < ng; ++i)
        ASSERT_NEAR(res.v_r[i], v_exact[i], 1e-9) << "at grid point " << i;

    // e_hartree == 0.5 Σ w ρ V, recomputed independently.
    double e_manual = 0.0;
    for (long i = 0; i < ng; ++i)
        e_manual += rho[i] * res.v_r[i];
    e_manual *= 0.5 * grid.weight();
    EXPECT_NEAR(res.e_hartree, e_manual,
                1e-12 * std::max(1.0, std::abs(e_manual)));
    EXPECT_GT(res.e_hartree, 0.0);
}

// Superposition of two wavevectors with different amplitudes.
TEST(GpwPoissonFFT, SuperpositionRoundTrip) {
    UniformGrid grid = cubic_grid(20, 12.0);
    const int k1[3] = {1, 0, 0};
    const int k2[3] = {0, 2, 0};
    const Gvec G1 = gvec(grid, k1);
    const Gvec G2 = gvec(grid, k2);
    const double a1 = 1.0, a2 = 0.5;

    const long ng = grid.npts();
    std::vector<double> v_exact(static_cast<std::size_t>(ng));
    std::vector<double> rho(static_cast<std::size_t>(ng));
    for (long i = 0; i < ng; ++i) {
        double r[3];
        grid.point(i, r);
        const double p1 = G1.g[0] * r[0] + G1.g[1] * r[1] + G1.g[2] * r[2];
        const double p2 = G2.g[0] * r[0] + G2.g[1] * r[1] + G2.g[2] * r[2];
        v_exact[i] = a1 * std::cos(p1) + a2 * std::cos(p2);
        rho[i] = a1 * (G1.g2 / (4.0 * M_PI)) * std::cos(p1)
                 + a2 * (G2.g2 / (4.0 * M_PI)) * std::cos(p2);
    }

    PoissonResult res = solve_poisson(grid, rho);
    for (long i = 0; i < ng; ++i)
        ASSERT_NEAR(res.v_r[i], v_exact[i], 1e-9) << "at grid point " << i;
}

// Constant density is pure G=0: zeroed -> V==0, e_hartree==0.
TEST(GpwPoissonFFT, ConstantDensityGivesZero) {
    UniformGrid grid = cubic_grid(12, 8.0);
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.37);
    PoissonResult res = solve_poisson(grid, rho);
    for (double v : res.v_r)
        EXPECT_NEAR(v, 0.0, 1e-10);
    EXPECT_NEAR(res.e_hartree, 0.0, 1e-10);
}
