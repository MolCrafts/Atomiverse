/// @file tests/cpu/gpw/test_cpu_gpw_grid_fft.cpp
/// @brief Spectral gradient / divergence on the uniform grid (band-limited exact).

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"

using ATV::cpu::gpw::divergence;
using ATV::cpu::gpw::gradient;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;

namespace {
// f(r) = cos(G.r), G = 2 pi k / L (cubic box).
struct Mode {
    UniformGrid grid;
    double G[3];
    std::vector<double> f;
};
Mode build_mode(double L, double cutoff, int kx, int ky, int kz) {
    Mode m;
    const double cell[3] = {L, L, L};
    m.grid = make_uniform_grid(cell, cutoff);
    m.G[0] = 2.0 * M_PI * kx / L;
    m.G[1] = 2.0 * M_PI * ky / L;
    m.G[2] = 2.0 * M_PI * kz / L;
    const long ng = m.grid.npts();
    m.f.resize(static_cast<std::size_t>(ng));
    for (long g = 0; g < ng; ++g) {
        double r[3];
        m.grid.point(g, r);
        m.f[g] = std::cos(m.G[0] * r[0] + m.G[1] * r[1] + m.G[2] * r[2]);
    }
    return m;
}
} // namespace

// grad cos(G.r) = -G sin(G.r), exact for a band-limited mode.
TEST(GpwGridFFT, GradientOfPlaneWave) {
    Mode m = build_mode(10.0, 200.0, 2, 1, 0);
    auto grad = gradient(m.grid, m.f);
    const long ng = m.grid.npts();
    double max_err = 0.0;
    for (long g = 0; g < ng; ++g) {
        double r[3];
        m.grid.point(g, r);
        const double s =
            std::sin(m.G[0] * r[0] + m.G[1] * r[1] + m.G[2] * r[2]);
        for (int d = 0; d < 3; ++d)
            max_err = std::max(max_err, std::abs(grad[d][g] - (-m.G[d] * s)));
    }
    EXPECT_LT(max_err, 1e-9);
}

// div(grad cos(G.r)) = -|G|^2 cos(G.r) (the Laplacian of the mode).
TEST(GpwGridFFT, DivGradIsLaplacian) {
    Mode m = build_mode(10.0, 200.0, 1, 2, 1);
    auto grad = gradient(m.grid, m.f);
    auto div = divergence(m.grid, grad);
    const double g2 = m.G[0] * m.G[0] + m.G[1] * m.G[1] + m.G[2] * m.G[2];
    const long ng = m.grid.npts();
    double max_err = 0.0;
    for (long g = 0; g < ng; ++g)
        max_err = std::max(max_err, std::abs(div[g] + g2 * m.f[g]));
    EXPECT_LT(max_err, 1e-8);
}

// Divergence of an explicit field v = (sin(Gx x), 0, 0): div = Gx cos(Gx x).
TEST(GpwGridFFT, DivergenceOfKnownField) {
    const double L = 8.0;
    const double cell[3] = {L, L, L};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double Gx = 2.0 * M_PI * 1 / L;
    const long ng = grid.npts();
    std::array<std::vector<double>, 3> v;
    for (int d = 0; d < 3; ++d)
        v[d].assign(static_cast<std::size_t>(ng), 0.0);
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        v[0][g] = std::sin(Gx * r[0]);
    }
    auto div = divergence(grid, v);
    double max_err = 0.0;
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        max_err =
            std::max(max_err, std::abs(div[g] - Gx * std::cos(Gx * r[0])));
    }
    EXPECT_LT(max_err, 1e-9);
}
