/// @file tests/cpu/gpw/test_cpu_gpw_grid_fft.cpp
/// @brief Spectral gradient / divergence on the uniform grid (band-limited exact).

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/fft_cache.h"
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

// Analytic inverse transform, without a forward-transform twin. The first c2r
// call must not let MEASURE planning destroy live primary/alternate arrays.
TEST(GpwGridFFT, FirstBackwardPreservesLiveBuffersAndAnalyticMode) {
    using ATV::cpu::gpw::FftwBuffer;
    using ATV::cpu::gpw::R2cFft;
    const int shapes[3][3] = {{5, 7, 8}, {7, 5, 8}, {5, 8, 7}};
    for (int path = 0; path < 3; ++path) {
        SCOPED_TRACE(path); // primary, alternate, caller-owned
        const auto& shape = shapes[path];
        auto& fft = R2cFft::for_shape(shape[0], shape[1], shape[2]);
        const long nr = fft.ntot(), nc = fft.ncplx();
        auto fill_complex = [nc](fftw_complex* data, double re, double im) {
            for (long i = 0; i < nc; ++i) {
                data[i][0] = re;
                data[i][1] = im;
            }
        };
        auto expect_real = [nr](const double* data, double value) {
            double error = 0;
            for (long i = 0; i < nr; ++i) {
                ASSERT_TRUE(std::isfinite(data[i]));
                error = std::max(error, std::abs(data[i] - value));
            }
            EXPECT_DOUBLE_EQ(error, 0);
        };
        auto expect_complex = [nc](const fftw_complex* data, double re,
                                   double im) {
            double error = 0;
            for (long i = 0; i < nc; ++i) {
                ASSERT_TRUE(std::isfinite(data[i][0]));
                ASSERT_TRUE(std::isfinite(data[i][1]));
                error = std::max(error, std::abs(data[i][0] - re));
                error = std::max(error, std::abs(data[i][1] - im));
            }
            EXPECT_DOUBLE_EQ(error, 0);
        };
        std::fill_n(fft.r(), nr, 13.0);
        fill_complex(fft.c(), 17.0, -19.0);
        std::fill_n(fft.r2(), nr, 23.0);
        fill_complex(fft.c2(), 29.0, -31.0);
        FftwBuffer<double> external_r(static_cast<std::size_t>(nr));
        FftwBuffer<fftw_complex> external_c(static_cast<std::size_t>(nc));
        double* output = path == 0   ? fft.r()
                         : path == 1 ? fft.r2()
                                     : external_r.data();
        fftw_complex* input = path == 0   ? fft.c()
                              : path == 1 ? fft.c2()
                                          : external_c.data();
        fill_complex(input, 0, 0);
        // f(x,y,z)=2+cos(2*pi*x/n0)+0.5*sin(2*pi*z/n2).
        // FFTW is unnormalized; these are its exact DFT coefficients.
        input[0][0] = 2.0 * nr;
        input[shape[1] * fft.n2c()][0] = 0.5 * nr;
        input[(shape[0] - 1) * shape[1] * fft.n2c()][0] = 0.5 * nr;
        input[1][1] = -0.25 * nr;
        if (path == 0)
            fft.execute_c2r();
        else if (path == 1)
            fft.execute_c2r_alt();
        else
            fft.execute_c2r(input, output);

        double error = 0;
        for (int x = 0; x < shape[0]; ++x)
            for (int y = 0; y < shape[1]; ++y)
                for (int z = 0; z < shape[2]; ++z) {
                    const long i = (x * shape[1] + y) * shape[2] + z;
                    ASSERT_TRUE(std::isfinite(output[i]));
                    const double exact =
                        2 + std::cos(2 * M_PI * x / shape[0])
                        + 0.5 * std::sin(2 * M_PI * z / shape[2]);
                    error = std::max(error, std::abs(output[i] / nr - exact));
                }
        EXPECT_LT(error, 1e-12);
        // c2r itself may destroy its input; unrelated work arrays must survive.
        if (path != 0) {
            expect_real(fft.r(), 13.0);
            expect_complex(fft.c(), 17.0, -19.0);
        }
        if (path != 1) {
            expect_real(fft.r2(), 23.0);
            expect_complex(fft.c2(), 29.0, -31.0);
        }
    }
}
