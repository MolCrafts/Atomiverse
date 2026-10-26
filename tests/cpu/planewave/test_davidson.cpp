/// @file cpu/planewave/test_davidson.cpp
/// @brief One fixed Fourier Hamiltonian eigensolve, without an SCF loop.
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <memory>
#include <xtensor/generators/xbuilder.hpp>

TEST(PwDavidson, ComplexFourierHamiltonianMatchesDenseSpectrumAndResidual) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    constexpr int ng = 16, nx = 32, bands = 2;
    PWWorkspace ws;
    ws.n_g = ng;
    ws.nr = ws.fft_nx = nx;
    ws.fft_ny = ws.fft_nz = 1;
    ws.omega = 2 * std::acos(-1.0);
    ws.fft = std::make_unique<FFTPlan3D>(nx, 1, 1);
    ws.gvec = &ws.gvec0;
    ws.T_G = xt::zeros<AccReal>({ng});
    ws.V_eff_r = xt::zeros<AccReal>({nx});
    cpu::linalg::CplxMatrix exact = xt::zeros<std::complex<double>>({ng, ng});
    for (int i = 0; i < ng; ++i) {
        const int h = i - ng / 2;
        ws.gvec0.h.push_back(h);
        ws.gvec0.k.push_back(0);
        ws.gvec0.l.push_back(0);
        ws.gvec0.g.push_back({double(h), 0, 0});
        ws.T_G[i] = 0.5 * (h + 0.17) * (h + 0.17);
        exact(i, i) = ws.T_G[i] - 0.2;
        if (i > 0) {
            // V(x)=-0.2+0.3*cos(x)+0.4*sin(x); V(+G)=0.15-0.2i.
            exact(i, i - 1) = {0.15, -0.2};
            exact(i - 1, i) = {0.15, 0.2};
        }
    }
    for (int x = 0; x < nx; ++x) {
        const double angle = 2 * std::acos(-1.0) * x / nx;
        ws.V_eff_r[x] = -0.2 + 0.3 * std::cos(angle) + 0.4 * std::sin(angle);
    }
    std::vector<double> reference;
    cpu::linalg::CplxMatrix eigenvectors;
    cpu::linalg::eigh_hermitian(exact, reference, eigenvectors);
    DavidsonSolver solver(bands, 1e-3, 40);
    solver.prepare(ws);
    for (double tolerance : {1e-3, 1e-8}) {
        SCOPED_TRACE(tolerance);
        solver.set_tolerance(tolerance);
        solver.step(ws);
        for (int band = 0; band < bands; ++band) {
            EXPECT_NEAR(ws.eigenvalues[band], reference[band], tolerance);
            double residual2 = 0;
            Complex norm{};
            for (int i = 0; i < ng; ++i) {
                Complex value{};
                for (int j = 0; j < ng; ++j)
                    value += exact(i, j) * ws.wfc(band, j);
                residual2 +=
                    std::norm(value - ws.eigenvalues[band] * ws.wfc(band, i));
                norm += std::conj(ws.wfc(band, i)) * ws.wfc(band, i);
            }
            EXPECT_NEAR(norm.real(), 1.0, 1e-12);
            EXPECT_LE(std::sqrt(residual2), tolerance);
        }
        Complex overlap{};
        for (int i = 0; i < ng; ++i)
            overlap += std::conj(ws.wfc(0, i)) * ws.wfc(1, i);
        EXPECT_LT(std::abs(overlap), 1e-12);
    }
    // A one-sweep budget returns its Ritz approximation without silently
    // doing the converged solve. The same independent H measures its residual.
    ws.wfc = xt::xarray<Complex>{};
    DavidsonSolver limited(bands, 1e-8, 1);
    limited.prepare(ws);
    limited.step(ws);
    double largest_residual2 = 0;
    for (int band = 0; band < bands; ++band) {
        double residual2 = 0;
        for (int i = 0; i < ng; ++i) {
            Complex value{};
            for (int j = 0; j < ng; ++j)
                value += exact(i, j) * ws.wfc(band, j);
            residual2 +=
                std::norm(value - ws.eigenvalues[band] * ws.wfc(band, i));
        }
        largest_residual2 = std::max(largest_residual2, residual2);
    }
    EXPECT_GT(std::sqrt(largest_residual2), 1e-8);
}

TEST(PwDavidson, ChangedHamiltonianRetainsSeedsOutsideWarmOccupiedSubspace) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    constexpr int ng = 3, nx = 8;
    PWWorkspace ws;
    ws.n_g = ng;
    ws.nr = ws.fft_nx = nx;
    ws.fft_ny = ws.fft_nz = 1;
    ws.omega = 2 * std::acos(-1.0);
    ws.fft = std::make_unique<FFTPlan3D>(nx, 1, 1);
    ws.gvec = &ws.gvec0;
    ws.T_G = xt::zeros<AccReal>({ng});
    ws.V_eff_r = xt::zeros<AccReal>({nx});
    for (int h = -1; h <= 1; ++h) {
        ws.gvec0.h.push_back(h);
        ws.gvec0.k.push_back(0);
        ws.gvec0.l.push_back(0);
        ws.gvec0.g.push_back({double(h), 0, 0});
        ws.T_G[h + 1] = h * h;
    }
    DavidsonSolver solver(1, 1e-10, 10);
    solver.prepare(ws);
    solver.step(ws);
    EXPECT_NEAR(ws.eigenvalues[0], 0.0, 1e-12);
    EXPECT_NEAR(std::abs(ws.wfc(0, 1)), 1.0, 1e-12); // old lowest: G=0

    // Same coefficient shape, but a shifted free-electron dispersion makes
    // G=-1 the lowest state. The old G=0 vector is still an exact eigenvector
    // (energy 1, residual 0); a warm-only initial subspace would return it.
    ws.gvec0.kvec = {1, 0, 0};
    ws.T_G[0] = 0;
    ws.T_G[1] = 1;
    ws.T_G[2] = 4;
    solver.step(ws);
    EXPECT_NEAR(ws.eigenvalues[0], 0.0, 1e-12);
    EXPECT_NEAR(std::abs(ws.wfc(0, 0)), 1.0, 1e-12);
    double residual2 = 0;
    for (int ig = 0; ig < ng; ++ig)
        residual2 +=
            std::norm((ws.T_G[ig] - ws.eigenvalues[0]) * ws.wfc(0, ig));
    EXPECT_LE(std::sqrt(residual2), 1e-10);
}

TEST(PwDavidson, LastIndependentDirectionCompletesFourierEigensolve) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    constexpr int ng = 5, nx = 16;
    PWWorkspace ws;
    ws.n_g = ng;
    ws.nr = ws.fft_nx = nx;
    ws.fft_ny = ws.fft_nz = 1;
    ws.omega = 2 * std::acos(-1.0);
    ws.fft = std::make_unique<FFTPlan3D>(nx, 1, 1);
    ws.gvec = &ws.gvec0;
    ws.T_G = xt::zeros<AccReal>({ng});
    ws.V_eff_r = xt::zeros<AccReal>({nx});
    cpu::linalg::CplxMatrix exact = xt::zeros<Complex>({ng, ng});
    for (int h = 0; h < ng; ++h) {
        ws.gvec0.h.push_back(h);
        ws.gvec0.k.push_back(0);
        ws.gvec0.l.push_back(0);
        ws.gvec0.g.push_back({double(h), 0, 0});
        ws.T_G[h] = 0.5 * h * h;
        exact(h, h) = ws.T_G[h];
        if (h > 0) {
            exact(h, h - 1) = {0.15, -0.2};
            exact(h - 1, h) = {0.15, 0.2};
        }
    }
    for (int x = 0; x < nx; ++x) {
        const double angle = 2 * std::acos(-1.0) * x / nx;
        ws.V_eff_r[x] = 0.3 * std::cos(angle) + 0.4 * std::sin(angle);
    }
    std::vector<double> reference;
    cpu::linalg::CplxMatrix eigenvectors;
    cpu::linalg::eigh_hermitian(exact, reference, eigenvectors);
    // One occupied band starts from four kinetic plane waves. The coupled
    // fifth direction is essential: the four-dimensional Ritz residual is
    // nonzero, so ng-1 must still be allowed to expand to the full space.
    DavidsonSolver solver(1, 1e-10, 10);
    solver.prepare(ws);
    solver.step(ws);
    EXPECT_NEAR(ws.eigenvalues[0], reference[0], 1e-10);
    double residual2 = 0;
    for (int i = 0; i < ng; ++i) {
        Complex value{};
        for (int j = 0; j < ng; ++j)
            value += exact(i, j) * ws.wfc(0, j);
        residual2 += std::norm(value - ws.eigenvalues[0] * ws.wfc(0, i));
    }
    EXPECT_LE(std::sqrt(residual2), 1e-10);
}
