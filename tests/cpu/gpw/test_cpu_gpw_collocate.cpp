/// @file tests/cpu/gpw/test_cpu_gpw_collocate.cpp
/// @brief GPW D2: uniform-grid collocation / integration round trips.
///
/// Validates the two GPW bridge identities without FFT or SCF:
///   (1) integrate_potential(V≡1) == overlap matrix S (analytic, s-Gaussians),
///   (2) ∫ ρ(r) dr == Tr(P S) for a collocated density.
/// s-type Gaussians give a closed-form overlap; the midpoint rule on a uniform
/// grid converges super-algebraically because the Gaussians decay to ~0 well
/// inside the box, so a tight tolerance is appropriate.

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"

using ATV::cpu::gpw::collocate_density;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::integrate_potential;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;

namespace {

// Closed-form overlap of two s-type primitive Gaussians.
double s_overlap(double a, const double A[3], double b, const double B[3]) {
    const double p = a + b;
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k)
        d2 += (A[k] - B[k]) * (A[k] - B[k]);
    return std::pow(M_PI / p, 1.5) * std::exp(-a * b / p * d2);
}

GridBasisFunction s_function(double alpha, const double c[3]) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = c[0];
    p.center[1] = c[1];
    p.center[2] = c[2];
    p.l[0] = p.l[1] = p.l[2] = 0;
    p.coeff = 1.0;
    return {p};
}

} // namespace

TEST(GpwCollocate, IntegratePotentialOneRecoversOverlap) {
    const double cell[3] = {12.0, 12.0, 12.0};
    UniformGrid grid = make_uniform_grid(cell, 400.0);

    const double cA[3] = {5.0, 6.0, 6.0};
    const double cB[3] = {7.0, 6.0, 6.0};
    const double aA = 0.8, aB = 1.2;
    std::vector<GridBasisFunction> basis = {s_function(aA, cA),
                                            s_function(aB, cB)};

    std::vector<double> V1(static_cast<std::size_t>(grid.npts()), 1.0);
    std::vector<double> S = integrate_potential(grid, basis, V1);

    const double S00 = s_overlap(aA, cA, aA, cA);
    const double S01 = s_overlap(aA, cA, aB, cB);
    const double S11 = s_overlap(aB, cB, aB, cB);

    EXPECT_NEAR(S[0], S00, 1e-4 * S00);
    EXPECT_NEAR(S[1], S01, 1e-4 * std::max(1e-3, S01));
    EXPECT_NEAR(S[2], S01, 1e-4 * std::max(1e-3, S01)); // symmetry
    EXPECT_NEAR(S[3], S11, 1e-4 * S11);
}

TEST(GpwCollocate, DensityIntegratesToTracePS) {
    const double cell[3] = {12.0, 12.0, 12.0};
    UniformGrid grid = make_uniform_grid(cell, 400.0);

    const double cA[3] = {5.0, 6.0, 6.0};
    const double cB[3] = {7.0, 6.0, 6.0};
    const double aA = 0.8, aB = 1.2;
    std::vector<GridBasisFunction> basis = {s_function(aA, cA),
                                            s_function(aB, cB)};

    // Symmetric density matrix.
    std::vector<double> P = {1.0, 0.3, 0.3, 0.7};

    std::vector<double> rho = collocate_density(grid, basis, P);
    double integral = 0.0;
    for (double v : rho)
        integral += v;
    integral *= grid.weight();

    const double S00 = s_overlap(aA, cA, aA, cA);
    const double S01 = s_overlap(aA, cA, aB, cB);
    const double S11 = s_overlap(aB, cB, aB, cB);
    // Tr(P S) with symmetric P, S: P00 S00 + 2 P01 S01 + P11 S11.
    const double trace_ps = P[0] * S00 + 2.0 * P[1] * S01 + P[3] * S11;

    EXPECT_NEAR(integral, trace_ps, 1e-4 * std::abs(trace_ps));
}

TEST(GpwCollocate, GridSizingFromCutoff) {
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid g = make_uniform_grid(cell, 400.0);
    // n = ceil(L*sqrt(cutoff)/pi) rounded up to even; L=10, sqrt(400)=20.
    const int expected = 64; // ceil(10*20/pi)=ceil(63.66)=64 (already even)
    EXPECT_EQ(g.n[0], expected);
    EXPECT_EQ(g.n[1], expected);
    EXPECT_EQ(g.n[2], expected);
    EXPECT_GT(g.weight(), 0.0);
}
