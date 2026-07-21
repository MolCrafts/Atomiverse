/// @file tests/cpu/gpw/test_cpu_gpw_multigrid.cpp
/// @brief CP2K-style multigrid foundation: level construction, exponent->level
///        assignment, and the adjoint trilinear prolong/restrict transfer.
///
/// The load-bearing property is ADJOINTNESS: restrict must be the exact
/// transpose of prolong w.r.t. the grid integration inner products. That is what
/// keeps the integrated KS matrix the exact gradient of the collocated energy
/// across the level transfer, so a multigrid SCF stays variational/convergent.

#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/multigrid.h"

using ATV::cpu::gpw::assign_levels;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::make_multigrid;
using ATV::cpu::gpw::MultiGrid;
using ATV::cpu::gpw::prolong_trilinear;
using ATV::cpu::gpw::restrict_transpose;
using ATV::cpu::gpw::UniformGrid;

namespace {
GridBasisFunction s_gauss(double alpha, double cx, double cy, double cz) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = cx;
    p.center[1] = cy;
    p.center[2] = cz;
    p.coeff = 1.0;
    return {p};
}
} // namespace

// ngrids=1 -> a single grid at CUTOFF (the unchanged default path).
TEST(GpwMultigrid, SingleLevelIsCutoffGrid) {
    const double cell[3] = {10.0, 12.0, 14.0};
    MultiGrid mg = make_multigrid(cell, 320.0, 1, 50.0);
    ASSERT_EQ(mg.levels.size(), 1u);
    EXPECT_DOUBLE_EQ(mg.level_cutoff_ry[0], 320.0);
}

// ngrids>1 -> geometric ladder, finest=CUTOFF, coarsest~REL_CUTOFF.
TEST(GpwMultigrid, GeometricLadder) {
    const double cell[3] = {10.0, 12.0, 14.0};
    MultiGrid mg = make_multigrid(cell, 320.0, 5, 50.0);
    ASSERT_EQ(mg.levels.size(), 5u);
    EXPECT_DOUBLE_EQ(mg.level_cutoff_ry.front(), 320.0); // finest
    EXPECT_NEAR(mg.level_cutoff_ry.back(), 50.0, 1e-9); // coarsest ~ rel_cutoff
    for (std::size_t l = 1; l < mg.levels.size(); ++l)  // strictly coarsening
        EXPECT_LT(mg.level_cutoff_ry[l], mg.level_cutoff_ry[l - 1]);
    // Coarser levels have fewer points per axis (coarser spacing).
    EXPECT_LT(mg.levels.back().n[0], mg.levels.front().n[0]);
}

// Sharp Gaussians need the fine level; diffuse ones go coarse.
TEST(GpwMultigrid, ExponentLevelAssignment) {
    const double cell[3] = {10.0, 10.0, 10.0};
    MultiGrid mg = make_multigrid(cell, 320.0, 5, 50.0);
    std::vector<GridBasisFunction> basis = {
        s_gauss(20.0, 5, 5, 5), // sharp -> fine
        s_gauss(0.1, 5, 5, 5),  // diffuse -> coarse
    };
    auto lev = assign_levels(mg, basis, 50.0);
    EXPECT_EQ(lev[0], 0);      // sharp on the finest
    EXPECT_GT(lev[1], lev[0]); // diffuse on a coarser level
}

// THE property: restrict == transpose of prolong w.r.t. the grid inner products.
// <V_fine, prolong(g)>_fine == <restrict(V_fine), g>_coarse for random V,g.
TEST(GpwMultigrid, ProlongRestrictAreAdjoint) {
    const double cell[3] = {8.0, 9.0, 10.0};
    UniformGrid fine = ATV::cpu::gpw::make_uniform_grid(cell, 200.0);
    UniformGrid coarse = ATV::cpu::gpw::make_uniform_grid(cell, 50.0);

    // Deterministic pseudo-random fields.
    auto fill = [](long n, unsigned seed) {
        std::vector<double> v(static_cast<std::size_t>(n));
        unsigned s = seed;
        for (auto& x : v) {
            s = s * 1664525u + 1013904223u;
            x = (s >> 8) / 16777216.0 - 0.5;
        }
        return v;
    };
    std::vector<double> V = fill(fine.npts(), 7u);
    std::vector<double> g = fill(coarse.npts(), 13u);

    std::vector<double> Pg =
        prolong_trilinear(coarse, fine, g, /*periodic=*/true);
    std::vector<double> RV =
        restrict_transpose(fine, coarse, V, /*periodic=*/true);

    // Integration inner products (sum w * a * b).
    double lhs = 0.0; // <V, Pg>_fine
    for (long i = 0; i < fine.npts(); ++i)
        lhs += fine.weight() * V[i] * Pg[i];
    double rhs = 0.0; // <RV, g>_coarse
    for (long i = 0; i < coarse.npts(); ++i)
        rhs += coarse.weight() * RV[i] * g[i];

    EXPECT_NEAR(lhs, rhs, 1e-9 * (std::abs(lhs) + 1.0))
        << "restrict is not the exact adjoint of prolong";
}

// Prolongation of a constant field is exact (partition of unity).
TEST(GpwMultigrid, ProlongConstantIsExact) {
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid fine = ATV::cpu::gpw::make_uniform_grid(cell, 200.0);
    UniformGrid coarse = ATV::cpu::gpw::make_uniform_grid(cell, 50.0);
    std::vector<double> g(static_cast<std::size_t>(coarse.npts()), 2.5);
    auto Pg = prolong_trilinear(coarse, fine, g, true);
    for (double v : Pg)
        EXPECT_NEAR(v, 2.5, 1e-10);
}

#ifdef ATV_GPW_HAS_FFT
#include <cmath>
using ATV::cpu::gpw::prolong_fft;
using ATV::cpu::gpw::restrict_fft;

namespace {
std::vector<double> fill_rand(long n, unsigned seed) {
    std::vector<double> v(static_cast<std::size_t>(n));
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (s >> 8) / 16777216.0 - 0.5;
    }
    return v;
}
} // namespace

// THE load-bearing property for the FFT transfer: restrict_fft is the EXACT
// adjoint of prolong_fft w.r.t. the grid integration inner products.
// <V_fine, prolong_fft(g)>_fine == <restrict_fft(V_fine), g>_coarse.
TEST(GpwMultigridFFT, ProlongRestrictAreAdjoint) {
    const double cell[3] = {8.0, 9.0, 10.0};
    UniformGrid fine = ATV::cpu::gpw::make_uniform_grid(cell, 200.0);
    UniformGrid coarse = ATV::cpu::gpw::make_uniform_grid(cell, 50.0);

    std::vector<double> V = fill_rand(fine.npts(), 7u);
    std::vector<double> g = fill_rand(coarse.npts(), 13u);

    std::vector<double> Pg = prolong_fft(coarse, fine, g, /*periodic=*/true);
    std::vector<double> RV = restrict_fft(fine, coarse, V, /*periodic=*/true);

    double lhs = 0.0; // <V, Pg>_fine
    for (long i = 0; i < fine.npts(); ++i)
        lhs += fine.weight() * V[i] * Pg[i];
    double rhs = 0.0; // <RV, g>_coarse
    for (long i = 0; i < coarse.npts(); ++i)
        rhs += coarse.weight() * RV[i] * g[i];

    EXPECT_NEAR(lhs, rhs, 1e-9 * (std::abs(lhs) + 1.0))
        << "restrict_fft is not the exact adjoint of prolong_fft";
}

// Mass conservation: ∫prolong(f) dr == ∫f dr (spectral interpolation preserves
// the G=0 mode), i.e. sum(Pf)*w_fine == sum(f)*w_coarse.
TEST(GpwMultigridFFT, ProlongConservesMass) {
    const double cell[3] = {8.0, 9.0, 10.0};
    UniformGrid fine = ATV::cpu::gpw::make_uniform_grid(cell, 200.0);
    UniformGrid coarse = ATV::cpu::gpw::make_uniform_grid(cell, 50.0);

    std::vector<double> f = fill_rand(coarse.npts(), 21u);
    std::vector<double> Pf = prolong_fft(coarse, fine, f, /*periodic=*/true);

    double m_coarse = 0.0;
    for (double v : f)
        m_coarse += v;
    m_coarse *= coarse.weight();
    double m_fine = 0.0;
    for (double v : Pf)
        m_fine += v;
    m_fine *= fine.weight();

    EXPECT_NEAR(m_fine, m_coarse, 1e-9 * (std::abs(m_coarse) + 1.0))
        << "prolong_fft does not conserve mass";
}

// Exactness: a band-limited cosine resolvable on the coarse grid is reproduced
// EXACTLY at the fine grid points (this is what trilinear cannot do).
TEST(GpwMultigridFFT, ProlongLowFreqIsExact) {
    const double cell[3] = {8.0, 9.0, 10.0};
    UniformGrid fine = ATV::cpu::gpw::make_uniform_grid(cell, 200.0);
    UniformGrid coarse = ATV::cpu::gpw::make_uniform_grid(cell, 50.0);

    // A few low-frequency modes well below the coarse Nyquist along each axis.
    auto field = [&](const double r[3]) {
        const double kx = 2.0 * M_PI / cell[0];
        const double ky = 2.0 * M_PI / cell[1];
        const double kz = 2.0 * M_PI / cell[2];
        return 0.3 + 0.7 * std::cos(kx * r[0]) * std::cos(2.0 * ky * r[1])
               + 0.5 * std::sin(kz * r[2]);
    };
    std::vector<double> fc(static_cast<std::size_t>(coarse.npts()));
    for (long i = 0; i < coarse.npts(); ++i) {
        double r[3];
        coarse.point(i, r);
        fc[static_cast<std::size_t>(i)] = field(r);
    }
    std::vector<double> Pf = prolong_fft(coarse, fine, fc, /*periodic=*/true);
    double max_err = 0.0;
    for (long i = 0; i < fine.npts(); ++i) {
        double r[3];
        fine.point(i, r);
        max_err = std::max(
            max_err, std::abs(Pf[static_cast<std::size_t>(i)] - field(r)));
    }
    EXPECT_LT(max_err, 1e-8)
        << "band-limited field not reproduced at fine points";
}
#endif // ATV_GPW_HAS_FFT
