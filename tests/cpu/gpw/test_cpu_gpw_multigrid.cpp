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

// ── Bloch image-cell product machinery (multi-k multigrid) ──────────────────

#include <array>
#include <complex>

#include "atomiverse/cpu/gpw/internal/kpoints.h"

using ATV::cpu::gpw::build_products;
using ATV::cpu::gpw::collocate_products;
using ATV::cpu::gpw::collocate_products_cells;
using ATV::cpu::gpw::integrate_products;
using ATV::cpu::gpw::integrate_products_cells;
using ATV::cpu::gpw::ProductSet;

namespace {

// Two s Gaussians in a small periodic cell: products reach several image
// cells (the multi-k phase regime) while windows stay unit-test sized.
// Combined exponents p ∈ {0.7, 1.15, 1.6} split across a (60, 12) Ry
// two-level ladder: need = rel·p puts the diffuse 0.7 pair on the coarse
// 12 Ry level and the rest on the fine level — both levels populated.
std::vector<GridBasisFunction> cells_toy_basis(double L) {
    GridPrimitive p1;
    p1.alpha = 0.35;
    p1.center[0] = 0.25 * L;
    p1.center[1] = 0.40 * L;
    p1.center[2] = 0.30 * L;
    p1.coeff = 1.0;
    GridPrimitive p2 = p1;
    p2.alpha = 0.80;
    p2.center[0] = 0.70 * L;
    p2.center[1] = 0.55 * L;
    p2.center[2] = 0.65 * L;
    p2.coeff = 0.8;
    return {GridBasisFunction{p1}, GridBasisFunction{p2}};
}

std::vector<double> cells_rand(std::size_t n, unsigned seed) {
    std::vector<double> v(n);
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (s >> 8) / 16777216.0 - 0.5;
    }
    return v;
}

} // namespace

// The image-cell index is a faithful inverse map: cells[] is sorted + unique,
// every product's tcell points at its own ket shift, and a diffuse periodic
// basis populates more than the home cell.
TEST(GpwMultigridCells, ImageCellIndexIsConsistent) {
    const double cell[3] = {6.0, 6.0, 6.0};
    MultiGrid mg = make_multigrid(cell, 60.0, 2, 12.0);
    const auto basis = cells_toy_basis(cell[0]);
    const ProductSet ps =
        build_products(mg, basis, cell, 12.0, 1e-12, /*periodic=*/true);

    ASSERT_GT(ps.cells.size(), 1u) << "diffuse periodic products stayed home";
    for (std::size_t c = 1; c < ps.cells.size(); ++c)
        EXPECT_LT(ps.cells[c - 1], ps.cells[c]) << "cells not sorted/unique";
    for (const auto& lev : ps.by_level)
        for (const auto& prod : lev) {
            ASSERT_GE(prod.tcell, 0);
            ASSERT_LT(static_cast<std::size_t>(prod.tcell), ps.cells.size());
            const auto& t = ps.cells[static_cast<std::size_t>(prod.tcell)];
            EXPECT_EQ(t[0], prod.t[0]);
            EXPECT_EQ(t[1], prod.t[1]);
            EXPECT_EQ(t[2], prod.t[2]);
        }
}

// Γ reduction: a uniform per-cell stack (every D[c] the same P) reproduces the
// shipped single-density collocation and the cell-summed integration exactly.
TEST(GpwMultigridCells, UniformStackReducesToGamma) {
    const double cell[3] = {6.0, 6.0, 6.0};
    MultiGrid mg = make_multigrid(cell, 60.0, 2, 12.0);
    const auto basis = cells_toy_basis(cell[0]);
    const int nb = static_cast<int>(basis.size());
    const ProductSet ps =
        build_products(mg, basis, cell, 12.0, 1e-12, /*periodic=*/true);

    std::vector<double> P = cells_rand(static_cast<std::size_t>(nb) * nb, 3u);
    // Symmetrize (a density matrix weight).
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < i; ++j) {
            const double s = 0.5
                             * (P[static_cast<std::size_t>(i) * nb + j]
                                + P[static_cast<std::size_t>(j) * nb + i]);
            P[static_cast<std::size_t>(i) * nb + j] = s;
            P[static_cast<std::size_t>(j) * nb + i] = s;
        }
    const std::vector<std::vector<double>> D(ps.cells.size(), P);

    // Both levels must actually hold products, or the loop tests nothing.
    ASSERT_EQ(mg.levels.size(), 2u);
    ASSERT_GT(ps.by_level[0].size(), 0u);
    ASSERT_GT(ps.by_level[1].size(), 0u);

    for (std::size_t l = 0; l < mg.levels.size(); ++l) {
        const auto rho_gamma = collocate_products(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, P, 1e-12);
        const auto rho_cells = collocate_products_cells(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, D, 1e-12);
        ASSERT_EQ(rho_gamma.size(), rho_cells.size());
        // Uniform D-stack must match the Γ product path mathematically; FP
        // reduction order across the two kernels can differ by a few ulps, so
        // match the integrate half of this test (1e-12 relative), not
        // bit-identical equality.
        for (std::size_t g = 0; g < rho_gamma.size(); ++g)
            ASSERT_NEAR(rho_gamma[g], rho_cells[g],
                        1e-12 * (std::abs(rho_gamma[g]) + 1.0));

        const auto V_r =
            cells_rand(static_cast<std::size_t>(mg.levels[l].npts()), 11u);
        const auto V_gamma = integrate_products(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, V_r, 1e-12);
        const auto V_cells = integrate_products_cells(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, V_r, ps.cells.size(), 1e-12);
        for (std::size_t e = 0; e < V_gamma.size(); ++e) {
            double s = 0.0;
            for (const auto& Vc : V_cells)
                s += Vc[e];
            EXPECT_NEAR(s, V_gamma[e], 1e-12 * (std::abs(V_gamma[e]) + 1.0));
        }
    }
}

// THE property: the per-cell integration is the exact transpose of the
// per-cell collocation — Σ_g w·V(g)·ρ_D(g) == Σ_c <D_c, V_t_c>_F for an
// arbitrary per-cell stack D (what keeps the multi-k multigrid SCF
// variational).
TEST(GpwMultigridCells, CellsIntegrateIsAdjointOfCellsCollocate) {
    const double cell[3] = {6.0, 6.0, 6.0};
    MultiGrid mg = make_multigrid(cell, 60.0, 2, 12.0);
    const auto basis = cells_toy_basis(cell[0]);
    const int nb = static_cast<int>(basis.size());
    const ProductSet ps =
        build_products(mg, basis, cell, 12.0, 1e-12, /*periodic=*/true);
    const std::size_t nn = static_cast<std::size_t>(nb) * nb;

    std::vector<std::vector<double>> D;
    for (std::size_t c = 0; c < ps.cells.size(); ++c)
        D.push_back(cells_rand(nn, 17u + static_cast<unsigned>(c)));

    double lhs = 0.0, rhs = 0.0;
    for (std::size_t l = 0; l < mg.levels.size(); ++l) {
        const auto V_r =
            cells_rand(static_cast<std::size_t>(mg.levels[l].npts()), 23u);
        const auto rho = collocate_products_cells(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, D, 1e-12);
        for (std::size_t g = 0; g < rho.size(); ++g)
            lhs += mg.levels[l].weight() * V_r[g] * rho[g];
        const auto Vt = integrate_products_cells(
            mg.levels[l], ps.by_level[l], ps.tasks_by_level[l], ps.prims,
            ps.contribs, nb, V_r, ps.cells.size(), 1e-12);
        for (std::size_t c = 0; c < ps.cells.size(); ++c)
            for (std::size_t e = 0; e < nn; ++e)
                rhs += D[c][e] * Vt[c][e];
    }
    EXPECT_NEAR(lhs, rhs, 1e-8 * (std::abs(lhs) + 1.0))
        << "per-cell integrate is not the transpose of per-cell collocate";
}

// Time-reversal folding is exact for the D stack: a conjugate ±κ pair at
// half weight and the folded single κ at full weight build the SAME real
// D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}] when P^{−k} = conj(P^k) — the mechanism
// behind the machine-exact folded-vs-full multi-k multigrid SCF energies
// (regressions/gpw/gpw-multigrid-kpoints REFERENCE.md).
TEST(GpwMultigridCells, TimeReversalDStackFoldsExactly) {
    using Cplx = std::complex<double>;
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const double kappa[3] = {0.25, -0.25, 0.125};
    const int nb = 2;
    // Hermitian P at +κ; time reversal gives P(−κ) = conj(P(+κ)).
    Cplx P[4] = {Cplx(0.9, 0.0), Cplx(0.3, 0.2), Cplx(0.3, -0.2),
                 Cplx(0.4, 0.0)};
    const int cells[3][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 2, 1}};
    for (const auto& t : cells) {
        const double arg =
            -kTwoPi * (kappa[0] * t[0] + kappa[1] * t[1] + kappa[2] * t[2]);
        const Cplx ph(std::cos(arg), std::sin(arg));
        for (int e = 0; e < nb * nb; ++e) {
            // Full mesh: +κ and −κ, weight 1/2 each (conj phase, conj P).
            const double d_full = 0.5 * (P[e] * ph).real()
                                  + 0.5 * (std::conj(P[e]) * std::conj(ph)).real();
            // Folded: +κ only, weight 1.
            const double d_folded = (P[e] * ph).real();
            ASSERT_DOUBLE_EQ(d_full, d_folded);
        }
    }
}

// Phase conventions against the independent pre-summed χ^k machinery: on one
// shared grid, the D-stack product density matches collocate_density_k and the
// phase-summed V_t stack matches integrate_potential_k (Hermitian per k).
TEST(GpwMultigridCells, MatchesKpointBasisGridPath) {
    using ATV::cpu::gpw::internal::CplxMatrix;
    using ATV::cpu::gpw::internal::KPoint;
    using Cplx = std::complex<double>;
    constexpr double kTwoPi = 6.283185307179586476925286766559;

    const double cell[3] = {6.0, 6.0, 6.0};
    // Single level: products and χ^k live on the same grid.
    MultiGrid mg = make_multigrid(cell, 60.0, 1, 0.0);
    const UniformGrid& grid = mg.levels[0];
    const auto basis = cells_toy_basis(cell[0]);
    const int nb = static_cast<int>(basis.size());
    const ProductSet ps =
        build_products(mg, basis, cell, 60.0, 1e-12, /*periodic=*/true);

    std::vector<KPoint> kpts(1);
    kpts[0].frac[0] = 0.25;
    kpts[0].frac[1] = -0.5;
    kpts[0].frac[2] = 0.125;
    kpts[0].weight = 1.0;
    const auto kbg = ATV::cpu::gpw::internal::build_kpoint_basis_grid(
        grid, basis, kpts, 1e-12);

    // Hermitian P_k with nonzero imaginary off-diagonals.
    CplxMatrix P = xt::zeros<Cplx>({nb, nb});
    P(0, 0) = 0.9;
    P(1, 1) = 0.4;
    P(0, 1) = Cplx(0.3, 0.2);
    P(1, 0) = std::conj(P(0, 1));

    // Per-cell phases e^{−2πi κ·t} and the real D stack.
    std::vector<Cplx> phase(ps.cells.size());
    std::vector<std::vector<double>> D(
        ps.cells.size(), std::vector<double>(static_cast<std::size_t>(nb) * nb));
    for (std::size_t c = 0; c < ps.cells.size(); ++c) {
        const auto& t = ps.cells[c];
        const double arg = -kTwoPi
                           * (kpts[0].frac[0] * t[0] + kpts[0].frac[1] * t[1]
                              + kpts[0].frac[2] * t[2]);
        phase[c] = Cplx(std::cos(arg), std::sin(arg));
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                D[c][static_cast<std::size_t>(i) * nb + j] =
                    (P(i, j) * phase[c]).real();
    }

    // Density: product path vs pre-summed χ^k path.
    const auto rho_prod = collocate_products_cells(
        grid, ps.by_level[0], ps.tasks_by_level[0], ps.prims, ps.contribs, nb,
        D, 1e-12);
    std::vector<double> rho_kbg(static_cast<std::size_t>(grid.npts()), 0.0);
    ATV::cpu::gpw::internal::collocate_density_k(kbg, 0, P, 1.0, rho_kbg);
    double rho_max = 0.0, rho_err = 0.0;
    for (std::size_t g = 0; g < rho_prod.size(); ++g) {
        rho_max = std::max(rho_max, std::abs(rho_kbg[g]));
        rho_err = std::max(rho_err, std::abs(rho_prod[g] - rho_kbg[g]));
    }
    EXPECT_LT(rho_err, 1e-7 * (rho_max + 1.0))
        << "product D-stack density disagrees with collocate_density_k";

    // KS: phase-summed V_t vs integrate_potential_k; Hermiticity.
    const auto V_r = cells_rand(static_cast<std::size_t>(grid.npts()), 29u);
    const auto Vt = integrate_products_cells(
        grid, ps.by_level[0], ps.tasks_by_level[0], ps.prims, ps.contribs, nb,
        V_r, ps.cells.size(), 1e-12);
    CplxMatrix Vk = xt::zeros<Cplx>({nb, nb});
    for (std::size_t c = 0; c < ps.cells.size(); ++c)
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                Vk(i, j) += std::conj(phase[c])
                            * Vt[c][static_cast<std::size_t>(i) * nb + j];
    const CplxMatrix Vk_ref =
        ATV::cpu::gpw::internal::integrate_potential_k(kbg, 0, V_r);
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j) {
            EXPECT_NEAR(Vk(i, j).real(), Vk_ref(i, j).real(),
                        1e-7 * (std::abs(Vk_ref(i, j)) + 1.0));
            EXPECT_NEAR(Vk(i, j).imag(), Vk_ref(i, j).imag(),
                        1e-7 * (std::abs(Vk_ref(i, j)) + 1.0));
            EXPECT_NEAR(Vk(i, j).real(), Vk(j, i).real(),
                        1e-9 * (std::abs(Vk(i, j)) + 1.0));
            EXPECT_NEAR(Vk(i, j).imag(), -Vk(j, i).imag(),
                        1e-9 * (std::abs(Vk(i, j)) + 1.0));
        }
}
