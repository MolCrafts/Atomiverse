/// @brief Complex Bloch-grid density/KS contractions against finite dense
///        algebra, including exactly-zero support and thread reproducibility.
#include <complex>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/gpw/internal/kpoints.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using namespace ATV::cpu::gpw::internal;
using Cplx = std::complex<double>;

// Restore the calling test process's width even on an assertion failure.
struct OmpWidth {
#ifdef _OPENMP
    int saved = omp_get_max_threads();
    ~OmpWidth() { omp_set_num_threads(saved); }
    void set(int width) { omp_set_num_threads(width); }
#else
    void set(int) {}
#endif
};

TEST(GpwKpointGridContractions, SparseComplexRowsMatchDenseAlgebraAndThreads) {
    OmpWidth width;
    KpointBasisGrid grid;
    grid.nb = 4;
    grid.npts = 257; // most grid rows are exactly zero
    grid.weight = 0.125;
    grid.chi.assign(1, std::vector<Cplx>(grid.npts * grid.nb));
    for (long g = 0; g < grid.npts; ++g) {
        Cplx* v = grid.chi[0].data() + g * grid.nb;
        if (g % 17 == 0) {
            v[0] = {0.5, 0.25};
            v[2] = {-0.25, 0.75};
        }
        if (g % 31 == 0) {
            v[1] = {0.25, -0.5};
            v[3] = {0.125, 0.375};
        }
    }
    CplxMatrix P = {
        {Cplx(1.5), Cplx(.125, .25), Cplx(-.0625, .125), Cplx(.125, -.0625)},
        {Cplx(.125, -.25), Cplx(.75), Cplx(.125, -.125), Cplx(-.0625, .125)},
        {Cplx(-.0625, -.125), Cplx(.125, .125), Cplx(1.25), Cplx(.0625, .125)},
        {Cplx(.125, .0625), Cplx(-.0625, -.125), Cplx(.0625, -.125), Cplx(.5)}};
    std::vector<double> potential(grid.npts), initial(grid.npts);
    for (long g = 0; g < grid.npts; ++g) {
        potential[g] = g % 11 == 0 ? 0.0 : (static_cast<int>(g % 9) - 4) / 16.0;
        initial[g] = (g % 7) / 8.0;
    }
    constexpr double wk = 0.25;
    auto expected_rho = initial;
    CplxMatrix expected_V = xt::zeros<Cplx>({4, 4});
    // Independent dense definitions: rho = wk χ^T P conj(χ),
    // V_uv = Σ_g weight V(g) conj(χ_u(g)) χ_v(g).
    // Values are binary fractions so this small algebra also admits exact
    // equality across alternative chunk groupings.
    for (long g = 0; g < grid.npts; ++g) {
        const Cplx* v = grid.chi[0].data() + g * grid.nb;
        Cplx density{0, 0};
        for (int u = 0; u < grid.nb; ++u)
            for (int t = 0; t < grid.nb; ++t)
                density += v[u] * P(u, t) * std::conj(v[t]);
        expected_rho[g] += wk * density.real();
        for (int u = 0; u < grid.nb; ++u)
            for (int t = 0; t < grid.nb; ++t)
                expected_V(u, t) +=
                    grid.weight * potential[g] * std::conj(v[u]) * v[t];
    }
    width.set(1);
    auto rho_one = initial;
    collocate_density_k(grid, 0, P, wk, rho_one);
    const auto V_one = integrate_potential_k(grid, 0, potential);
    width.set(8);
    auto rho_eight = initial;
    collocate_density_k(grid, 0, P, wk, rho_eight);
    const auto V_eight = integrate_potential_k(grid, 0, potential);
    for (long g = 0; g < grid.npts; ++g) {
        EXPECT_NEAR(rho_one[g], expected_rho[g], 2e-14) << "grid point " << g;
        EXPECT_DOUBLE_EQ(rho_one[g], rho_eight[g]) << "grid point " << g;
    }
    for (int u = 0; u < grid.nb; ++u)
        for (int t = 0; t < grid.nb; ++t) {
            EXPECT_NEAR(std::abs(V_one(u, t) - expected_V(u, t)), 0.0, 2e-14);
            EXPECT_EQ(V_one(u, t), V_eight(u, t));
        }
}

TEST(GpwKpointGridContractions, ExactZeroCasesAndTinyNonzeroSupport) {
    KpointBasisGrid grid;
    grid.nb = 2;
    grid.npts = 7;
    grid.weight = 0.5;
    grid.chi.assign(1, std::vector<Cplx>(14));
    grid.chi[0][3 * 2 + 1] = {1e-12, 0.0}; // must not become an epsilon screen
    const CplxMatrix P = {{Cplx(1), Cplx(0)}, {Cplx(0), Cplx(1)}};
    std::vector<double> rho(7, 0.0);
    collocate_density_k(grid, 0, P, 1.0, rho);
    EXPECT_GT(rho[3], 0.0);
    EXPECT_NEAR(rho[3], 1e-24, 2e-39);
    for (int g = 0; g < 7; ++g)
        if (g != 3)
            EXPECT_DOUBLE_EQ(rho[g], 0.0);
    const auto before = rho;
    collocate_density_k(grid, 0, P, 0.0, rho);
    EXPECT_EQ(rho, before);
    std::vector<double> potential(7, 0.0);
    auto V = integrate_potential_k(grid, 0, potential);
    for (const auto x : V)
        EXPECT_EQ(x, Cplx(0));
    potential[3] = 2.0;
    V = integrate_potential_k(grid, 0, potential);
    EXPECT_GT(V(1, 1).real(), 0.0);
    EXPECT_NEAR(V(1, 1).real(), 1e-24, 2e-39);
    EXPECT_DOUBLE_EQ(V(1, 1).imag(), 0.0);
    EXPECT_EQ(V(0, 0), Cplx(0));
    EXPECT_EQ(V(0, 1), Cplx(0));
    EXPECT_EQ(V(1, 0), Cplx(0));
    grid.weight = 0;
    V = integrate_potential_k(grid, 0, potential);
    for (const auto x : V)
        EXPECT_EQ(x, Cplx(0));
}
} // namespace
