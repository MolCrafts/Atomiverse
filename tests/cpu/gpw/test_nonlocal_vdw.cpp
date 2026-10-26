/// @brief Nonlocal RPS functional: demand, derivative and parallel identities
///        on a tiny analytic periodic density. No SCF or external process.
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/vdw_df.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using namespace ATV::cpu::gpw;
struct Width {
#ifdef _OPENMP
    int saved = omp_get_max_threads();
    ~Width() { omp_set_num_threads(saved); }
    void set(int width) { omp_set_num_threads(width); }
#else
    void set(int) {}
#endif
};

TEST(GpwNonlocalVdw, EnergyDemandAndThreadWidthsAgreeOnPeriodicDensity) {
    Width width;
    UniformGrid grid;
    for (int d = 0; d < 3; ++d) {
        grid.n[d] = 6;
        grid.L[d] = 6.0;
    }
    std::vector<double> rho(grid.npts());
    for (long g = 0; g < grid.npts(); ++g) {
        double r[3];
        grid.point(g, r);
        rho[g] = 0.07 + 0.005 * std::cos(2 * M_PI * r[0] / 6.0)
                 + 0.003 * std::sin(2 * M_PI * r[1] / 6.0);
    }
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        width.set(1);
        const auto energy = nonlocal_vdw_on_grid(grid, rho, kind == 0,
                                                 kind == 1, kind == 2, false);
        const auto first = nonlocal_vdw_on_grid(grid, rho, kind == 0, kind == 1,
                                                kind == 2, true);
        EXPECT_TRUE(energy.v_r.empty());
        ASSERT_EQ(first.v_r.size(), rho.size());
        EXPECT_TRUE(std::isfinite(first.energy));
        EXPECT_GT(std::abs(first.energy), 1e-8);
        EXPECT_DOUBLE_EQ(energy.energy, first.energy);
        width.set(8);
        const auto parallel = nonlocal_vdw_on_grid(grid, rho, kind == 0,
                                                   kind == 1, kind == 2, true);
        EXPECT_DOUBLE_EQ(parallel.energy, first.energy);
        for (std::size_t g = 0; g < rho.size(); ++g)
            EXPECT_DOUBLE_EQ(parallel.v_r[g], first.v_r[g])
                << "grid point " << g;
    }
}

TEST(GpwNonlocalVdw, PotentialIsEnergyDirectionalDerivative) {
    Width width;
    width.set(1);
    UniformGrid grid;
    for (int d = 0; d < 3; ++d) {
        grid.n[d] = 6;
        grid.L[d] = 6.0;
    }
    std::vector<double> rho(grid.npts()), direction(grid.npts());
    constexpr double h = 1e-6;
    for (long g = 0; g < grid.npts(); ++g) {
        double r[3];
        grid.point(g, r);
        rho[g] = 0.07 + 0.005 * std::cos(2 * M_PI * r[0] / 6.0)
                 + 0.003 * std::sin(2 * M_PI * r[1] / 6.0);
        direction[g] = 0.1 + 0.3 * std::cos(2 * M_PI * r[0] / 6.0)
                       + 0.2 * std::sin(2 * M_PI * r[2] / 6.0);
    }
    auto plus = rho, minus = rho;
    for (std::size_t g = 0; g < rho.size(); ++g) {
        plus[g] += h * direction[g];
        minus[g] -= h * direction[g];
    }
    for (int kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        const auto first = nonlocal_vdw_on_grid(grid, rho, kind == 0, kind == 1,
                                                kind == 2, true);
        const double ep = nonlocal_vdw_on_grid(grid, plus, kind == 0, kind == 1,
                                               kind == 2, false)
                              .energy;
        const double em = nonlocal_vdw_on_grid(grid, minus, kind == 0,
                                               kind == 1, kind == 2, false)
                              .energy;
        double derivative = 0;
        for (std::size_t g = 0; g < rho.size(); ++g)
            derivative += first.v_r[g] * direction[g] * grid.weight();
        EXPECT_NEAR(derivative, (ep - em) / (2 * h), 2e-6);
    }
}
} // namespace
