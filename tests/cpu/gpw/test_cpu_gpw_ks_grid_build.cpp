/// @file tests/cpu/gpw/test_cpu_gpw_ks_grid_build.cpp
/// @brief Unit tests for shared fine-grid KS build (Poisson + XC + V_eff).

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cpu::gpw::internal::ks_grid_build_uks;
using ATV::cpu::xc::XCType;

namespace {

UniformGrid tiny_grid() {
    UniformGrid g;
    g.n[0] = g.n[1] = g.n[2] = 4;
    g.L[0] = g.L[1] = g.L[2] = 8.0;
    return g;
}

} // namespace

// Uniform density + compensating n_ion → finite Hartree, non-empty V_eff.
TEST(KsGridBuild, RksUniformDensityProducesVeff) {
    const UniformGrid grid = tiny_grid();
    const std::size_t n = static_cast<std::size_t>(grid.npts());
    std::vector<double> rho(n, 0.02);
    std::vector<double> n_ion(n, 0.02); // neutral
    std::vector<double> v_loc;
    const auto out = ks_grid_build_rks(grid, rho, n_ion, v_loc, XCType::PADE,
                                       /*nonperiodic=*/false,
                                       /*vdw_df=*/false, /*vdw_df2=*/false,
                                       /*rvv10=*/false);
    ASSERT_EQ(out.veff.size(), n);
    EXPECT_TRUE(std::isfinite(out.e_xc));
    EXPECT_TRUE(std::isfinite(out.e_hartree));
    EXPECT_DOUBLE_EQ(out.e_locgrid, 0.0);
    EXPECT_DOUBLE_EQ(out.e_nl_vdw, 0.0);
    for (double v : out.veff)
        EXPECT_TRUE(std::isfinite(v));
}

// Dual-spin build: α and β V_eff same length; total ρ feeds Hartree.
TEST(KsGridBuild, UksDualVeffSameLength) {
    const UniformGrid grid = tiny_grid();
    const std::size_t n = static_cast<std::size_t>(grid.npts());
    std::vector<double> rho_a(n, 0.012);
    std::vector<double> rho_b(n, 0.008);
    std::vector<double> n_ion(n, 0.02);
    std::vector<double> v_loc(n, -0.1);
    const auto out =
        ks_grid_build_uks(grid, rho_a, rho_b, n_ion, v_loc, XCType::PADE,
                          /*nonperiodic=*/false, false, false, false);
    ASSERT_EQ(out.veff_a.size(), n);
    ASSERT_EQ(out.veff_b.size(), n);
    EXPECT_LT(out.e_locgrid, 0.0); // ρ·V_loc with negative V_loc
    for (std::size_t g = 0; g < n; ++g) {
        EXPECT_TRUE(std::isfinite(out.veff_a[g]));
        EXPECT_TRUE(std::isfinite(out.veff_b[g]));
    }
}

// A nonuniform +/- potential stresses cancellation in the energy integral.
// Changing OMP width must not change which additions define that integral.
TEST(KsGridBuild, UksLocalEnergyHasFixedReductionOrder) {
    UniformGrid grid;
    for (int d = 0; d < 3; ++d) {
        grid.n[d] = 18;
        grid.L[d] = 8;
    }
    const auto n = static_cast<std::size_t>(grid.npts());
    std::vector<double> ra(n), rb(n), ion(n), vloc(n);
    long double expected = 0;
    for (std::size_t g = 0; g < n; ++g) {
        ra[g] = 0.012 + 0.002 * std::sin(double(g));
        rb[g] = 0.008 + 0.001 * std::cos(double(g));
        ion[g] = ra[g] + rb[g];
        vloc[g] =
            (g % 3 == 0 ? -1.0 : 0.5) * (1 + 0.1 * std::sin(2 * double(g)));
        expected += (static_cast<long double>(ra[g]) + rb[g]) * vloc[g];
    }
    expected *= grid.weight();
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    struct RestoreThreads {
        int count;
        ~RestoreThreads() { omp_set_num_threads(count); }
    } restore{saved_threads};
    omp_set_num_threads(1);
#endif
    const auto serial = ks_grid_build_uks(grid, ra, rb, ion, vloc, XCType::PADE,
                                          false, false, false, false);
    EXPECT_NEAR(serial.e_locgrid, static_cast<double>(expected), 1e-13);
#ifdef _OPENMP
    for (int threads : {2, 4}) {
        omp_set_num_threads(threads);
        const auto parallel = ks_grid_build_uks(
            grid, ra, rb, ion, vloc, XCType::PADE, false, false, false, false);
        EXPECT_DOUBLE_EQ(parallel.e_locgrid, serial.e_locgrid);
    }
#endif
}
