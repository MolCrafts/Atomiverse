/// @file tests/cpu/gpw/test_cpu_gpw_ks_grid_build.cpp
/// @brief Unit tests for shared fine-grid KS build (Poisson + XC + V_eff).

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"

using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::internal::ks_grid_build_rks;
using ATV::cpu::gpw::internal::ks_grid_build_uks;

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
    const auto out = ks_grid_build_rks(grid, rho, n_ion, v_loc, XcKind::PADE,
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
        ks_grid_build_uks(grid, rho_a, rho_b, n_ion, v_loc, XcKind::PADE,
                          /*nonperiodic=*/false, false, false, false);
    ASSERT_EQ(out.veff_a.size(), n);
    ASSERT_EQ(out.veff_b.size(), n);
    EXPECT_LT(out.e_locgrid, 0.0); // ρ·V_loc with negative V_loc
    for (std::size_t g = 0; g < n; ++g) {
        EXPECT_TRUE(std::isfinite(out.veff_a[g]));
        EXPECT_TRUE(std::isfinite(out.veff_b[g]));
    }
}
