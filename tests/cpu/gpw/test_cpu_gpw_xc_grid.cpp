/// @file tests/cpu/gpw/test_cpu_gpw_xc_grid.cpp
/// @brief GPW XC-on-grid: PADE energy/potential from a collocated density, and
///        the KS-matrix XC block via integrate_potential.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/xc/pade.h"

using ATV::cpu::gpw::collocate_density;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::integrate_potential;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::xc_grid_pade;
using ATV::cpu::gpw::XcOnGrid;

namespace {
GridBasisFunction s_function(double alpha, const double c[3]) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = c[0];
    p.center[1] = c[1];
    p.center[2] = c[2];
    p.coeff = 1.0;
    return {p};
}
} // namespace

// Constant density over the box: E_xc = Volume * rho0 * exc(rho0/2, rho0/2),
// and the grid potential equals the pointwise PADE potential everywhere.
TEST(GpwXcGrid, ConstantDensityEnergyAndPotential) {
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);

    const double rho0 = 0.15;
    std::vector<double> rho_r(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid xc = xc_grid_pade(grid, rho_r);

    const ATV::cpu::xc::PadeXC ref =
        ATV::cpu::xc::pade_lsd(0.5 * rho0, 0.5 * rho0);
    const double volume = cell[0] * cell[1] * cell[2];
    const double e_expected = volume * rho0 * ref.exc;

    EXPECT_NEAR(xc.e_xc, e_expected, 1e-10 * std::abs(e_expected));
    ASSERT_EQ(static_cast<long>(xc.vxc_r.size()), grid.npts());
    EXPECT_DOUBLE_EQ(xc.vxc_r[0], ref.vrho_a);
    EXPECT_DOUBLE_EQ(xc.vxc_r[grid.npts() / 2], ref.vrho_a);
}

// From a real collocated density: energy equals the explicit weighted sum, and
// the KS XC block is finite and symmetric.
TEST(GpwXcGrid, CollocatedDensityComposesToSymmetricBlock) {
    const double cell[3] = {12.0, 12.0, 12.0};
    UniformGrid grid = make_uniform_grid(cell, 300.0);

    const double cA[3] = {5.5, 6.0, 6.0};
    const double cB[3] = {6.5, 6.0, 6.0};
    std::vector<GridBasisFunction> basis = {s_function(0.9, cA),
                                            s_function(1.1, cB)};
    std::vector<double> P = {1.0, 0.2, 0.2, 0.8};

    std::vector<double> rho = collocate_density(grid, basis, P);
    XcOnGrid xc = xc_grid_pade(grid, rho);

    // Independent recomputation of the energy from the same density.
    double e_manual = 0.0;
    const double w = grid.weight();
    for (double r : rho) {
        if (r > 1e-12)
            e_manual += w * r * ATV::cpu::xc::pade_lsd(0.5 * r, 0.5 * r).exc;
    }
    EXPECT_NEAR(xc.e_xc, e_manual, 1e-12 * std::max(1.0, std::abs(e_manual)));
    EXPECT_LT(xc.e_xc, 0.0); // XC energy is negative for a real density

    // KS XC block V_uv = integral phi_u Vxc phi_v: symmetric and finite.
    std::vector<double> Vxc = integrate_potential(grid, basis, xc.vxc_r);
    ASSERT_EQ(Vxc.size(), 4u);
    EXPECT_NEAR(Vxc[1], Vxc[2], 1e-12 * std::max(1.0, std::abs(Vxc[1])));
    for (double v : Vxc)
        EXPECT_TRUE(std::isfinite(v));
}
