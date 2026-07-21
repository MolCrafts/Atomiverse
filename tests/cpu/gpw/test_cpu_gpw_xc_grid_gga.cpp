/// @file tests/cpu/gpw/test_cpu_gpw_xc_grid_gga.cpp
/// @brief PBE GGA XC on the grid: LDA-limit energy + functional-derivative check.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::xc_grid_pbe;
using ATV::cpu::gpw::XcOnGrid;

namespace {
std::vector<double> gaussian_density(const UniformGrid& grid, const double c[3],
                                     double alpha, double amp) {
    const long ng = grid.npts();
    std::vector<double> rho(static_cast<std::size_t>(ng));
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        const double dx = r[0] - c[0], dy = r[1] - c[1], dz = r[2] - c[2];
        rho[g] = amp * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
    }
    return rho;
}
} // namespace

// Constant density => grad rho = 0 => PBE reduces to its LDA limit and the
// divergence term vanishes.
TEST(GpwXcGGA, ConstantDensityLdaLimit) {
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid xc = xc_grid_pbe(grid, rho);

    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::PBE, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(xc.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
    // grad = 0 -> potential is purely the local vrho.
    EXPECT_NEAR(xc.vxc_r[0], ref.vrho_alpha, 1e-9);
    EXPECT_NEAR(xc.vxc_r[grid.npts() / 2], ref.vrho_alpha, 1e-9);
}

TEST(GpwXcGGA, GaussianEnergyNegative) {
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    XcOnGrid xc = xc_grid_pbe(grid, rho);
    EXPECT_TRUE(std::isfinite(xc.e_xc));
    EXPECT_LT(xc.e_xc, 0.0);
}

// Functional-derivative consistency: for a band-limited perturbation direction
// `bump`, d E_xc / d eps = integral v_xc bump dr. Central finite difference vs
// the assembled v_xc (this exercises the divergence term and its adjoint).
TEST(GpwXcGGA, PotentialIsFunctionalDerivative) {
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 120.0);
    const long ng = grid.npts();
    const double w = grid.weight();

    const double cR[3] = {3.5, 4.0, 4.0};
    std::vector<double> rho = gaussian_density(grid, cR, 0.7, 1.0);
    // Smooth perturbation direction (well-resolved => band-limited, no Nyquist).
    const double cB[3] = {4.5, 4.0, 4.0};
    std::vector<double> bump = gaussian_density(grid, cB, 0.9, 1.0);

    XcOnGrid xc0 = xc_grid_pbe(grid, rho);
    double analytic = 0.0; // integral v_xc * bump
    for (long g = 0; g < ng; ++g)
        analytic += w * xc0.vxc_r[g] * bump[g];

    const double eps = 1e-5;
    std::vector<double> rp(rho.size()), rm(rho.size());
    for (long g = 0; g < ng; ++g) {
        rp[g] = rho[g] + eps * bump[g];
        rm[g] = rho[g] - eps * bump[g];
    }
    const double ep = xc_grid_pbe(grid, rp).e_xc;
    const double em = xc_grid_pbe(grid, rm).e_xc;
    const double fd = (ep - em) / (2.0 * eps);

    EXPECT_NEAR(fd, analytic, 1e-4 * std::max(1.0, std::abs(analytic)))
        << "fd=" << fd << " analytic=" << analytic;
}
