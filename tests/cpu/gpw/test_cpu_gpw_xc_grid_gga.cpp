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

TEST(GpwXcGGA, PBEsolConstantDensityMatchesPBE) {
    using ATV::cpu::gpw::xc_grid_pbesol;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid sol = xc_grid_pbesol(grid, rho);
    EXPECT_NEAR(sol.e_xc, pbe.e_xc, 1e-12 * std::max(1.0, std::abs(pbe.e_xc)));
    EXPECT_NEAR(sol.vxc_r[0], pbe.vxc_r[0], 1e-12);

    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::PBEsol, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(sol.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
}

// BLYP at σ=0 must match evaluate_xc(BLYP) LDA limit (B88+LYP at zero grad).
TEST(GpwXcGGA, BLYPConstantDensityLdaLimit) {
    using ATV::cpu::gpw::xc_grid_blyp;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid xc = xc_grid_blyp(grid, rho);
    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::BLYP, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(xc.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
    EXPECT_NEAR(xc.vxc_r[0], ref.vrho_alpha, 1e-9);
}

// Nonuniform density: BLYP and PBE GGA energies must differ (activation gate).
TEST(GpwXcGGA, BLYPDiffersFromPBEOnGaussian) {
    using ATV::cpu::gpw::xc_grid_blyp;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid blyp = xc_grid_blyp(grid, rho);
    EXPECT_TRUE(std::isfinite(blyp.e_xc));
    EXPECT_LT(blyp.e_xc, 0.0);
    EXPECT_GT(std::abs(blyp.e_xc - pbe.e_xc), 1e-6)
        << "BLYP inactive? |dE|=" << std::abs(blyp.e_xc - pbe.e_xc);
}

// revPBE at σ=0 matches evaluate_xc(RevPBE) LDA limit (= PBE LDA base).
TEST(GpwXcGGA, RevPBEConstantDensityMatchesPBE) {
    using ATV::cpu::gpw::xc_grid_revpbe;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid rev = xc_grid_revpbe(grid, rho);
    EXPECT_NEAR(rev.e_xc, pbe.e_xc, 1e-12 * std::max(1.0, std::abs(pbe.e_xc)));
    EXPECT_NEAR(rev.vxc_r[0], pbe.vxc_r[0], 1e-12);

    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::RevPBE, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(rev.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
}

// Nonuniform density: revPBE and PBE must differ (κ activation gate).
TEST(GpwXcGGA, RevPBEDiffersFromPBEOnGaussian) {
    using ATV::cpu::gpw::xc_grid_revpbe;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid rev = xc_grid_revpbe(grid, rho);
    EXPECT_TRUE(std::isfinite(rev.e_xc));
    EXPECT_LT(rev.e_xc, 0.0);
    EXPECT_GT(std::abs(rev.e_xc - pbe.e_xc), 1e-6)
        << "revPBE inactive? |dE|=" << std::abs(rev.e_xc - pbe.e_xc);
}

// RPBE at σ=0 matches evaluate_xc(RPBE) LDA limit (= PBE LDA base).
TEST(GpwXcGGA, RPBEConstantDensityMatchesPBE) {
    using ATV::cpu::gpw::xc_grid_rpbe;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid rpbe = xc_grid_rpbe(grid, rho);
    EXPECT_NEAR(rpbe.e_xc, pbe.e_xc, 1e-12 * std::max(1.0, std::abs(pbe.e_xc)));
    EXPECT_NEAR(rpbe.vxc_r[0], pbe.vxc_r[0], 1e-12);

    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::RPBE, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(rpbe.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
}

// Nonuniform density: RPBE and PBE must differ (exponential F_x gate).
TEST(GpwXcGGA, RPBEDiffersFromPBEOnGaussian) {
    using ATV::cpu::gpw::xc_grid_rpbe;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid rpbe = xc_grid_rpbe(grid, rho);
    EXPECT_TRUE(std::isfinite(rpbe.e_xc));
    EXPECT_LT(rpbe.e_xc, 0.0);
    EXPECT_GT(std::abs(rpbe.e_xc - pbe.e_xc), 1e-6)
        << "RPBE inactive? |dE|=" << std::abs(rpbe.e_xc - pbe.e_xc);
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

// Spin-resolved BLYP collapses to closed-shell BLYP; differs from spin-PBE.
TEST(GpwXcGGA, BLYPSpinCollapseAndDiffersFromPBE) {
    using ATV::cpu::gpw::xc_grid_blyp;
    using ATV::cpu::gpw::xc_grid_blyp_spin;
    using ATV::cpu::gpw::xc_grid_pbe_spin;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    std::vector<double> ra(rho.size()), rb(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g) {
        ra[g] = 0.55 * rho[g];
        rb[g] = 0.45 * rho[g];
    }
    // polarized densities: BLYP vs PBE must differ
    auto blyp = xc_grid_blyp_spin(grid, ra, rb);
    auto pbe = xc_grid_pbe_spin(grid, ra, rb);
    EXPECT_TRUE(std::isfinite(blyp.e_xc));
    EXPECT_GT(std::abs(blyp.e_xc - pbe.e_xc), 1e-6);
    // equal-spin collapse
    for (std::size_t g = 0; g < rho.size(); ++g) {
        ra[g] = 0.5 * rho[g];
        rb[g] = 0.5 * rho[g];
    }
    auto uni = xc_grid_blyp(grid, rho);
    auto spin = xc_grid_blyp_spin(grid, ra, rb);
    EXPECT_NEAR(uni.e_xc, spin.e_xc, 1e-9 * std::max(1.0, std::abs(uni.e_xc)));
}

TEST(GpwXcGGA, SpinCollapseMatchesClosedShell) {
    using ATV::cpu::gpw::xc_grid_pbe_spin;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 120.0);
    const double c[3] = {4.0, 4.0, 4.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    std::vector<double> ra(rho.size()), rb(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g) {
        ra[g] = 0.5 * rho[g];
        rb[g] = 0.5 * rho[g];
    }
    XcOnGrid uni = xc_grid_pbe(grid, rho);
    auto spin = xc_grid_pbe_spin(grid, ra, rb);
    EXPECT_NEAR(uni.e_xc, spin.e_xc, 1e-9 * std::max(1.0, std::abs(uni.e_xc)));
    for (long g = 0; g < grid.npts(); ++g) {
        EXPECT_NEAR(uni.vxc_r[static_cast<std::size_t>(g)],
                    spin.vxc_a[static_cast<std::size_t>(g)], 1e-8);
        EXPECT_NEAR(spin.vxc_a[static_cast<std::size_t>(g)],
                    spin.vxc_b[static_cast<std::size_t>(g)], 1e-12);
    }
}

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

// OptB88VdW at σ=0 matches evaluate_xc (optB88-x + PW92).
TEST(GpwXcGGA, OptB88VdWConstantDensityLdaLimit) {
    using ATV::cpu::gpw::xc_grid_optb88_vdw;
    const double cell[3] = {8.0, 8.0, 8.0};
    UniformGrid grid = make_uniform_grid(cell, 150.0);
    const double rho0 = 0.2;
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), rho0);

    XcOnGrid xc = xc_grid_optb88_vdw(grid, rho);
    const ATV::cpu::scf::XCOutput ref = ATV::cpu::scf::evaluate_xc(
        ATV::cpu::scf::XCType::OptB88VdW, 0.5 * rho0, 0.5 * rho0, 0, 0, 0);
    const double volume = cell[0] * cell[1] * cell[2];
    EXPECT_NEAR(xc.e_xc, volume * rho0 * ref.exc,
                1e-9 * std::abs(volume * rho0 * ref.exc));
    EXPECT_NEAR(xc.vxc_r[0], ref.vrho_alpha, 1e-9);
}

// Nonuniform density: OptB88VdW differs from PBE (activation).
TEST(GpwXcGGA, OptB88VdWDiffersFromPBEOnGaussian) {
    using ATV::cpu::gpw::xc_grid_optb88_vdw;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    XcOnGrid pbe = xc_grid_pbe(grid, rho);
    XcOnGrid opt = xc_grid_optb88_vdw(grid, rho);
    EXPECT_TRUE(std::isfinite(opt.e_xc));
    EXPECT_LT(opt.e_xc, 0.0);
    EXPECT_GT(std::abs(opt.e_xc - pbe.e_xc), 1e-6)
        << "OptB88VdW inactive? |dE|=" << std::abs(opt.e_xc - pbe.e_xc);
}

// Spin-resolved OptB88VdW collapses to closed-shell; differs from spin-PBE.
TEST(GpwXcGGA, OptB88VdWSpinCollapseAndDiffersFromPBE) {
    using ATV::cpu::gpw::xc_grid_optb88_vdw;
    using ATV::cpu::gpw::xc_grid_optb88_vdw_spin;
    using ATV::cpu::gpw::xc_grid_pbe_spin;
    const double cell[3] = {10.0, 10.0, 10.0};
    UniformGrid grid = make_uniform_grid(cell, 200.0);
    const double c[3] = {5.0, 5.0, 5.0};
    std::vector<double> rho = gaussian_density(grid, c, 0.8, 1.0);
    std::vector<double> ra(rho.size()), rb(rho.size());
    for (std::size_t g = 0; g < rho.size(); ++g) {
        ra[g] = 0.55 * rho[g];
        rb[g] = 0.45 * rho[g];
    }
    auto opt = xc_grid_optb88_vdw_spin(grid, ra, rb);
    auto pbe = xc_grid_pbe_spin(grid, ra, rb);
    EXPECT_TRUE(std::isfinite(opt.e_xc));
    EXPECT_GT(std::abs(opt.e_xc - pbe.e_xc), 1e-6)
        << "OptB88VdW spin inactive? |dE|=" << std::abs(opt.e_xc - pbe.e_xc);
    for (std::size_t g = 0; g < rho.size(); ++g) {
        ra[g] = 0.5 * rho[g];
        rb[g] = 0.5 * rho[g];
    }
    auto uni = xc_grid_optb88_vdw(grid, rho);
    auto spin = xc_grid_optb88_vdw_spin(grid, ra, rb);
    EXPECT_NEAR(uni.e_xc, spin.e_xc, 1e-9 * std::max(1.0, std::abs(uni.e_xc)));
}
