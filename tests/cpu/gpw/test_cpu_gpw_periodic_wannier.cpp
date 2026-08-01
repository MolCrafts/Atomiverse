/// @file test_cpu_gpw_periodic_wannier.cpp
/// @brief Unit tests for GPW periodic MLWF pure math (bond midpoints, SMV
///        weights, phase→center, argument gates). No full SCF.

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/periodic_wannier.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

TEST(GpwPeriodicWannierMath, DiamondBctBondMidpoints) {
    const double a = 5.43;
    const auto m = diamond_bct_bond_midpoints(a);
    EXPECT_EQ(m.size(), 8u);
    // All midpoints finite; pairwise min-image separations non-zero.
    const double at = a / std::sqrt(2.0);
    const double L[3] = {at, at, a};
    for (int i = 0; i < 8; ++i) {
        for (int c = 0; c < 3; ++c)
            EXPECT_TRUE(std::isfinite(m[static_cast<std::size_t>(i)]
                                         [static_cast<std::size_t>(c)]));
        for (int j = i + 1; j < 8; ++j) {
            double dx = m[static_cast<std::size_t>(i)][0]
                        - m[static_cast<std::size_t>(j)][0];
            double dy = m[static_cast<std::size_t>(i)][1]
                        - m[static_cast<std::size_t>(j)][1];
            double dz = m[static_cast<std::size_t>(i)][2]
                        - m[static_cast<std::size_t>(j)][2];
            dx -= L[0] * std::nearbyint(dx / L[0]);
            dy -= L[1] * std::nearbyint(dy / L[1]);
            dz -= L[2] * std::nearbyint(dz / L[2]);
            EXPECT_GT(dx * dx + dy * dy + dz * dz, 1e-6);
        }
    }
    // Each midpoint is halfway between two Si sites (distance to nearest atom
    // = half bond = a√3/8).
    const double half_bond = a * std::sqrt(3.0) / 8.0;
    const std::array<std::array<double, 3>, 4> atoms = {{
        {{0.0, 0.0, 0.0}},
        {{0.5 * at, 0.5 * at, 0.5 * a}},
        {{0.5 * at, 0.0, 0.25 * a}},
        {{0.0, 0.5 * at, 0.75 * a}},
    }};
    for (int i = 0; i < 8; ++i) {
        double dmin = 1e9;
        for (int j = 0; j < 4; ++j) {
            double dx = m[static_cast<std::size_t>(i)][0] - atoms[static_cast<std::size_t>(j)][0];
            double dy = m[static_cast<std::size_t>(i)][1] - atoms[static_cast<std::size_t>(j)][1];
            double dz = m[static_cast<std::size_t>(i)][2] - atoms[static_cast<std::size_t>(j)][2];
            dx -= L[0] * std::nearbyint(dx / L[0]);
            dy -= L[1] * std::nearbyint(dy / L[1]);
            dz -= L[2] * std::nearbyint(dz / L[2]);
            dmin = std::min(dmin, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        EXPECT_NEAR(dmin, half_bond, 1e-6);
    }
    EXPECT_THROW(diamond_bct_bond_midpoints(0.0), std::invalid_argument);
    EXPECT_THROW(diamond_bct_bond_midpoints(-1.0), std::invalid_argument);
}

TEST(GpwPeriodicWannierMath, CubicSmvWeight) {
    const double b = 0.5;
    EXPECT_NEAR(mv_pair_weight_cubic(b), 1.0 / (2.0 * b * b), 1e-14);
    EXPECT_THROW(mv_pair_weight_cubic(0.0), std::invalid_argument);
    EXPECT_THROW(mv_pair_weight_cubic(-0.1), std::invalid_argument);
}

TEST(GpwPeriodicWannierMath, CenterFromDiagonalPhases) {
    const double b = 1.2;
    const double wb = mv_pair_weight_cubic(b);
    const std::array<double, 3> r0{0.3, -0.2, 0.5};
    std::vector<std::array<double, 3>> b_cart = {
        {{b, 0, 0}},  {{-b, 0, 0}}, {{0, b, 0}},
        {{0, -b, 0}}, {{0, 0, b}},  {{0, 0, -b}},
    };
    std::vector<double> w(6, wb);
    std::vector<std::complex<double>> M(6);
    for (int i = 0; i < 6; ++i) {
        const double phase = b_cart[static_cast<std::size_t>(i)][0] * r0[0]
                             + b_cart[static_cast<std::size_t>(i)][1] * r0[1]
                             + b_cart[static_cast<std::size_t>(i)][2] * r0[2];
        M[static_cast<std::size_t>(i)] =
            std::exp(std::complex<double>{0, -phase});
    }
    const auto r = center_from_diagonal_phases(b_cart, w, M);
    EXPECT_NEAR(r[0], r0[0], 1e-9);
    EXPECT_NEAR(r[1], r0[1], 1e-9);
    EXPECT_NEAR(r[2], r0[2], 1e-9);
    EXPECT_THROW(center_from_diagonal_phases({}, {}, {}),
                 std::invalid_argument);
}

TEST(GpwPeriodicWannierMath, InvalidArgumentsOnCompute) {
    PeriodicWannier mlwf;
    mlwf.k_mesh = 0;
    GpwDft method;
    method.n_occ = 1;
    method.cell[0] = method.cell[1] = method.cell[2] = 10.0;
    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double pos[3] = {5.0, 5.0, 5.0};
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos, bH, pH, 1);
    method.build(atoms);
    EXPECT_THROW(mlwf.compute(method, atoms), std::invalid_argument);
    mlwf.k_mesh = 2;
    mlwf.tol = 0.0;
    EXPECT_THROW(mlwf.compute(method, atoms), std::invalid_argument);
    mlwf.tol = 1e-8;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.01;
    EXPECT_THROW(mlwf.compute(method, atoms), std::invalid_argument);
    method.smearing = Smearing::none;
}

TEST(GpwPeriodicWannierMath, UksNoLongerFootholdThrow) {
    // Collinear UKS is product: the foothold message is gone. Invalid k_mesh
    // still fails before SCF; metallic smearing still rejected.
    PeriodicWannier mlwf;
    mlwf.k_mesh = 0;
    GpwDft uks;
    uks.n_alpha = 1;
    uks.n_beta = 0;
    uks.cell[0] = uks.cell[1] = uks.cell[2] = 10.0;
    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double pos[3] = {5.0, 5.0, 5.0};
    std::vector<GpwAtom> atoms;
    atoms.emplace_back(pos, bH, pH, 1);
    uks.build(atoms);
    try {
        mlwf.compute(uks, atoms);
        FAIL() << "expected invalid_argument for k_mesh=0";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("UKS / spin MLWF not in this foothold"),
                  std::string::npos);
        EXPECT_NE(msg.find("k_mesh"), std::string::npos);
    }
    mlwf.k_mesh = 2;
    uks.smearing = Smearing::fermi_dirac;
    uks.degauss = 0.01;
    EXPECT_THROW(mlwf.compute(uks, atoms), std::invalid_argument);
}
