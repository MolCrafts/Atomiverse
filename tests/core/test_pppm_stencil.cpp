/// @file tests/core/test_pppm_stencil.cpp
/// @brief core/pppm_ik.h against literal LAMMPS ports: the centred stencil
///        (`particle_map` + `compute_rho1d`), the influence denominator
///        coefficients and the ik Green's function, and the mesh-size rule.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/longrange/pme_spline.h"
#include "core/pppm_lammps_reference.h"

using namespace ATV;
namespace ref = ATV::test::lammps_pppm;

TEST(PppmStencil, OffsetsAreLammpsNlowerNupper) {
    // set_grid_local: nlower = -(order-1)/2, nupper = order/2.
    const int lower[] = {0, 0, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5};
    const int upper[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
    for (int order = 2; order <= 12; ++order) {
        EXPECT_EQ(longrange::pppm_stencil_lower(order), lower[order - 1]);
        EXPECT_EQ(longrange::pppm_stencil_upper(order), upper[order - 1]);
        EXPECT_EQ(longrange::pppm_stencil_upper(order)
                      - longrange::pppm_stencil_lower(order) + 1,
                  order);
    }
}

TEST(PppmStencil, OddOrderCentresOnNearestPointEvenOnLowerLeft) {
    double frac = 0.0;
    // u = 3.3: nearest point 3, lower-left point 3.
    EXPECT_EQ(longrange::pppm_particle_map(3.3, 5, frac), 3);
    EXPECT_NEAR(frac, 0.8, 1e-15); // 0.5 − dx, dx = 3 − 3.3
    EXPECT_EQ(longrange::pppm_particle_map(3.3, 6, frac), 3);
    EXPECT_NEAR(frac, 0.3, 1e-15);
    // u = 3.7: nearest point 4, lower-left point 3.
    EXPECT_EQ(longrange::pppm_particle_map(3.7, 5, frac), 4);
    EXPECT_NEAR(frac, 0.2, 1e-15);
    EXPECT_EQ(longrange::pppm_particle_map(3.7, 7, frac), 4);
    EXPECT_EQ(longrange::pppm_particle_map(3.7, 6, frac), 3);
    EXPECT_NEAR(frac, 0.7, 1e-15);
    // Order 7 at u = 3.7 covers 1..7; order 6 covers 1..6 (the one-sided
    // smooth-PME stencil would be −3..3 and −2..3).
    EXPECT_EQ(longrange::pppm_particle_map(3.7, 7, frac)
                  + longrange::pppm_stencil_lower(7),
              1);
    EXPECT_EQ(longrange::pppm_particle_map(3.7, 6, frac)
                  + longrange::pppm_stencil_lower(6),
              1);
    // Slightly below a grid point keeps the floor below it (OFFSET).
    EXPECT_EQ(longrange::pppm_particle_map(-0.25, 6, frac), -1);
    EXPECT_EQ(longrange::pppm_particle_map(-0.75, 5, frac), -1);
}

TEST(PppmStencil, WeightsAndPlacementMatchLammpsRho1d) {
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> uni(0.0, 32.0);
    // bspline() needs order ≥ 3 (both solvers reject smaller orders).
    for (int order = 3; order <= 12; ++order) {
        const std::vector<double> coeff = ref::rho_coeff(order);
        double theta[12], dtheta[12], lammps[12];
        for (int trial = 0; trial < 400; ++trial) {
            const double u = trial < 3 ? 5.0 + 0.5 * trial : uni(rng);
            double frac = 0.0, dx = 0.0;
            const int nx = longrange::pppm_particle_map(u, order, frac);
            ASSERT_EQ(nx, ref::particle_map(u, order, dx))
                << "order " << order << " u " << u;
            ASSERT_GE(frac, 0.0);
            ASSERT_LT(frac, 1.0);
            longrange::bspline(frac, order, theta, dtheta);
            ref::rho1d(coeff, order, dx, lammps);
            double sum = 0.0;
            for (int a = 0; a < order; ++a) {
                // frac and dx each round at the ulp of u ≤ 32.
                EXPECT_NEAR(theta[a], lammps[a], 1e-14)
                    << "order " << order << " u " << u << " point "
                    << nx + longrange::pppm_stencil_lower(order) + a;
                sum += theta[a];
            }
            EXPECT_NEAR(sum, 1.0, 1e-14);
        }
    }
}

TEST(PppmStencil, OddOrderSmoothPmeStencilIsHalfACellAway) {
    // The one-sided Essmann stencil puts weight M_p(u − g) on g ≤ ⌊u⌋; the
    // centred one W_p(g − u). They coincide up to a whole-cell shift for
    // even p and differ for odd p: the defect this port removes.
    for (int order : {5, 6, 7}) {
        const double u = 10.3;
        double frac = 0.0, theta_c[12], theta_o[12], d[12];
        const int nx = longrange::pppm_particle_map(u, order, frac);
        longrange::bspline(frac, order, theta_c, d);
        const int u0 = static_cast<int>(std::floor(u));
        longrange::bspline(u - u0, order, theta_o, d);
        double diff = 0.0;
        for (int a = 0; a < order; ++a)
            diff = std::max(diff, std::abs(theta_c[a] - theta_o[a]));
        if (order % 2 == 0) {
            EXPECT_LT(diff, 1e-15) << order;
            EXPECT_EQ(nx + longrange::pppm_stencil_lower(order) - order / 2,
                      u0 - (order - 1))
                << "even orders: centred = one-sided + order/2 cells";
        } else {
            EXPECT_GT(diff, 1e-2) << order;
        }
    }
}

TEST(PppmInfluence, DenominatorCoefficientsMatchComputeGfDenom) {
    for (int order = 2; order <= 10; ++order) {
        double b[12] = {};
        longrange::pppm_gf_b_fill(order, b);
        const std::vector<double> lammps = ref::gf_b(order);
        for (int l = 0; l < order; ++l)
            EXPECT_NEAR(b[l], lammps[l], 1e-15 * std::abs(lammps[l]))
                << "order " << order << " l " << l;
        for (double x : {0.0, 0.13, 0.5, 0.97})
            EXPECT_NEAR(longrange::pppm_gf_denom_scalar(b, order, x, 0.3, 0.8),
                        ref::gf_denom(lammps, x, 0.3, 0.8),
                        1e-15 * ref::gf_denom(lammps, x, 0.3, 0.8));
    }
}

namespace {
void expect_green_matches(const double* cell, const int K[3], double alpha,
                          int order) {
    double inv[9];
    ref::inverse3(cell, inv);
    double gf_b[12] = {};
    longrange::pppm_gf_b_fill(order, gf_b);
    // LAMMPS writes 4π as 12.5663706; compare with that factor divided out.
    const double scale = ref::kLammpsFourPi / (4.0 * ATV::units::pi);
    double worst = 0.0, gmax = 0.0;
    for (int m0 = 0; m0 < K[0]; ++m0)
        for (int m1 = 0; m1 < K[1]; ++m1)
            for (int m2 = 0; m2 < K[2]; ++m2) {
                const int h0 = longrange::pppm_signed_mode(m0, K[0]);
                const int h1 = longrange::pppm_signed_mode(m1, K[1]);
                const int h2 = longrange::pppm_signed_mode(m2, K[2]);
                const double g = longrange::pppm_green_ik(
                    h0, h1, h2, cell, K, alpha, order, gf_b, inv);
                const double l = ref::green_ik(h0, h1, h2, cell, K, alpha,
                                               order, ref::kLammpsFourPi);
                worst = std::max(worst, std::abs(g * scale - l));
                gmax = std::max(gmax, std::abs(l));
            }
    EXPECT_GT(gmax, 0.0);
    EXPECT_LT(worst, 1e-13 * gmax) << "order " << order;
}
} // namespace

TEST(PppmInfluence, OrthorhombicMatchesComputeGfIk) {
    const double cell[9] = {6.3, 0, 0, 0, 7.1, 0, 0, 0, 10.5};
    const int K[3] = {8, 9, 15};
    for (int order : {3, 4, 5, 6, 7})
        expect_green_matches(cell, K, 3.0, order); // ±1 alias images
}

TEST(PppmInfluence, TriclinicAliasBoundMatchesComputeGfIkTriclinic) {
    // Upper-triangular H (LAMMPS restricted triclinic); xy, xz, yz tilts
    // make H^T·bound differ from the column-length bound on y and z.
    const double cell[9] = {6.3, 2.9, -2.4, 0, 6.25, 2.6, 0, 0, 10.4};
    const int K[3] = {6, 6, 10};
    for (int order : {4, 5, 7})
        expect_green_matches(cell, K, 2.2, order);
}

TEST(PppmMesh, SizesAreLammpsFactorable) {
    EXPECT_EQ(longrange::pppm_fft_size(1), 4);
    EXPECT_EQ(longrange::pppm_fft_size(7), 8);
    EXPECT_EQ(longrange::pppm_fft_size(9), 9);
    EXPECT_EQ(longrange::pppm_fft_size(13), 15);
    EXPECT_EQ(longrange::pppm_fft_size(14), 15);
    EXPECT_EQ(longrange::pppm_fft_size(28), 30);
    EXPECT_EQ(longrange::pppm_fft_size(32), 32);
    EXPECT_EQ(longrange::pppm_fft_size(49), 50);
}
