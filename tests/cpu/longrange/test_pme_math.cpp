/// @file tests/cpu/longrange/test_pme_math.cpp
/// @brief Unit tests for ATV::longrange PME pure math (spline / moduli / FFT size).
///
/// Expected values pinned to the pre-refactor anonymous-namespace helpers
/// (now longrange/pme_spline.h). No Frame/Component/MPI/FFTW.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/cpu/longrange/pme_influence.h"
#include "atomiverse/cpu/longrange/pme_spline.h"

using namespace ATV;
using ATV::longrange::bspline;
using ATV::longrange::good_fft_size;
using ATV::longrange::spline_moduli;

TEST(PmeMath, GoodFftSizeFloorIsFour) {
    EXPECT_EQ(good_fft_size(1), 4);
    EXPECT_EQ(good_fft_size(2), 4);
    EXPECT_EQ(good_fft_size(3), 4);
    EXPECT_EQ(good_fft_size(4), 4);
}

TEST(PmeMath, GoodFftSizePinnedFactors) {
    // Pre-refactor pins: smallest m ≥ n factorising into {2,3,5,7}.
    EXPECT_EQ(good_fft_size(11), 12);
    EXPECT_EQ(good_fft_size(13), 14);
    EXPECT_EQ(good_fft_size(15), 15);
    EXPECT_EQ(good_fft_size(16), 16);
    EXPECT_EQ(good_fft_size(23), 24);
    EXPECT_EQ(good_fft_size(49), 49); // 7·7
}

TEST(PmeMath, BsplinePartitionOfUnity) {
    // Cardinal B-splines of order n ≥ 2 form a partition of unity on [0,1).
    for (int n : {3, 4, 5, 6, 8}) {
        for (double frac : {0.0, 0.1, 0.33, 0.5, 0.75, 0.999}) {
            std::vector<double> th(static_cast<std::size_t>(n)),
                dth(static_cast<std::size_t>(n));
            bspline(frac, n, th.data(), dth.data());
            double s = 0;
            for (int k = 0; k < n; ++k)
                s += th[static_cast<std::size_t>(k)];
            EXPECT_NEAR(s, 1.0, 1e-12) << "n=" << n << " frac=" << frac;
        }
    }
}

TEST(PmeMath, BsplineOrderFourEssmannAppendix) {
    // Order-4 cardinal B-spline at frac=0: fill_bspline returns
    // [M_4(1), M_4(2), M_4(3), 0] = [1/6, 4/6, 1/6, 0] (Essmann appendix /
    // pre-refactor pin).
    double th[4], dth[4];
    bspline(0.0, 4, th, dth);
    EXPECT_NEAR(th[0], 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(th[1], 4.0 / 6.0, 1e-14);
    EXPECT_NEAR(th[2], 1.0 / 6.0, 1e-14);
    EXPECT_NEAR(th[3], 0.0, 1e-14);
}

TEST(PmeMath, SplineModuliEvenOrderHalfNyquistRepaired) {
    // Even order: any exact zero (denom ≤ 1e-12, often near i=K/2) is repaired
    // by neighbour averaging so every entry is strictly positive.
    auto mod = spline_moduli(/*K=*/8, /*n=*/4);
    ASSERT_EQ(mod.size(), 8u);
    for (double m : mod)
        EXPECT_GT(m, 0.0);
    // Pre-refactor pin for K=8, order=4, zero mode and Nyquist.
    EXPECT_NEAR(mod[0], 1.0, 1e-12); // |Σ M_4(k)|⁻² at m=0
    EXPECT_NEAR(mod[4], 9.0, 1e-12); // pin from pre-refactor spline_moduli
}

TEST(PmeMath, SplineModuliPinnedZeroMode) {
    // Pre-refactor pin: |b(0)|⁻² for order 6, K=16.
    // M_6 values at integers → sr = Σ data[k], si=0 → mod[0] = 1/sr².
    auto mod = spline_moduli(/*K=*/16, /*n=*/6);
    ASSERT_EQ(mod.size(), 16u);
    double th[6], dth[6];
    bspline(0.0, 6, th, dth);
    double sr = 0;
    for (int k = 0; k < 5; ++k)
        sr += th[k];
    EXPECT_NEAR(mod[0], 1.0 / (sr * sr), 1e-12);
}

TEST(PmeMath, NonSymmetricCellSplineAndForceUseDualTransforms) {
    Mat3 h = Mat3::diagonal(4.0, 5.0, 6.0);
    h(0, 1) = 1.5;
    h(2, 0) = 0.8;
    const Mat3 inv = h.inverse();
    const Mat3 identity = Mat3::diagonal(1.0, 1.0, 1.0);
    const Vec3 s{0.23, 0.37, 0.41};
    const Vec3 r = h * s;
    const StorageReal x[] = {r[0]}, y[] = {r[1]}, z[] = {r[2]};
    const StorageReal sx[] = {s[0]}, sy[] = {s[1]}, sz[] = {s[2]};
    const int K[3] = {8, 10, 12};
    constexpr int order = 4;
    int base[3], base_ref[3];
    double theta[12], dtheta[12], theta_ref[12], dtheta_ref[12];
    longrange::pme_fill_atom_splines(x, y, z, 1, inv, K, order, base, theta,
                                     dtheta);
    longrange::pme_fill_atom_splines(sx, sy, sz, 1, identity, K, order,
                                     base_ref, theta_ref, dtheta_ref);
    for (int d = 0; d < 3; ++d)
        EXPECT_EQ(base[d], base_ref[d]);
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(theta[i], theta_ref[i], 1e-12);
        EXPECT_NEAR(dtheta[i], dtheta_ref[i], 1e-12);
    }
    std::vector<std::complex<double>> grid(K[0] * K[1] * K[2]);
    for (std::size_t i = 0; i < grid.size(); ++i)
        grid[i] = std::sin(0.37 * static_cast<double>(i));
    const StorageReal q[] = {0.7};
    AccReal fx[1]{}, fy[1]{}, fz[1]{}, sxforce[1]{}, syforce[1]{}, szforce[1]{};
    longrange::pme_interpolate_forces(q, 1, order, K, inv, 1.0, base, theta,
                                      dtheta, grid.data(), fx, fy, fz);
    longrange::pme_interpolate_forces(q, 1, order, K, identity, 1.0, base_ref,
                                      theta_ref, dtheta_ref, grid.data(),
                                      sxforce, syforce, szforce);
    // Cartesian forces must be H^-T times the fractional-coordinate gradient.
    const Vec3 expected =
        inv.transpose() * Vec3{sxforce[0], syforce[0], szforce[0]};
    EXPECT_NEAR(fx[0], expected[0], 1e-11);
    EXPECT_NEAR(fy[0], expected[1], 1e-11);
    EXPECT_NEAR(fz[0], expected[2], 1e-11);
}

TEST(PmeMath, SkewInfluenceUsesReciprocalColumnVector) {
    Mat3 h = Mat3::diagonal(4.0, 5.0, 6.0);
    h(0, 1) = 1.5;
    h(2, 0) = 0.8;
    const Mat3 inv = h.inverse();
    const int K[3] = {4, 4, 4};
    const double mod[] = {1, 1, 1, 1};
    longrange::PmeInfluenceTables table;
    longrange::pme_build_influence_tables(K, mod, mod, mod, inv, 0.4, table);
    const Vec3 m = inv.transpose() * Vec3{1, 1, -1};
    const int index = 3 + 4 * (1 + 4 * 1);
    EXPECT_NEAR(table.mx[index], m[0], 1e-14);
    EXPECT_NEAR(table.my[index], m[1], 1e-14);
    EXPECT_NEAR(table.mz[index], m[2], 1e-14);
    const double expected =
        std::exp(-units::pi * units::pi * m.norm_squared() / 0.16)
        / m.norm_squared();
    EXPECT_NEAR(table.H[index], expected, 1e-14);
}
