/// @file tests/cpu/longrange/test_pme_math.cpp
/// @brief Unit tests for ATV::longrange PME pure math (spline / moduli / FFT size).
///
/// Expected values pinned to the pre-refactor anonymous-namespace helpers
/// (now longrange/pme_spline.h). No Frame/Component/MPI/FFTW.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/longrange/pme_spline.h"

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
