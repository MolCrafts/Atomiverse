/// @file test_box.cpp
/// @brief Unit tests for Box value type.

#include <gtest/gtest.h>

#include <random>

#include "atomiverse/core/box.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/vec3.h"

using namespace ATV;

namespace {
constexpr AccReal kTol = 1e-10;
}

TEST(Box, CubicFactory) {
    auto b = Box::cubic(10);
    EXPECT_EQ(b.cell(0, 0), AccReal(10));
    EXPECT_EQ(b.cell(1, 1), AccReal(10));
    EXPECT_EQ(b.cell(2, 2), AccReal(10));
    EXPECT_EQ(b.cell(0, 1), AccReal(0));
    EXPECT_EQ(b.origin[0], AccReal(0));
    EXPECT_EQ(b.origin[1], AccReal(0));
    EXPECT_EQ(b.origin[2], AccReal(0));
    EXPECT_TRUE(b.pbc[0]);
    EXPECT_TRUE(b.pbc[1]);
    EXPECT_TRUE(b.pbc[2]);
}

TEST(Box, Volume) {
    EXPECT_NEAR(Box::cubic(10).volume(), AccReal(1000), kTol);
    EXPECT_NEAR(Box::cubic(5.43).volume(), AccReal(5.43 * 5.43 * 5.43), kTol);
}

TEST(Box, ToFracToCartRoundtrip) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<AccReal> dist(-50, 50);
    auto box = Box::cubic(10);
    for (int n = 0; n < 5; ++n) {
        Vec3 cart{dist(rng), dist(rng), dist(rng)};
        auto frac = box.to_frac(cart);
        auto back = box.to_cart(frac);
        EXPECT_NEAR(back[0], cart[0], kTol);
        EXPECT_NEAR(back[1], cart[1], kTol);
        EXPECT_NEAR(back[2], cart[2], kTol);
    }
}

TEST(Box, Wrap) {
    auto box = Box::cubic(10);
    for (int n = 0; n < 100; ++n) {
        Vec3 cart{AccReal(-50 + n % 100), AccReal(-30 + (n * 7) % 60),
                  AccReal(-20 + (n * 13) % 40)};
        auto wrapped = box.wrap(cart);
        auto frac = box.to_frac(wrapped);
        EXPECT_GE(frac[0], AccReal(0)) << "frac x < 0";
        EXPECT_LT(frac[0], AccReal(1)) << "frac x >= 1";
        EXPECT_GE(frac[1], AccReal(0)) << "frac y < 0";
        EXPECT_LT(frac[1], AccReal(1)) << "frac y >= 1";
        EXPECT_GE(frac[2], AccReal(0)) << "frac z < 0";
        EXPECT_LT(frac[2], AccReal(1)) << "frac z >= 1";
    }
}

TEST(Box, ShortestVector) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<AccReal> dist(-8, 8);
    auto box = Box::cubic(10);
    for (int n = 0; n < 100; ++n) {
        Vec3 a{dist(rng), dist(rng), dist(rng)};
        Vec3 b{dist(rng), dist(rng), dist(rng)};
        auto dr = box.shortest_vector(a, b);
        auto frac = box.to_frac(dr);
        EXPECT_GE(frac[0], AccReal(-0.5) - kTol);
        EXPECT_LE(frac[0], AccReal(0.5) + kTol);
        EXPECT_GE(frac[1], AccReal(-0.5) - kTol);
        EXPECT_LE(frac[1], AccReal(0.5) + kTol);
        EXPECT_GE(frac[2], AccReal(-0.5) - kTol);
        EXPECT_LE(frac[2], AccReal(0.5) + kTol);
    }
}

TEST(Box, RecipCell) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<AccReal> dist(5, 15);
    for (int n = 0; n < 3; ++n) {
        Mat3 cell = Mat3::diagonal(dist(rng), dist(rng), dist(rng));
        Box b{cell, Vec3::zeros(), {true, true, true}};
        auto recip = b.recip_cell();
        auto prod = recip * cell;
        auto expected = Mat3::identity() * (AccReal(2) * ATV::units::pi);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                EXPECT_NEAR(prod(i, j), expected(i, j), kTol)
                    << "recip*cell != 2πI at (" << i << "," << j << ")";
    }
}

TEST(Box, Lengths) {
    auto box = Box::cubic(10);
    auto lens = box.lengths();
    EXPECT_NEAR(lens[0], AccReal(10), kTol);
    EXPECT_NEAR(lens[1], AccReal(10), kTol);
    EXPECT_NEAR(lens[2], AccReal(10), kTol);
}

TEST(Box, WrapBatch) {
    auto box = Box::cubic(10);
    AccReal xyz[12] = {15, -3, 22, -8, 5, 12, 0, 0, 0, 10, 10, 10};
    box.wrap_batch(xyz, 4);
    for (int i = 0; i < 4; ++i) {
        auto frac =
            box.to_frac(Vec3{xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]});
        EXPECT_GE(frac[0], AccReal(0));
        EXPECT_LT(frac[0], AccReal(1));
        EXPECT_GE(frac[1], AccReal(0));
        EXPECT_LT(frac[1], AccReal(1));
        EXPECT_GE(frac[2], AccReal(0));
        EXPECT_LT(frac[2], AccReal(1));
    }
}
