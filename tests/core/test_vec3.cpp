/// @file test_vec3.cpp
/// @brief Unit tests for Vec3 OOP class.

#include <gtest/gtest.h>

#include <random>

#include "atomiverse/core/vec3.h"

using namespace ATV;

namespace {
constexpr AccReal kTol = 1e-12;
}

TEST(Vec3, Zeros) {
    auto v = Vec3::zeros();
    EXPECT_EQ(v[0], AccReal(0));
    EXPECT_EQ(v[1], AccReal(0));
    EXPECT_EQ(v[2], AccReal(0));
}

TEST(Vec3, UnitVectors) {
    EXPECT_EQ(Vec3::unit_x()[0], AccReal(1));
    EXPECT_EQ(Vec3::unit_x()[1], AccReal(0));
    EXPECT_EQ(Vec3::unit_x()[2], AccReal(0));

    EXPECT_EQ(Vec3::unit_y()[0], AccReal(0));
    EXPECT_EQ(Vec3::unit_y()[1], AccReal(1));
    EXPECT_EQ(Vec3::unit_y()[2], AccReal(0));

    EXPECT_EQ(Vec3::unit_z()[0], AccReal(0));
    EXPECT_EQ(Vec3::unit_z()[1], AccReal(0));
    EXPECT_EQ(Vec3::unit_z()[2], AccReal(1));
}

TEST(Vec3, ConstructValues) {
    Vec3 v{1, 2, 3};
    EXPECT_EQ(v[0], AccReal(1));
    EXPECT_EQ(v[1], AccReal(2));
    EXPECT_EQ(v[2], AccReal(3));
}

TEST(Vec3, Addition) {
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    auto c = a + b;
    EXPECT_EQ(c[0], AccReal(5));
    EXPECT_EQ(c[1], AccReal(7));
    EXPECT_EQ(c[2], AccReal(9));
}

TEST(Vec3, Subtraction) {
    Vec3 a{10, 10, 10}, b{3, 4, 5};
    auto c = a - b;
    EXPECT_EQ(c[0], AccReal(7));
    EXPECT_EQ(c[1], AccReal(6));
    EXPECT_EQ(c[2], AccReal(5));
}

TEST(Vec3, ScalarMultiply) {
    Vec3 v{1, 2, 3};
    auto r = v * AccReal(2);
    EXPECT_EQ(r[0], AccReal(2));
    EXPECT_EQ(r[1], AccReal(4));
    EXPECT_EQ(r[2], AccReal(6));
}

TEST(Vec3, ScalarDivide) {
    Vec3 v{6, 8, 10};
    auto r = v / AccReal(2);
    EXPECT_EQ(r[0], AccReal(3));
    EXPECT_EQ(r[1], AccReal(4));
    EXPECT_EQ(r[2], AccReal(5));
}

TEST(Vec3, DotProduct) {
    Vec3 a{1, 2, 3}, b{4, -5, 6};
    AccReal expected = 1 * 4 + 2 * (-5) + 3 * 6; // = 12
    EXPECT_NEAR(a.dot(b), expected, kTol);
}

TEST(Vec3, DotProductRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<AccReal> dist(-10, 10);
    for (int n = 0; n < 5; ++n) {
        Vec3 u{dist(rng), dist(rng), dist(rng)};
        Vec3 v{dist(rng), dist(rng), dist(rng)};
        AccReal ref = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        EXPECT_NEAR(u.dot(v), ref, kTol);
    }
}

TEST(Vec3, CrossProduct) {
    auto u = Vec3::unit_x();
    auto v = Vec3::unit_y();
    auto c = u.cross(v);
    EXPECT_NEAR(c[0], AccReal(0), kTol);
    EXPECT_NEAR(c[1], AccReal(0), kTol);
    EXPECT_NEAR(c[2], AccReal(1), kTol);
}

TEST(Vec3, CrossProductRandom) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<AccReal> dist(-10, 10);
    for (int n = 0; n < 5; ++n) {
        Vec3 u{dist(rng), dist(rng), dist(rng)};
        Vec3 v{dist(rng), dist(rng), dist(rng)};
        auto c = u.cross(v);
        AccReal cx = u[1] * v[2] - u[2] * v[1];
        AccReal cy = u[2] * v[0] - u[0] * v[2];
        AccReal cz = u[0] * v[1] - u[1] * v[0];
        EXPECT_NEAR(c[0], cx, kTol);
        EXPECT_NEAR(c[1], cy, kTol);
        EXPECT_NEAR(c[2], cz, kTol);
        // u × u = 0
        EXPECT_NEAR(u.cross(u).norm(), AccReal(0), kTol);
    }
}

TEST(Vec3, Norm) {
    Vec3 v{3, 4, 0};
    EXPECT_NEAR(v.norm(), AccReal(5), kTol);
    EXPECT_NEAR(v.norm_squared(), AccReal(25), kTol);
}

TEST(Vec3, Normalized) {
    std::mt19937 rng(456);
    std::uniform_real_distribution<AccReal> dist(1, 10);
    for (int n = 0; n < 5; ++n) {
        Vec3 v{dist(rng), dist(rng), dist(rng)};
        auto nv = v.normalized();
        EXPECT_NEAR(nv.norm(), AccReal(1), kTol);
        // direction preserved
        EXPECT_NEAR(nv[0] * v.norm(), v[0], kTol);
        EXPECT_NEAR(nv[1] * v.norm(), v[1], kTol);
        EXPECT_NEAR(nv[2] * v.norm(), v[2], kTol);
    }
}

TEST(Vec3, Accessors) {
    Vec3 v{1, 2, 3};
    EXPECT_EQ(v.x(), AccReal(1));
    EXPECT_EQ(v.y(), AccReal(2));
    EXPECT_EQ(v.z(), AccReal(3));
}
