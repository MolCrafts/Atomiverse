/// @file test_mat3.cpp
/// @brief Unit tests for Mat3 OOP class (xtensor_fixed rewrite).

#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/vec3.h"

using namespace ATV;

namespace {
constexpr AccReal kTol = 1e-8;
}

TEST(Mat3, ZerosIsAllZero) {
    auto m = Mat3::zeros();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(m(i, j), AccReal(0));
}

TEST(Mat3, Identity) {
    auto m = Mat3::identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j)
                EXPECT_EQ(m(i, j), AccReal(1));
            else
                EXPECT_EQ(m(i, j), AccReal(0));
        }
    }
}

TEST(Mat3, Diagonal) {
    auto m = Mat3::diagonal(1, 2, 3);
    EXPECT_EQ(m(0, 0), AccReal(1));
    EXPECT_EQ(m(0, 1), AccReal(0));
    EXPECT_EQ(m(0, 2), AccReal(0));
    EXPECT_EQ(m(1, 0), AccReal(0));
    EXPECT_EQ(m(1, 1), AccReal(2));
    EXPECT_EQ(m(1, 2), AccReal(0));
    EXPECT_EQ(m(2, 0), AccReal(0));
    EXPECT_EQ(m(2, 1), AccReal(0));
    EXPECT_EQ(m(2, 2), AccReal(3));
}

TEST(Mat3, Addition) {
    auto a = Mat3::diagonal(1, 2, 3);
    auto b = Mat3::diagonal(3, 2, 1);
    auto s = a + b;
    EXPECT_EQ(s(0, 0), AccReal(4));
    EXPECT_EQ(s(1, 1), AccReal(4));
    EXPECT_EQ(s(2, 2), AccReal(4));
}

TEST(Mat3, Subtraction) {
    auto a = Mat3::diagonal(5, 5, 5);
    auto b = Mat3::diagonal(2, 2, 2);
    auto d = a - b;
    EXPECT_EQ(d(0, 0), AccReal(3));
    EXPECT_EQ(d(1, 1), AccReal(3));
    EXPECT_EQ(d(2, 2), AccReal(3));
}

TEST(Mat3, ScalarMultiply) {
    auto m = Mat3::identity() * AccReal(3);
    EXPECT_EQ(m(0, 0), AccReal(3));
    EXPECT_EQ(m(1, 1), AccReal(3));
    EXPECT_EQ(m(2, 2), AccReal(3));
    EXPECT_EQ(m(0, 1), AccReal(0));
}

TEST(Mat3, ScalarDivide) {
    auto m = Mat3::diagonal(6, 6, 6) / AccReal(2);
    EXPECT_EQ(m(0, 0), AccReal(3));
}

TEST(Mat3, Trace) {
    EXPECT_EQ(Mat3::diagonal(1, 2, 3).trace(), AccReal(6));
    EXPECT_EQ(Mat3::identity().trace(), AccReal(3));
    EXPECT_EQ(Mat3::zeros().trace(), AccReal(0));
}

TEST(Mat3, Determinant) {
    EXPECT_NEAR(Mat3::identity().det(), AccReal(1), kTol);
    EXPECT_NEAR(Mat3::diagonal(2, 3, 4).det(), AccReal(24), kTol);
    EXPECT_NEAR(Mat3::zeros().det(), AccReal(0), kTol);
}

TEST(Mat3, DeterminantRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<AccReal> dist(-10, 10);
    for (int n = 0; n < 5; ++n) {
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = dist(rng);
        // cofactor expansion
        AccReal ref = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                      - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                      + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        EXPECT_NEAR(m.det(), ref, kTol)
            << "det mismatch for random matrix " << n;
    }
}

TEST(Mat3, InverseTimesIdentity) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<AccReal> dist(-5, 5);
    for (int n = 0; n < 5; ++n) {
        Mat3 m;
        m(0, 0) = dist(rng) + AccReal(6); // ensure non-singular
        m(1, 1) = dist(rng) + AccReal(6);
        m(2, 2) = dist(rng) + AccReal(6);
        m(0, 1) = dist(rng);
        m(0, 2) = dist(rng);
        m(1, 0) = dist(rng);
        m(1, 2) = dist(rng);
        m(2, 0) = dist(rng);
        m(2, 1) = dist(rng);

        auto inv = m.inverse();
        auto prod = m * inv;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                AccReal expected = (i == j) ? AccReal(1) : AccReal(0);
                EXPECT_NEAR(prod(i, j), expected, kTol)
                    << "I - M*inv(M) at (" << i << "," << j << ") for matrix "
                    << n;
            }
        }
    }
}

TEST(Mat3, Transpose) {
    Mat3 m;
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 4;
    m(1, 1) = 5;
    m(1, 2) = 6;
    m(2, 0) = 7;
    m(2, 1) = 8;
    m(2, 2) = 9;
    auto t = m.transpose();
    EXPECT_EQ(t(0, 0), AccReal(1));
    EXPECT_EQ(t(0, 1), AccReal(4));
    EXPECT_EQ(t(0, 2), AccReal(7));
    EXPECT_EQ(t(1, 0), AccReal(2));
    EXPECT_EQ(t(1, 1), AccReal(5));
    EXPECT_EQ(t(1, 2), AccReal(8));
    EXPECT_EQ(t(2, 0), AccReal(3));
    EXPECT_EQ(t(2, 1), AccReal(6));
    EXPECT_EQ(t(2, 2), AccReal(9));
}

TEST(Mat3, MatrixMultiply) {
    auto a = Mat3::diagonal(1, 2, 3);
    auto b = Mat3::diagonal(4, 5, 6);
    auto c = a * b;
    EXPECT_EQ(c(0, 0), AccReal(4));
    EXPECT_EQ(c(1, 1), AccReal(10));
    EXPECT_EQ(c(2, 2), AccReal(18));
}

TEST(Mat3, Mat3TimesVec3) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<AccReal> dist(-5, 5);
    for (int n = 0; n < 5; ++n) {
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = dist(rng);
        Vec3 v{dist(rng), dist(rng), dist(rng)};
        auto r = m * v;
        for (int i = 0; i < 3; ++i) {
            AccReal ref = m(i, 0) * v[0] + m(i, 1) * v[1] + m(i, 2) * v[2];
            EXPECT_NEAR(r[i], ref, kTol) << "M*v mismatch at component " << i;
        }
    }
}

TEST(Mat3, FrobeniusNorm) {
    auto m = Mat3::diagonal(3, 4, 0);
    AccReal expected = std::sqrt(AccReal(9 + 16));
    EXPECT_NEAR(m.norm(), expected, kTol);
}

TEST(Mat3, DefaultConstructIsZero) {
    Mat3 m;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(m(i, j), AccReal(0));
}
