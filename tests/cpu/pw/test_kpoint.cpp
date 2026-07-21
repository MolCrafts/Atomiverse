/// @file test_kpoint.cpp
#include "atomiverse/cpu/pw/internal/kpoint.h"
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(KPoint, MP_4x4x4) {
    auto kg = generate_mp_mesh(4, 4, 4);
    EXPECT_EQ(kg.kpoints.size(), 64u);
    AccReal sum = 0;
    for (auto w : kg.weights)
        sum += w;
    EXPECT_NEAR(static_cast<double>(sum), 1.0, 1e-14);
}

TEST(KPoint, MP_1x1x1) {
    auto kg = generate_mp_mesh(1, 1, 1);
    EXPECT_EQ(kg.kpoints.size(), 1u);
    EXPECT_NEAR(static_cast<double>(kg.weights[0]), 1.0, 1e-14);
    EXPECT_NEAR(static_cast<double>(kg.kpoints[0][0]), 0.0, 1e-12);
}

TEST(KPoint, Shifted) {
    auto kg = generate_mp_mesh(2, 2, 2, 0.5, 0.5, 0.5);
    EXPECT_EQ(kg.kpoints.size(), 8u);
    // Shifted MP grid: k_i = (i + shift) / n. For shift=0.5, n=2: values are {0.25, 0.75}
    for (auto& k : kg.kpoints) {
        for (int i = 0; i < 3; ++i) {
            double v = static_cast<double>(k[i]);
            bool ok = std::abs(v - 0.25) < 1e-12 || std::abs(v - 0.75) < 1e-12;
            EXPECT_TRUE(ok) << "k[" << i << "]=" << v << " not 0.25 or 0.75";
        }
    }
}
