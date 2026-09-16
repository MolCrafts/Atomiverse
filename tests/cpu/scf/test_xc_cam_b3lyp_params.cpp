/// @file tests/cpu/scf/test_xc_cam_b3lyp_params.cpp
/// @brief CAM-B3LYP RSH parameter table (Yanai 2004 / PySCF defaults).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV::cpu::scf;

TEST(TestRshExchangeParams, CamB3lypYanaiSet) {
    const auto p = rsh_exchange_params(XCType::CAMB3LYP);
    EXPECT_DOUBLE_EQ(p.alpha, 0.19);
    EXPECT_DOUBLE_EQ(p.beta, 0.46);
    EXPECT_DOUBLE_EQ(p.mu, 0.33);
}

TEST(TestRshExchangeParams, PureFunctionalsAreNotRsh) {
    for (auto t : {XCType::LSDA, XCType::PBE, XCType::BLYP}) {
        const auto p = rsh_exchange_params(t);
        EXPECT_DOUBLE_EQ(p.alpha, 0.0) << static_cast<int>(t);
        EXPECT_DOUBLE_EQ(p.beta, 0.0) << static_cast<int>(t);
        EXPECT_DOUBLE_EQ(p.mu, 0.0) << static_cast<int>(t);
    }
}

TEST(TestRshExchangeParams, B3lypStaysGlobalHybrid) {
    const auto p = rsh_exchange_params(XCType::B3LYP);
    EXPECT_DOUBLE_EQ(p.alpha, 0.20);
    EXPECT_DOUBLE_EQ(p.beta, 0.0);
    EXPECT_DOUBLE_EQ(p.mu, 0.0);
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::B3LYP), 0.20);
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::CAMB3LYP), 0.19);
}

TEST(TestRshExchangeParams, CamIsGgaNoPt2) {
    EXPECT_TRUE(is_gga(XCType::CAMB3LYP));
    EXPECT_DOUBLE_EQ(hybrid_pt2_fraction(XCType::CAMB3LYP), 0.0);
}

#endif // ATV_HAS_HF
