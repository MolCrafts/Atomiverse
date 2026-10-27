/// @file tests/cpu/scf/test_xc_cam_b3lyp_params.cpp
/// @brief CAM-B3LYP RSH parameter table (Yanai 2004 / PySCF defaults).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cpu/xc/xc_internal.h"
#include "oracle/cam_b3lyp_pins.h"

using namespace ATV::cpu::xc;

TEST(TestRshExchangeParams, CamB3lypYanaiSet) {
    const auto p =
        ATV::cpu::xc::rsh_exchange_params(ATV::cpu::xc::XCType::CAMB3LYP);
    EXPECT_DOUBLE_EQ(p.alpha, 0.19);
    EXPECT_DOUBLE_EQ(p.beta, 0.46);
    EXPECT_DOUBLE_EQ(p.mu, 0.33);
}

TEST(TestRshExchangeParams, PureFunctionalsAreNotRsh) {
    for (auto t : {ATV::cpu::xc::XCType::LSDA, ATV::cpu::xc::XCType::PBE,
                   ATV::cpu::xc::XCType::BLYP}) {
        const auto p = ATV::cpu::xc::rsh_exchange_params(t);
        EXPECT_DOUBLE_EQ(p.alpha, 0.0) << static_cast<int>(t);
        EXPECT_DOUBLE_EQ(p.beta, 0.0) << static_cast<int>(t);
        EXPECT_DOUBLE_EQ(p.mu, 0.0) << static_cast<int>(t);
    }
}

TEST(TestRshExchangeParams, B3lypStaysGlobalHybrid) {
    const auto p =
        ATV::cpu::xc::rsh_exchange_params(ATV::cpu::xc::XCType::B3LYP);
    EXPECT_DOUBLE_EQ(p.alpha, 0.20);
    EXPECT_DOUBLE_EQ(p.beta, 0.0);
    EXPECT_DOUBLE_EQ(p.mu, 0.0);
    EXPECT_DOUBLE_EQ(
        ATV::cpu::xc::hybrid_exchange_fraction(ATV::cpu::xc::XCType::B3LYP),
        0.20);
    EXPECT_DOUBLE_EQ(
        ATV::cpu::xc::hybrid_exchange_fraction(ATV::cpu::xc::XCType::CAMB3LYP),
        0.19);
}

TEST(TestRshExchangeParams, CamIsGgaNoPt2) {
    EXPECT_TRUE(is_gga(ATV::cpu::xc::XCType::CAMB3LYP));
    EXPECT_DOUBLE_EQ(
        ATV::cpu::xc::hybrid_pt2_fraction(ATV::cpu::xc::XCType::CAMB3LYP), 0.0);
}

TEST(TestCamB3lyp, StandardLibxc433EnergyAndAllFirstDerivatives) {
    // Both spin densities are positive and min/max >= 1e-6. Vacant-spin
    // and more extreme polarization comparisons remain in the diagnostic
    // JSON, where inherited LYP/VWN floors and libxc's zeta cancellation
    // are recorded separately from normal-domain qualification.
    for (std::size_t i = 0; i < cam_b3lyp_pin::points.size(); ++i) {
        const auto& p = cam_b3lyp_pin::points[i];
        const auto& x = p.input;
        const auto value = ATV::cpu::xc::evaluate_xc(
            ATV::cpu::xc::XCType::CAMB3LYP, x[0], x[1], x[2], x[3], x[4]);
        const std::array<double, 6> got = {
            value.energy_density, value.vrho_alpha, value.vrho_beta,
            value.vsigma_aa,      value.vsigma_ab,  value.vsigma_bb};
        for (std::size_t k = 0; k < got.size(); ++k) {
            EXPECT_TRUE(std::isfinite(got[k]));
            EXPECT_NEAR(got[k], p.output[k],
                        1e-11 * std::max(1.0, std::abs(p.output[k])))
                << "point=" << i << " channel=" << k;
        }
    }
}

TEST(TestCamB3lyp, SpinExchangeSymmetryIncludingEmptyChannel) {
    for (const auto& p : cam_b3lyp_pin::all_points) {
        const auto& x = p.input;
        const auto a = ATV::cpu::xc::evaluate_xc(ATV::cpu::xc::XCType::CAMB3LYP,
                                                 x[0], x[1], x[2], x[3], x[4]);
        const auto b = ATV::cpu::xc::evaluate_xc(ATV::cpu::xc::XCType::CAMB3LYP,
                                                 x[1], x[0], x[4], x[3], x[2]);
        const std::array<double, 6> first = {a.energy_density, a.vrho_alpha,
                                             a.vrho_beta,      a.vsigma_aa,
                                             a.vsigma_ab,      a.vsigma_bb};
        const std::array<double, 6> swapped = {b.energy_density, b.vrho_beta,
                                               b.vrho_alpha,     b.vsigma_bb,
                                               b.vsigma_ab,      b.vsigma_aa};
        for (std::size_t k = 0; k < first.size(); ++k)
            EXPECT_NEAR(first[k], swapped[k],
                        1e-12 * std::max(1.0, std::abs(first[k])));
    }
}

TEST(TestCamB3lyp, ShortRangeB88MatchesLibxcItyhIncludingDiluteSpin) {
    // Genuine spin-unpolarized ITYH calls recover the one-spin kernel
    // without the oracle's extreme-polarization 1-zeta cancellation.
    for (std::size_t i = 0; i < cam_b3lyp_pin::short_range_points.size(); ++i) {
        const auto& p = cam_b3lyp_pin::short_range_points[i];
        double energy, vrho, vsigma;
        b88_short_range_exchange_spin(p.input[0], p.input[1], 0.33, energy,
                                      vrho, vsigma);
        const std::array<double, 3> got = {energy, vrho, vsigma};
        for (std::size_t k = 0; k < got.size(); ++k)
            EXPECT_NEAR(got[k], p.output[k],
                        1e-11 * std::max(1.0, std::abs(p.output[k])))
                << "point=" << i << " channel=" << k;
    }
}

TEST(TestCamB3lyp, FirstDerivativesAreDerivativesOfTheSameEnergy) {
    // Auxiliary variational identity, alongside the genuine libxc pins.
    for (auto x : {std::array{0.3, 0.2, 0.02, 0.001, 0.01},
                   std::array{5e-5, 3e-5, 1e-10, 2e-11, 1e-10}}) {
        const auto out = ATV::cpu::xc::evaluate_xc(
            ATV::cpu::xc::XCType::CAMB3LYP, x[0], x[1], x[2], x[3], x[4]);
        const std::array<double, 5> derivative = {out.vrho_alpha, out.vrho_beta,
                                                  out.vsigma_aa, out.vsigma_ab,
                                                  out.vsigma_bb};
        for (std::size_t k = 0; k < x.size(); ++k) {
            auto xp = x, xm = x;
            const double h = x[k] * 1e-5;
            xp[k] += h;
            xm[k] -= h;
            const auto p =
                ATV::cpu::xc::evaluate_xc(ATV::cpu::xc::XCType::CAMB3LYP, xp[0],
                                          xp[1], xp[2], xp[3], xp[4]);
            const auto m =
                ATV::cpu::xc::evaluate_xc(ATV::cpu::xc::XCType::CAMB3LYP, xm[0],
                                          xm[1], xm[2], xm[3], xm[4]);
            EXPECT_NEAR((p.energy_density - m.energy_density) / (2 * h),
                        derivative[k],
                        2e-7 * std::max(1.0, std::abs(derivative[k])))
                << "variable=" << k;
        }
    }
}

TEST(TestCamB3lyp, ShortRangeB88ZeroMuAndEmptySpinLimits) {
    double full, vfull, sfull, sr, vsr, ssr;
    b88_exchange_spin(0.4, 0.3, full, vfull, sfull);
    b88_short_range_exchange_spin(0.4, 0.3, 0.0, sr, vsr, ssr);
    EXPECT_DOUBLE_EQ(sr, full);
    EXPECT_DOUBLE_EQ(vsr, vfull);
    EXPECT_DOUBLE_EQ(ssr, sfull);
    b88_short_range_exchange_spin(0.0, 0.0, 0.33, sr, vsr, ssr);
    EXPECT_DOUBLE_EQ(sr, 0.0);
    EXPECT_DOUBLE_EQ(vsr, 0.0);
    EXPECT_DOUBLE_EQ(ssr, 0.0);
}

TEST(TestCamB3lyp, LargeMuAnalyticDensityLimitRetainsSmallSigmaDerivative) {
    double energy, vrho, vsigma;
    constexpr double rho = 0.4, mu = 1e6;
    b88_short_range_exchange_spin(rho, 0.3, mu, energy, vrho, vsigma);
    // A(a) ~ 1/(36 a^2), so f_SR ~ -pi*rho^2/(2 mu^2),
    // independent of the B88 enhancement to leading order.
    EXPECT_NEAR(energy * mu * mu, -ATV::units::pi * rho * rho / 2.0, 1e-10);
    EXPECT_NEAR(vrho * mu * mu, -ATV::units::pi * rho, 1e-10);
    EXPECT_LT(std::abs(vsigma), 1e-20);
}

#endif // ATV_HAS_HF
