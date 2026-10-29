// Uniform-density quadrature is exactly volume times one XC point value.
// This component test has no SCF and does not use any grid sum as its oracle.
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <string>

using namespace ATV;
using namespace ATV::cpu::pw::internal;
namespace {
struct Case {
    cpu::xc::XCType xc;
    bool spin;
    const char* name;
};
class PwXcEnergyPrecision : public ::testing::TestWithParam<Case> {};

void constant_potential(const xt::xarray<AccReal>& values, AccReal expected) {
    bool finite = true;
    AccReal maximum_error = 0;
    for (const auto value : values) {
        finite = finite && std::isfinite(value);
        maximum_error = std::max(maximum_error, std::abs(value - expected));
    }
    EXPECT_TRUE(finite);
    EXPECT_LE(maximum_error, 1e-11);
}

TEST_P(PwXcEnergyPrecision, UniformIntegralMatchesSinglePointRoundingBound) {
    const auto test = GetParam();
    PWWorkspace ws;
    ws.fft_nx = ws.fft_ny = ws.fft_nz = 28;
    ws.fft = std::make_unique<FFTPlan3D>(28, 28, 28);
    ws.nr = ws.fft->nr();
    ws.nc = ws.fft->nc();
    ws.nspin = test.spin ? 2 : 1;
    ws.omega = 1000;
    ws.gvec0.g = {{0, 0, 0}};
    ws.gvec0.h = ws.gvec0.k = ws.gvec0.l = {0};
    ws.gvec0.ekin = {0};
    ws.gvec0.nx = ws.gvec0.ny = ws.gvec0.nz = 28;
    ws.gvec = &ws.gvec0;
    ws.n_g = 1;
    ws.rho_G = xt::xarray<std::complex<AccReal>>{std::complex<AccReal>{.05, 0}};
    const AccReal alpha = test.spin ? .037 : .025;
    const AccReal beta = test.spin ? .013 : .025;
    if (test.spin) {
        ws.rho_spin_G.resize(2);
        ws.rho_spin_G[0] =
            xt::xarray<std::complex<AccReal>>{std::complex<AccReal>{alpha, 0}};
        ws.rho_spin_G[1] =
            xt::xarray<std::complex<AccReal>>{std::complex<AccReal>{beta, 0}};
    }
    // One admitted positive-density point, with exactly zero gradients.
    const auto point = cpu::xc::evaluate_xc(test.xc, alpha, beta, 0, 0, 0);
    const AccReal exact_integral = ws.omega * point.energy_density;
    const AccReal bound = 8 * std::numeric_limits<AccReal>::epsilon()
                          * std::max(AccReal(1), std::abs(exact_integral));
    XCBuilder builder;
    builder.prepare(ws); // Retain its existing automatic padded quadrature.
    if (test.spin)
        builder.step_spin(ws, test.xc);
    else
        builder.step(ws, test.xc);
    ASSERT_TRUE(std::isfinite(ws.E_xc));
    EXPECT_LE(std::abs(ws.E_xc - exact_integral), bound)
        << "actual=" << ws.E_xc << " single-point integral=" << exact_integral;
    if (test.spin) {
        ASSERT_EQ(ws.V_xc_spin_G.size(), 2u);
        for (int channel = 0; channel < 2; ++channel) {
            const auto expected = channel ? point.vrho_beta : point.vrho_alpha;
            ASSERT_EQ(ws.V_xc_spin_r[channel].size(),
                      static_cast<std::size_t>(ws.nr));
            constant_potential(ws.V_xc_spin_r[channel], expected);
            ASSERT_EQ(ws.V_xc_spin_G[channel].size(), 1u);
            EXPECT_NEAR(ws.V_xc_spin_G[channel][0].real(), expected, 1e-11);
            EXPECT_NEAR(ws.V_xc_spin_G[channel][0].imag(), 0, 1e-11);
        }
    } else {
        ASSERT_EQ(ws.V_xc_r.size(), static_cast<std::size_t>(ws.nr));
        constant_potential(ws.V_xc_r, point.vrho_alpha);
        ASSERT_EQ(ws.V_xc_G.size(), 1u);
        EXPECT_NEAR(ws.V_xc_G[0].real(), point.vrho_alpha, 1e-11);
        EXPECT_NEAR(ws.V_xc_G[0].imag(), 0, 1e-11);
    }
}

INSTANTIATE_TEST_SUITE_P(
    UniformRksAndUks, PwXcEnergyPrecision,
    ::testing::Values(Case{cpu::xc::XCType::LSDA, false, "RksLsda"},
                      Case{cpu::xc::XCType::PBE, false, "RksPbe"},
                      Case{cpu::xc::XCType::PBEsol, false, "RksPbesol"},
                      Case{cpu::xc::XCType::RevPBE, false, "RksRevPbe"},
                      Case{cpu::xc::XCType::RPBE, false, "RksRpbe"},
                      Case{cpu::xc::XCType::LSDA, true, "UksLsda"},
                      Case{cpu::xc::XCType::PBE, true, "UksPbe"},
                      Case{cpu::xc::XCType::PBEsol, true, "UksPbesol"},
                      Case{cpu::xc::XCType::RevPBE, true, "UksRevPbe"},
                      Case{cpu::xc::XCType::RPBE, true, "UksRpbe"}),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });
} // namespace
