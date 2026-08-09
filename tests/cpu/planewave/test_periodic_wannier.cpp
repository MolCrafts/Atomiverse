/// @file test_periodic_wannier.cpp
/// @brief Unit tests for periodic MLWF pure math (bond midpoints, SMV
///        weights, phase→center). No full SCF (that lives in
///        regressions/pw/periodic-wannier-si).

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/periodic_wannier.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

TEST(PeriodicWannierMath, DiamondBondMidpoints) {
    const double a = 5.43;
    const auto m = diamond_bond_midpoints(a);
    const double s = a / 8.0;
    EXPECT_NEAR(m[0][0], s, 1e-12);
    EXPECT_NEAR(m[0][1], s, 1e-12);
    EXPECT_NEAR(m[0][2], s, 1e-12);
    EXPECT_NEAR(m[1][0], s, 1e-12);
    EXPECT_NEAR(m[1][1], -s, 1e-12);
    EXPECT_NEAR(m[1][2], -s, 1e-12);
    EXPECT_NEAR(m[2][0], -s, 1e-12);
    EXPECT_NEAR(m[2][1], s, 1e-12);
    EXPECT_NEAR(m[2][2], -s, 1e-12);
    EXPECT_NEAR(m[3][0], -s, 1e-12);
    EXPECT_NEAR(m[3][1], -s, 1e-12);
    EXPECT_NEAR(m[3][2], s, 1e-12);
    EXPECT_THROW(diamond_bond_midpoints(0.0), std::invalid_argument);
    EXPECT_THROW(diamond_bond_midpoints(-1.0), std::invalid_argument);
}

TEST(PeriodicWannierMath, CubicSmvWeight) {
    const double b = 0.5; // 1/Å
    EXPECT_NEAR(mv_pair_weight_cubic(b), 1.0 / (2.0 * b * b), 1e-14);
    EXPECT_THROW(mv_pair_weight_cubic(0.0), std::invalid_argument);
    EXPECT_THROW(mv_pair_weight_cubic(-0.1), std::invalid_argument);
}

TEST(PeriodicWannierMath, CenterFromDiagonalPhases) {
    // Synthetic cubic shell: b = (±b,0,0), (0,±b,0), (0,0,±b).
    // M = exp(-i b · r0) with r0 = (0.3, -0.2, 0.5) Å.
    const double b = 1.2;
    const double wb = mv_pair_weight_cubic(b);
    const std::array<double, 3> r0{0.3, -0.2, 0.5};
    std::vector<std::array<double, 3>> b_cart = {
        {{b, 0, 0}},  {{-b, 0, 0}}, {{0, b, 0}},
        {{0, -b, 0}}, {{0, 0, b}},  {{0, 0, -b}},
    };
    std::vector<double> w(6, wb);
    std::vector<std::complex<double>> M(6);
    for (int i = 0; i < 6; ++i) {
        const double phase = b_cart[static_cast<std::size_t>(i)][0] * r0[0]
                             + b_cart[static_cast<std::size_t>(i)][1] * r0[1]
                             + b_cart[static_cast<std::size_t>(i)][2] * r0[2];
        M[static_cast<std::size_t>(i)] = std::exp(std::complex<double>{0, -phase});
    }
    const auto r = center_from_diagonal_phases(b_cart, w, M);
    EXPECT_NEAR(r[0], r0[0], 1e-9);
    EXPECT_NEAR(r[1], r0[1], 1e-9);
    EXPECT_NEAR(r[2], r0[2], 1e-9);

    EXPECT_THROW(center_from_diagonal_phases({}, {}, {}),
                 std::invalid_argument);
}

TEST(PeriodicWannierMath, InvalidArgumentsOnCompute) {
    PeriodicWannier mlwf;
    mlwf.k_mesh = 0;
    PW method;
    method.gth_name = "GTH-PBE-q1";
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.box = cpu::Box::cubic(5.0);
    EXPECT_THROW(mlwf.compute(method, f), std::invalid_argument);
    mlwf.k_mesh = 2;
    mlwf.tol = 0.0;
    EXPECT_THROW(mlwf.compute(method, f), std::invalid_argument);
}

TEST(PeriodicWannierMath, SpinPolarizedGateNoLongerFootholdThrow) {
    // Collinear spin is a product path: the foothold message is gone.
    // Invalid k_mesh still fails before SCF; metallic smearing still rejected.
    PeriodicWannier mlwf;
    mlwf.k_mesh = 0;
    PW method;
    method.gth_name = "GTH-PBE-q1";
    method.spin_polarized = true;
    method.multiplicity = 1;
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.box = cpu::Box::cubic(5.0);
    try {
        mlwf.compute(method, f);
        FAIL() << "expected invalid_argument for k_mesh=0";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("spin-polarized MLWF not supported"),
                  std::string::npos);
        EXPECT_NE(msg.find("k_mesh"), std::string::npos);
    }
    mlwf.k_mesh = 2;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.01;
    EXPECT_THROW(mlwf.compute(method, f), std::invalid_argument);
}
