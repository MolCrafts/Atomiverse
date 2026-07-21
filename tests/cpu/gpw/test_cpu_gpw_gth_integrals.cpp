/// @file tests/cpu/gpw/test_cpu_gpw_gth_integrals.cpp
/// @brief Validation of analytic GTH matrix elements (GPW spec D3).
///
/// The integral machinery is checked against exact closed forms (derivable by
/// hand for same-center s-Gaussians), against the Gaussian-product symmetry
/// <a|O|b> == <b|O|a>, and for finiteness / correct limits. Full GTH-convention
/// parity (normalization, solid-harmonic sign conventions) is validated later
/// against CP2K at GPW milestone M1; here we prove the integral bookkeeping.

#include <cmath>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/gth_integrals.h"

using ATV::cpu::gpw::gth_local_sr;
using ATV::cpu::gpw::gth_nonlocal;
using ATV::cpu::gpw::gth_projector_overlap;
using ATV::cpu::gpw::GthNonlocalChannel;
using ATV::cpu::gpw::Prim;

namespace {
constexpr double kPi = 3.14159265358979323846;
const double kOrigin[3] = {0.0, 0.0, 0.0};

Prim s_prim(double alpha, double x, double y, double z) {
    Prim p;
    p.alpha = alpha;
    p.center[0] = x;
    p.center[1] = y;
    p.center[2] = z;
    p.l[0] = p.l[1] = p.l[2] = 0;
    return p;
}
} // namespace

// Local SR, constant term only: <g_a|C1 exp(-|r-D|^2/(2 rloc^2))|g_b> for two
// s-Gaussians at the origin with D at the origin = C1 (pi/gamma)^{3/2},
// gamma = alpha_a + alpha_b + 1/(2 rloc^2).
TEST(GthIntegrals, LocalConstantClosedForm) {
    const double aa = 0.8, ab = 1.3, rloc = 0.5, C1 = 0.7;
    const double beta = 1.0 / (2.0 * rloc * rloc);
    const double gamma = aa + ab + beta;
    const double expected = C1 * std::pow(kPi / gamma, 1.5);

    const double got = gth_local_sr(s_prim(aa, 0, 0, 0), s_prim(ab, 0, 0, 0),
                                    kOrigin, rloc, {C1});
    EXPECT_NEAR(got, expected, 1e-11);
}

// Local SR, first moment term: V = C2 (|r-D|^2/rloc^2) exp(-|r-D|^2/(2rloc^2)).
// Closed form for same-center s-Gaussians: C2/rloc^2 * 3 (pi/gamma)^{3/2}/(2 gamma).
TEST(GthIntegrals, LocalMomentClosedForm) {
    const double aa = 0.8, ab = 1.3, rloc = 0.5, C2 = 0.9;
    const double beta = 1.0 / (2.0 * rloc * rloc);
    const double gamma = aa + ab + beta;
    const double expected =
        C2 / (rloc * rloc) * 3.0 * std::pow(kPi / gamma, 1.5) / (2.0 * gamma);

    const double got = gth_local_sr(s_prim(aa, 0, 0, 0), s_prim(ab, 0, 0, 0),
                                    kOrigin, rloc, {0.0, C2});
    EXPECT_NEAR(got, expected, 1e-11);
}

// Gaussian-product symmetry: <a|V_sr|b> == <b|V_sr|a>, off-center, mixed l.
TEST(GthIntegrals, LocalSymmetry) {
    Prim a = s_prim(0.9, 0.1, 0.0, 0.0);
    a.l[0] = 1; // p_x on a
    Prim b = s_prim(1.1, 0.0, 0.2, -0.1);
    const double D[3] = {0.05, 0.0, 0.0};
    const std::vector<double> C = {-3.2, 0.8};

    const double ab = gth_local_sr(a, b, D, 0.45, C);
    const double ba = gth_local_sr(b, a, D, 0.45, C);
    EXPECT_NEAR(ab, ba, 1e-12);
    EXPECT_TRUE(std::isfinite(ab));
}

// Projector overlap l=0, i=1: same-center s-Gaussian closed form
// N (1/sqrt(4pi)) (pi/(alpha+beta))^{3/2}, beta = 1/(2 r_l^2),
// N = sqrt(2)/(r_l^{3/2} sqrt(Gamma(3/2))).
TEST(GthIntegrals, ProjectorL0ClosedForm) {
    const double alpha = 1.4, r_l = 0.5;
    const double beta = 1.0 / (2.0 * r_l * r_l);
    const double N =
        std::sqrt(2.0) / (std::pow(r_l, 1.5) * std::sqrt(std::tgamma(1.5)));
    const double expected =
        N / std::sqrt(4.0 * kPi) * std::pow(kPi / (alpha + beta), 1.5);

    const double got =
        gth_projector_overlap(s_prim(alpha, 0, 0, 0), kOrigin, 0, 1, r_l, 0);
    EXPECT_NEAR(got, expected, 1e-11);
}

// Nonlocal channel: single l=0 projector, a==b -> h00 * overlap^2 (positive for
// h00 > 0), and the result is symmetric under a<->b.
TEST(GthIntegrals, NonlocalL0PositiveAndSymmetric) {
    GthNonlocalChannel ch;
    ch.l = 0;
    ch.r = 0.5;
    ch.nproj = 1;
    ch.h = {2.5};

    Prim g = s_prim(1.2, 0.1, -0.2, 0.0);
    const double D[3] = {0.0, 0.0, 0.0};

    const double ov = gth_projector_overlap(g, D, 0, 1, ch.r, 0);
    const double diag = gth_nonlocal(g, g, D, ch);
    EXPECT_GT(diag, 0.0);
    EXPECT_NEAR(diag, 2.5 * ov * ov, 1e-12);

    Prim g2 = s_prim(0.7, 0.0, 0.3, 0.1);
    EXPECT_NEAR(gth_nonlocal(g, g2, D, ch), gth_nonlocal(g2, g, D, ch), 1e-12);
}

// Zero coupling -> zero contribution.
TEST(GthIntegrals, NonlocalZeroCouplingIsZero) {
    GthNonlocalChannel ch;
    ch.l = 0;
    ch.r = 0.5;
    ch.nproj = 1;
    ch.h = {0.0};
    Prim g = s_prim(1.2, 0.1, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(gth_nonlocal(g, g, kOrigin, ch), 0.0);
}

// l=1 channel with a 2x2 h matrix: finite, non-NaN, and symmetric.
TEST(GthIntegrals, NonlocalL1FiniteAndSymmetric) {
    GthNonlocalChannel ch;
    ch.l = 1;
    ch.r = 0.45;
    ch.nproj = 2;
    ch.h = {1.3, -0.4, -0.4, 0.9}; // symmetric 2x2

    Prim a = s_prim(0.9, 0.1, 0.0, 0.0);
    a.l[0] = 1;
    Prim b = s_prim(1.1, 0.0, 0.15, 0.0);
    const double D[3] = {0.0, 0.0, 0.0};

    const double ab = gth_nonlocal(a, b, D, ch);
    const double ba = gth_nonlocal(b, a, D, ch);
    EXPECT_TRUE(std::isfinite(ab));
    EXPECT_NEAR(ab, ba, 1e-12);
}
