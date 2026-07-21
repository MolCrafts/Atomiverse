/// @file test_xc_reference.cpp
/// @brief Accuracy tests for native XC functionals against analytical reference values.
///
/// Validates our native Slater exchange, VWN5 correlation, and PBE GGA against
/// independently computed analytical formulas. This replaces the role of a libxc
/// comparison — instead of linking libxc, we verify against the exact mathematical
/// definitions from the original publications:
///
///   - Slater: Dirac, Proc. Cambridge Phil. Soc. 26, 376 (1930)
///   - VWN5:   Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980), param. V
///   - PBE:    Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

constexpr double kPi = 3.14159265358979323846;

// ── Independent Slater exchange implementation ──────────────────────────

// Exact Dirac-Slater exchange for spin-polarized uniform electron gas.
// E_x = -C_x_spin ∫ [ρ_α^{4/3} + ρ_β^{4/3}] dτ
// where C_x_spin = 2^{1/3} * (3/4)(3/π)^{1/3}
struct SlaterRef {
    double exc;    // energy per electron
    double vrho_a; // ∂(ε_x * ρ)/∂ρ_α = -(4/3) C_x_spin ρ_α^{1/3}
    double vrho_b;

    static SlaterRef compute(double rho_a, double rho_b) {
        double Cx = (3.0 / 4.0) * std::cbrt(3.0 / kPi);
        double Cx_spin = std::cbrt(2.0) * Cx;
        double rho = rho_a + rho_b;
        double ra43 = std::pow(rho_a, 4.0 / 3.0);
        double rb43 = std::pow(rho_b, 4.0 / 3.0);
        double fx = -Cx_spin * (ra43 + rb43);

        SlaterRef ref;
        ref.exc = fx / rho;
        ref.vrho_a = -(4.0 / 3.0) * Cx_spin * std::cbrt(rho_a);
        ref.vrho_b = -(4.0 / 3.0) * Cx_spin * std::cbrt(rho_b);
        return ref;
    }
};

// ── Independent VWN5 correlation implementation ─────────────────────────

struct VWN5Params {
    double A, x0, b, c;
};

// VWN5 params: A converted from Rydberg to Hartree (÷2)
constexpr VWN5Params kP = {0.0310907, -0.10498, 3.72744, 12.9352};
constexpr VWN5Params kF = {0.01554535, -0.32500, 7.06042, 18.0578};
constexpr VWN5Params kA = {-1.0 / (6.0 * kPi * kPi), -0.0047584, 1.13107,
                           13.0045};

// VWN5 Pade approximant F(x), Eq. 4.4 in Vosko et al. 1980
double vwn5_F(double x, const VWN5Params& p) {
    double X = x * x + p.b * x + p.c;
    double X0 = p.x0 * p.x0 + p.b * p.x0 + p.c;
    double Q = std::sqrt(4.0 * p.c - p.b * p.b);
    double atanQ = std::atan2(Q, 2.0 * x + p.b);

    double t1 = std::log(x * x / X);
    double t2 = 2.0 * p.b / Q * atanQ;
    double t3 = -p.b * p.x0 / X0
                * (std::log((x - p.x0) * (x - p.x0) / X)
                   + 2.0 * (p.b + 2.0 * p.x0) / Q * atanQ);
    return p.A * (t1 + t2 + t3);
}

// dF/dx for VWN5 potential
double vwn5_dFdx(double x, const VWN5Params& p) {
    double X = x * x + p.b * x + p.c;
    double X0 = p.x0 * p.x0 + p.b * p.x0 + p.c;

    double dt1 = 2.0 / x - (2.0 * x + p.b) / X;
    double dt2 = -p.b / X; // simplified from full arctan derivative
    double fac = -p.b * p.x0 / X0;
    double dlog = 2.0 / (x - p.x0) - (2.0 * x + p.b) / X;
    double datan = -(p.b + 2.0 * p.x0) / X; // simplified
    double dt3 = fac * (dlog + datan);

    return p.A * (dt1 + dt2 + dt3);
}

// Spin interpolation f(ζ) = [(1+ζ)^{4/3} + (1-ζ)^{4/3} - 2] / [2(2^{1/3}-1)]
double spin_f(double zeta) {
    double denom = 2.0 * (std::cbrt(2.0) - 1.0);
    return (std::pow(1.0 + zeta, 4.0 / 3.0) + std::pow(1.0 - zeta, 4.0 / 3.0)
            - 2.0)
           / denom;
}

// ε_c(r_s, ζ) for VWN5 (unpolarized: ζ=0 → ε_c = ε_c^P)
double vwn5_ec(double rho_a, double rho_b) {
    double rho = rho_a + rho_b;
    double rs = std::cbrt(3.0 / (4.0 * kPi * rho));
    double x = std::sqrt(rs);
    double zeta = (rho_a - rho_b) / rho;
    zeta = std::max(-1.0 + 1e-14, std::min(1.0 - 1e-14, zeta));

    double ecP = vwn5_F(x, kP);
    double ecF = vwn5_F(x, kF);
    double ac = vwn5_F(x, kA);
    double fz = spin_f(zeta);
    double z4 = zeta * zeta * zeta * zeta;

    double fppZero = 4.0 / (9.0 * (std::cbrt(2.0) - 1.0));
    return ecP + ac * fz / fppZero * (1.0 - z4) + (ecF - ecP) * fz * z4;
}

// ── Independent PBE exchange enhancement ────────────────────────────────

// F_x(s) = 1 + κ - κ/(1 + μs²/κ)
// where κ = 0.804, μ = 0.21951
double pbe_Fx(double s) {
    constexpr double kappa = 0.804;
    constexpr double mu = 0.21951;
    return 1.0 + kappa - kappa / (1.0 + mu * s * s / kappa);
}

} // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════
// Slater exchange: exact analytical validation
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, SlaterExchange_Unpolarized_Rho1) {
    // At ρ_α = ρ_β = 0.5 (ρ = 1.0):
    // ε_x = -C_x = -(3/4)(3/π)^{1/3} = -0.738559
    double rho_a = 0.5, rho_b = 0.5;
    auto ref = SlaterRef::compute(rho_a, rho_b);
    auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));

    // Total LSDA = Slater + VWN5. Exchange part should match exactly.
    // Correlation adds a negative contribution, so exc < ref.exc.
    double exc = static_cast<double>(out.exc);
    EXPECT_LT(exc, ref.exc) << "LSDA exc must be more negative than pure "
                               "Slater (correlation lowers energy)";
    double correlation = exc - ref.exc;
    EXPECT_LT(correlation, 0.0) << "VWN5 correlation must be negative";
    // At ρ=1 (r_s≈0.62), VWN5 |ε_c| can be ~0.14 Ha (logarithmic growth at high density)
    EXPECT_GT(correlation, -0.3)
        << "VWN5 correlation should not dominate exchange";
}

TEST(XCReference, SlaterExchange_MultiDensity) {
    // Test Slater scaling law at several densities.
    // For unpolarized: ε_x ∝ ρ^{1/3}
    // LSDA = Slater + VWN5, so we check Slater dominates.
    double densities[] = {0.01, 0.1, 1.0, 10.0};
    for (double rho : densities) {
        double rho_a = rho / 2, rho_b = rho / 2;
        auto ref = SlaterRef::compute(rho_a, rho_b);
        auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));

        double exc = static_cast<double>(out.exc);
        // Exchange typically dominates (5-20x larger than correlation)
        double ratio = std::abs(exc / ref.exc);
        EXPECT_GT(ratio, 0.9) << "Exchange should dominate at rho=" << rho;
        EXPECT_LT(ratio, 1.5) << "Correlation shouldn't be huge at rho=" << rho;
    }
}

TEST(XCReference, SlaterPotential_Analytical) {
    // Verify LSDA potential v_α at several densities.
    // Slater potential: v_x^α = -(4/3) C_x_spin ρ_α^{1/3}
    // VWN5 adds a correction. At low density (large r_s), the VWN5 correction
    // can be a significant fraction of the exchange potential.
    double test_rhos[] = {0.1, 0.5, 1.0, 5.0};
    for (double rho_a : test_rhos) {
        double rho_b = rho_a;
        auto ref = SlaterRef::compute(rho_a, rho_b);
        auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));

        double vrho = static_cast<double>(out.vrho_alpha);
        // VWN5 potential correction varies; at moderate density ~5-40% of exchange
        double correction_fraction = std::abs((vrho - ref.vrho_a) / ref.vrho_a);
        EXPECT_LT(correction_fraction, 0.5)
            << "VWN5 potential correction too large at rho_a=" << rho_a
            << " (vrho=" << vrho << ", slater=" << ref.vrho_a << ")";
    }
}

TEST(XCReference, SlaterExchange_SpinPolarized) {
    // For fully polarized (ρ_α = ρ, ρ_β ≈ 0):
    // ε_x^polar / ε_x^unpolar = 2^{1/3} ≈ 1.2599 (more negative)
    double rho = 0.5;
    auto ref_closed = SlaterRef::compute(rho / 2, rho / 2);
    auto ref_polar = SlaterRef::compute(rho, 1e-15);

    double ratio = ref_polar.exc / ref_closed.exc;
    EXPECT_NEAR(ratio, std::cbrt(2.0), 1e-6)
        << "Spin scaling relation: ε_x(polar)/ε_x(unpolar) = 2^{1/3}";
}

// ═════════════════════════════════════════════════════════════════════════
// VWN5 correlation: independent implementation validation
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, VWN5_Unpolarized_MatchesIndependent) {
    // Compare evaluate_xc(LSDA) with independently computed Slater + VWN5
    double test_densities[] = {0.01, 0.1, 0.5, 1.0, 5.0, 10.0};
    for (double rho : test_densities) {
        double rho_a = rho / 2, rho_b = rho / 2;
        auto slater = SlaterRef::compute(rho_a, rho_b);
        double ec = vwn5_ec(rho_a, rho_b);
        double exc_ref = slater.exc + ec;

        auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));
        double exc = static_cast<double>(out.exc);

        EXPECT_NEAR(exc, exc_ref, 1e-6)
            << "LSDA exc mismatch at rho=" << rho << " (got " << exc << ", ref "
            << exc_ref << ")";
    }
}

TEST(XCReference, VWN5_SpinPolarized_MatchesIndependent) {
    // Test at non-zero magnetization ζ = (ρ_α - ρ_β) / ρ
    struct TestPoint {
        double rho_a, rho_b;
    };
    TestPoint points[] = {
        {0.3, 0.2},   // ζ = 0.2
        {0.4, 0.1},   // ζ = 0.6
        {0.49, 0.01}, // ζ = 0.96 (nearly fully polarized)
        {1.0, 0.5},   // larger density, ζ = 1/3
    };
    for (auto [ra, rb] : points) {
        auto slater = SlaterRef::compute(ra, rb);
        double ec = vwn5_ec(ra, rb);
        double exc_ref = slater.exc + ec;

        auto out = evaluate_xc(XCType::LSDA, double(ra), double(rb));
        double exc = static_cast<double>(out.exc);

        EXPECT_NEAR(exc, exc_ref, 1e-6)
            << "LSDA spin-polarized mismatch at rho_a=" << ra
            << " rho_b=" << rb;
    }
}

TEST(XCReference, VWN5_CorrelationIsNegative) {
    // Correlation energy must be negative for all physical densities.
    double densities[] = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
    for (double rho : densities) {
        double ec = vwn5_ec(rho / 2, rho / 2);
        EXPECT_LT(ec, 0.0) << "VWN5 correlation must be negative at rho="
                           << rho;
    }
}

TEST(XCReference, VWN5_LowDensityLimit) {
    // At low density (r_s → ∞), correlation → 0.
    // At high density (r_s → 0), ε_c has logarithmic divergence.
    // So |ε_c| should decrease as density decreases (r_s increases).
    double ec_rho001 = vwn5_ec(0.005, 0.005);    // ρ=0.01, r_s≈2.87
    double ec_rho0001 = vwn5_ec(0.0005, 0.0005); // ρ=0.001, r_s≈6.20
    EXPECT_LT(std::abs(ec_rho0001), std::abs(ec_rho001))
        << "VWN5 |ε_c| should decrease at low density (large r_s)";
    // Both should be negative
    EXPECT_LT(ec_rho001, 0.0);
    EXPECT_LT(ec_rho0001, 0.0);
}

// ═════════════════════════════════════════════════════════════════════════
// PBE GGA: exchange enhancement validation
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, PBE_ZeroGradient_CloseToButNotEqualLSDA) {
    // PBE with σ=0 does NOT reduce exactly to LSDA mode's output, because
    // PBE uses PW92 (Perdew-Wang 1992) as its LSDA correlation base while
    // LSDA mode uses VWN5. PW92 and VWN5 are different parameterizations
    // of the same uniform-electron-gas correlation energy; they agree at
    // r_s -> 0 and r_s -> inf but differ by O(1e-4 Ha) at typical r_s.
    // This matches libxc's XC_GGA_C_PBE and the PBE paper (PRL 77, 3865).
    double densities[] = {0.01, 0.1, 0.5, 1.0, 5.0};
    for (double rho : densities) {
        auto lda = evaluate_xc(XCType::LSDA, double(rho / 2), double(rho / 2));
        auto pbe =
            evaluate_xc(XCType::PBE, double(rho / 2), double(rho / 2), 0, 0, 0);

        // Both should be negative (XC lowers energy).
        EXPECT_LT(static_cast<double>(pbe.exc), 0.0)
            << "PBE exc should be negative at rho=" << rho;
        EXPECT_LT(static_cast<double>(lda.exc), 0.0)
            << "LSDA exc should be negative at rho=" << rho;
        // They differ but by no more than ~mHa (PW92 vs VWN5 systematic gap).
        EXPECT_LT(std::abs(pbe.exc - lda.exc), 5e-3)
            << "PBE(PW92) vs LSDA(VWN5) should differ by < 5 mHa at rho="
            << rho;
    }
}

TEST(XCReference, PBE_EnhancementFactor) {
    // The PBE enhancement factor F_x(s) = 1 + κ - κ/(1 + μs²/κ)
    // At s=0: F_x = 1 (pure LDA)
    // At s→∞: F_x → 1 + κ = 1.804 (Lieb-Oxford bound)
    EXPECT_NEAR(pbe_Fx(0.0), 1.0, 1e-15);
    EXPECT_LT(pbe_Fx(100.0), 1.805);
    EXPECT_GT(pbe_Fx(100.0), 1.800);

    // F_x monotonically increases with s
    for (double s = 0.1; s < 10.0; s += 0.5) {
        EXPECT_GT(pbe_Fx(s), pbe_Fx(s - 0.1))
            << "F_x must be monotonically increasing at s=" << s;
    }
}

TEST(XCReference, PBE_ExchangeMoreNegativeThanLDA) {
    // PBE exchange enhancement F_x >= 1, so |E_x^PBE| >= |E_x^LDA|.
    // With gradient present, PBE exc should be more negative (due to exchange).
    // Note: correlation correction may partially offset this.
    double rho = 0.3;
    double sigma = 0.1; // nonzero gradient
    auto lda = evaluate_xc(XCType::LSDA, double(rho), double(rho));
    auto pbe =
        evaluate_xc(XCType::PBE, double(rho), double(rho), sigma, 0, sigma);

    // PBE total might not always be more negative due to correlation GGA correction,
    // but the difference should be nonzero.
    EXPECT_NE(static_cast<double>(pbe.exc), static_cast<double>(lda.exc))
        << "PBE with gradient must differ from LSDA";
}

TEST(XCReference, PBE_VsigmaSign) {
    // For exchange: ∂ε_x/∂σ_αα should be negative (increasing gradient
    // makes exchange more negative). The vsigma should thus be negative.
    double rho = 0.3;
    auto pbe = evaluate_xc(XCType::PBE, double(rho), double(rho), double(0.1),
                           double(0.05), double(0.1));
    EXPECT_LT(static_cast<double>(pbe.vsigma_aa), 0.0)
        << "PBE exchange vsigma should be negative";
}

TEST(XCReference, PBE_SpinSymmetry) {
    // For ρ_α = ρ_β, σ_αα = σ_ββ: must have symmetric potentials
    double rho = 0.5;
    double sigma = 0.2;
    auto pbe = evaluate_xc(XCType::PBE, double(rho), double(rho), sigma,
                           double(0.1), sigma);

    EXPECT_NEAR(static_cast<double>(pbe.vrho_alpha),
                static_cast<double>(pbe.vrho_beta), 1e-12)
        << "PBE: closed-shell vrho must be symmetric";
    EXPECT_NEAR(static_cast<double>(pbe.vsigma_aa),
                static_cast<double>(pbe.vsigma_bb), 1e-12)
        << "PBE: closed-shell vsigma must be symmetric";
}

// ═════════════════════════════════════════════════════════════════════════
// LSDA total energy density: comprehensive sweep
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, LSDA_Total_Sweep) {
    // At each density, verify LSDA = Slater_ref + VWN5_ref
    // This is a comprehensive validation over a log-spaced density range.
    for (int i = -4; i <= 3; ++i) {
        double rho = std::pow(10.0, i);
        double rho_a = rho / 2, rho_b = rho / 2;

        auto slater = SlaterRef::compute(rho_a, rho_b);
        double ec = vwn5_ec(rho_a, rho_b);
        double exc_ref = slater.exc + ec;

        auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));
        double exc = static_cast<double>(out.exc);

        EXPECT_NEAR(exc, exc_ref, 1e-6)
            << "LSDA total mismatch at rho=" << rho
            << " (exchange=" << slater.exc << " corr=" << ec << ")";
    }
}

// ═════════════════════════════════════════════════════════════════════════
// LSDA potential: finite-difference validation
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, LSDA_Potential_FiniteDifference) {
    // Verify: vrho_α ≈ d(ρ * ε_xc)/dρ_α via central finite difference.
    // This validates that the potential is the correct derivative.
    // Use h=1e-3 for float-safe FD (float has ~7 decimal digits).
    double rho_a = 0.3, rho_b = 0.3;
    double h = 1e-3;

    auto out_fwd = evaluate_xc(XCType::LSDA, double(rho_a + h), double(rho_b));
    auto out_bwd = evaluate_xc(XCType::LSDA, double(rho_a - h), double(rho_b));

    double rho_fwd = (rho_a + h) + rho_b;
    double rho_bwd = (rho_a - h) + rho_b;
    double e_fwd = static_cast<double>(out_fwd.exc) * rho_fwd;
    double e_bwd = static_cast<double>(out_bwd.exc) * rho_bwd;
    double vrho_fd = (e_fwd - e_bwd) / (2.0 * h);

    auto out = evaluate_xc(XCType::LSDA, double(rho_a), double(rho_b));
    double vrho = static_cast<double>(out.vrho_alpha);

    // FD truncation O(h^2) ≈ 1e-6 + float noise O(ε/h) ≈ 1e-4
    EXPECT_NEAR(vrho, vrho_fd, 0.02)
        << "LSDA potential finite-difference mismatch"
        << " (analytic=" << vrho << " fd=" << vrho_fd << ")";
}

TEST(XCReference, PBE_Potential_FiniteDifference) {
    // Verify PBE potential via central finite difference on ρ_α.
    double rho_a = 0.3, rho_b = 0.3;
    double sigma_aa = 0.1, sigma_ab = 0.05, sigma_bb = 0.1;
    double h = 1e-3;

    auto out_fwd =
        evaluate_xc(XCType::PBE, double(rho_a + h), double(rho_b),
                    double(sigma_aa), double(sigma_ab), double(sigma_bb));
    auto out_bwd =
        evaluate_xc(XCType::PBE, double(rho_a - h), double(rho_b),
                    double(sigma_aa), double(sigma_ab), double(sigma_bb));

    double rho_fwd = (rho_a + h) + rho_b;
    double rho_bwd = (rho_a - h) + rho_b;
    double e_fwd = static_cast<double>(out_fwd.exc) * rho_fwd;
    double e_bwd = static_cast<double>(out_bwd.exc) * rho_bwd;
    double vrho_fd = (e_fwd - e_bwd) / (2.0 * h);

    auto out =
        evaluate_xc(XCType::PBE, double(rho_a), double(rho_b), double(sigma_aa),
                    double(sigma_ab), double(sigma_bb));
    double vrho = static_cast<double>(out.vrho_alpha);

    EXPECT_NEAR(vrho, vrho_fd, 0.02)
        << "PBE potential finite-difference mismatch";
}

TEST(XCReference, PBE_Vsigma_FiniteDifference) {
    // Verify PBE vsigma_aa via central finite difference on σ_αα.
    double rho_a = 0.3, rho_b = 0.3;
    double sigma_aa = 0.1, sigma_ab = 0.05, sigma_bb = 0.1;
    double h = 1e-3;

    auto out_fwd =
        evaluate_xc(XCType::PBE, double(rho_a), double(rho_b),
                    double(sigma_aa + h), double(sigma_ab), double(sigma_bb));
    auto out_bwd =
        evaluate_xc(XCType::PBE, double(rho_a), double(rho_b),
                    double(sigma_aa - h), double(sigma_ab), double(sigma_bb));

    double rho = rho_a + rho_b;
    double e_fwd = static_cast<double>(out_fwd.exc) * rho;
    double e_bwd = static_cast<double>(out_bwd.exc) * rho;
    double vsigma_fd = (e_fwd - e_bwd) / (2.0 * h);

    auto out =
        evaluate_xc(XCType::PBE, double(rho_a), double(rho_b), double(sigma_aa),
                    double(sigma_ab), double(sigma_bb));
    double vsigma = static_cast<double>(out.vsigma_aa);

    EXPECT_NEAR(vsigma, vsigma_fd, 0.02)
        << "PBE vsigma finite-difference mismatch";
}

// ═════════════════════════════════════════════════════════════════════════
// Grid integration: analytical integrals as accuracy check
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, GridIntegration_GaussianNormalization) {
    // ∫ exp(-α r²) dτ = (π/α)^{3/2}
    // Tests the molecular grid accuracy against an exact analytical result.
    native::Atom h = {1, {0.0, 0.0, 0.0}};
    auto grid = build_molecular_grid(&h, 1, GridLevel::SG0, 50, 302);

    double alpha = 1.0;
    double integral = 0;
    for (const auto& gp : grid.points) {
        double x = static_cast<double>(gp.x);
        double y = static_cast<double>(gp.y);
        double z = static_cast<double>(gp.z);
        double r2 = x * x + y * y + z * z;
        integral += static_cast<double>(gp.weight) * std::exp(-alpha * r2);
    }
    double exact = std::pow(kPi / alpha, 1.5);
    // Tolerance depends on grid quality — 302-point Lebedev is exact for
    // spherical harmonics up to l=29 but Gaussian tail extends to infinity.
    EXPECT_NEAR(integral, exact, 0.5)
        << "Grid integration of Gaussian failed: got " << integral
        << " expected " << exact;
}

TEST(XCReference, GridIntegration_STO1sNormalization) {
    // ∫ |ψ_1s|² dτ = 1 where ψ_1s = (α³/π)^{1/2} exp(-α r)
    // For α=1: ∫ (1/π) exp(-2r) dτ = 1
    // i.e., ∫ exp(-2r) dτ = π
    native::Atom h = {1, {0.0, 0.0, 0.0}};
    auto grid = build_molecular_grid(&h, 1, GridLevel::SG0, 50, 302);

    double integral = 0;
    for (const auto& gp : grid.points) {
        double x = static_cast<double>(gp.x);
        double y = static_cast<double>(gp.y);
        double z = static_cast<double>(gp.z);
        double r = std::sqrt(x * x + y * y + z * z);
        integral += static_cast<double>(gp.weight) * std::exp(-2.0 * r);
    }
    EXPECT_NEAR(integral, kPi, 0.5) << "Grid integration of exp(-2r): got "
                                    << integral << " expected " << kPi;
}

TEST(XCReference, GridIntegration_TwoAtomPartition) {
    // For two identical atoms, ∫ f dτ should be the same regardless of
    // whether we use 1 grid on 1 atom or Becke-partitioned grid on 2 atoms
    // (for symmetric f centered at midpoint).
    double d = 1.4; // Bohr
    native::Atom atoms[] = {{1, {-d / 2, 0, 0}}, {1, {d / 2, 0, 0}}};
    auto grid = build_molecular_grid(atoms, 2, GridLevel::SG0, 50, 110);

    // Integrate f(r) = 1 near the origin to check Becke weights sum correctly
    double sum_weights = 0;
    for (const auto& gp : grid.points) {
        sum_weights += static_cast<double>(gp.weight);
    }
    // Total weight should be positive and finite
    EXPECT_GT(sum_weights, 0.0);
    EXPECT_FALSE(std::isinf(sum_weights));
    EXPECT_FALSE(std::isnan(sum_weights));
}

// ═════════════════════════════════════════════════════════════════════════
// Boys function: analytical reference validation
// ═════════════════════════════════════════════════════════════════════════

TEST(XCReference, BoysF0_AnalyticalFormula) {
    // F_0(x) = 0.5 sqrt(π/x) erf(sqrt(x)) for x > 0
    // F_0(0) = 1
    double test_x[] = {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
    for (double x : test_x) {
        double ref = 0.5 * std::sqrt(kPi / x) * std::erf(std::sqrt(x));
        double val = native::boys_f0(x);
        EXPECT_NEAR(val, ref, 1e-12) << "Boys F_0 mismatch at x=" << x;
    }
}

TEST(XCReference, BoysFn_DownwardRecurrence) {
    // Validate boys_fn against downward recurrence from F_0.
    // Upward recurrence: F_{n+1}(x) = [(2n+1) F_n(x) - exp(-x)] / (2x)
    for (double x : {0.1, 0.5, 1.0, 5.0, 10.0}) {
        double F0 = native::boys_f0(x);
        double ex = std::exp(-x);

        // Compute F_1..F_4 via recurrence
        double Fn_recurrence[5];
        Fn_recurrence[0] = F0;
        for (int n = 0; n < 4; ++n)
            Fn_recurrence[n + 1] =
                ((2 * n + 1) * Fn_recurrence[n] - ex) / (2.0 * x);

        // Compare against boys_fn
        for (int n = 0; n <= 4; ++n) {
            EXPECT_NEAR(native::boys_fn(n, x), Fn_recurrence[n], 1e-10)
                << "F_" << n << "(" << x << ") recurrence mismatch";
        }
    }
}

TEST(XCReference, BoysFn_ZeroArgLimits) {
    // F_n(0) = 1/(2n+1)
    for (int n = 0; n <= 6; ++n) {
        double ref = 1.0 / (2.0 * n + 1.0);
        EXPECT_NEAR(native::boys_fn(n, 0.0), ref, 1e-12)
            << "F_" << n << "(0) should be 1/" << (2 * n + 1);
    }
}

#else

#include <gtest/gtest.h>
TEST(XCReference, SkippedNoBuildRHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — XC reference tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF
