/// @file test_xc_functional.cpp
/// @brief Unit tests for XC functional evaluator (Slater exchange, VWN5 correlation, PBE).

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {
constexpr double kTol = 1e-6;
}

// ── Slater Exchange ─────────────────────────────────────────────────────

TEST(XCFunctional, SlaterExchangeUniformDensity) {
    // For uniform density ρ_α = ρ_β = 0.5, the exchange energy density should be
    // ε_x = -C_x * ρ^{1/3} for closed-shell (standard Slater formula).
    // C_x = (3/4)(3/π)^{1/3} ≈ 0.7386
    // For spin-polarized: ε_x = -2^{1/3} * C_x * [ρ_α^{4/3} + ρ_β^{4/3}] / ρ
    auto out = evaluate_xc(XCType::LSDA, 0.5, 0.5);

    // Exchange part: exc should be negative (exchange lowers energy)
    EXPECT_LT(static_cast<double>(out.exc), 0.0);
    // Potential should also be negative
    EXPECT_LT(static_cast<double>(out.vrho_alpha), 0.0);
    EXPECT_LT(static_cast<double>(out.vrho_beta), 0.0);
}

TEST(XCFunctional, SlaterExchangeSpinScaling) {
    // Spin scaling relation: for closed-shell (ρ_α = ρ_β = ρ/2),
    // the exchange potential must be the same for alpha and beta.
    for (double rho : {0.01, 0.1, 1.0, 10.0}) {
        auto out = evaluate_xc(XCType::LSDA, rho / 2, rho / 2);
        EXPECT_NEAR(static_cast<double>(out.vrho_alpha),
                    static_cast<double>(out.vrho_beta), 1e-10)
            << "alpha/beta potentials differ at rho=" << rho;
    }
}

TEST(XCFunctional, SlaterExchangeFullyPolarized) {
    // Fully polarized: ρ_α = ρ, ρ_β = 0.
    // E_x[ρ, 0] should be 2^{1/3} times E_x[ρ/2, ρ/2] per electron.
    double rho = 0.5;
    auto closed = evaluate_xc(XCType::LSDA, rho / 2, rho / 2);
    auto polar = evaluate_xc(XCType::LSDA, rho, double(1e-13));

    // The fully polarized exchange should be more negative (lower energy)
    // than unpolarized, because 2^{1/3} > 1.
    EXPECT_LT(static_cast<double>(polar.exc), static_cast<double>(closed.exc));
}

// ── VWN5 Correlation ────────────────────────────────────────────────────

TEST(XCFunctional, VWN5UnpolarizedNegative) {
    // Correlation energy density should be negative for physical densities.
    // (electrons lower their energy by correlating)
    for (double rho : {0.01, 0.1, 1.0}) {
        auto out = evaluate_xc(XCType::LSDA, rho / 2, rho / 2);
        // Total exc includes exchange (negative) + correlation (negative)
        EXPECT_LT(static_cast<double>(out.exc), 0.0);
    }
}

TEST(XCFunctional, VWN5ClosedShellSymmetry) {
    // For ρ_α = ρ_β, the correlation potential must be equal for both spins.
    double rho = 0.3;
    auto out = evaluate_xc(XCType::LSDA, rho, rho);
    EXPECT_NEAR(static_cast<double>(out.vrho_alpha),
                static_cast<double>(out.vrho_beta), 1e-10);
}

TEST(XCFunctional, VWN5HighDensityCorrelation) {
    // At high density (small r_s), correlation should be small.
    // r_s = (3/(4πρ))^{1/3}
    // At ρ = 100, r_s ≈ 0.13 — high density limit.
    double rho = 100.0;
    auto out = evaluate_xc(XCType::LSDA, rho / 2, rho / 2);
    // Exchange dominates; correlation is a small correction.
    // Just check it doesn't blow up.
    EXPECT_FALSE(std::isnan(static_cast<double>(out.exc)));
    EXPECT_FALSE(std::isinf(static_cast<double>(out.exc)));
}

// ── Density Cutoff ──────────────────────────────────────────────────────

TEST(XCFunctional, DensityCutoffReturnsZero) {
    // Below ρ_min = 1e-12, all outputs should be zero.
    auto out = evaluate_xc(XCType::LSDA, double(1e-14), double(1e-14));
    EXPECT_EQ(static_cast<double>(out.exc), 0.0);
    EXPECT_EQ(static_cast<double>(out.vrho_alpha), 0.0);
    EXPECT_EQ(static_cast<double>(out.vrho_beta), 0.0);
}

TEST(XCFunctional, DensityCutoffNoNaN) {
    // Even at very small densities just above cutoff, no NaN.
    auto out = evaluate_xc(XCType::LSDA, double(1e-11), double(1e-11));
    EXPECT_FALSE(std::isnan(static_cast<double>(out.exc)));
    EXPECT_FALSE(std::isnan(static_cast<double>(out.vrho_alpha)));
    EXPECT_FALSE(std::isnan(static_cast<double>(out.vrho_beta)));
}

// ── PBE GGA ─────────────────────────────────────────────────────────────

TEST(XCFunctional, PBEAtZeroGradientHasNoEnhancement) {
    // At sigma=0, PBE exchange has F_x(0) = 1 so it equals LSDA exchange,
    // and PBE correlation H(t=0) = 0. The total `exc` therefore does NOT
    // equal LSDA's `exc`, because PBE uses PW92 (Perdew-Wang 1992) as the
    // LSDA correlation base while LSDA mode uses VWN5. PW92 and VWN5 differ
    // by a few mHa at typical densities -- this is intentional and matches
    // the PBE paper (PRL 77, 3865 (1996)) and libxc's XC_GGA_C_PBE.
    //
    // What we CAN assert: PBE@sigma=0 and VWN5-LSDA are within ~1 mHa.
    double rho = 0.3;
    auto lda = evaluate_xc(XCType::LSDA, rho, rho);
    auto pbe = evaluate_xc(XCType::PBE, rho, rho, 0, 0, 0);
    EXPECT_LT(std::abs(pbe.exc - lda.exc), 5e-3)
        << "PBE@sigma=0 (PW92 base) and VWN5-LSDA should be within ~mHa";
}

TEST(XCFunctional, PBEEnhancementFactor) {
    // With nonzero gradient, PBE exchange should be enhanced (more negative).
    double rho = 0.3;
    auto lda = evaluate_xc(XCType::LSDA, rho, rho, 0, 0, 0);
    auto pbe = evaluate_xc(XCType::PBE, rho, rho, 0.1, 0, 0.1);

    // PBE exchange enhancement factor F_x >= 1, so |E_x^PBE| >= |E_x^LDA|
    // The total exc may differ due to correlation too.
    EXPECT_NE(static_cast<double>(pbe.exc), static_cast<double>(lda.exc));
}

TEST(XCFunctional, PBEVsigmaZeroForLDA) {
    // LSDA should have zero vsigma outputs.
    auto lda = evaluate_xc(XCType::LSDA, 0.3, 0.3);
    EXPECT_EQ(static_cast<double>(lda.vsigma_aa), 0.0);
    EXPECT_EQ(static_cast<double>(lda.vsigma_ab), 0.0);
    EXPECT_EQ(static_cast<double>(lda.vsigma_bb), 0.0);
}

TEST(XCFunctional, PBEVsigmaNonzero) {
    // PBE with nonzero gradient should have nonzero vsigma.
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.1, 0.05, 0.1);
    // vsigma should be nonzero when gradient is nonzero
    EXPECT_NE(static_cast<double>(pbe.vsigma_aa), 0.0);
}

TEST(XCFunctional, PBEClosedShellSymmetry) {
    // For ρ_α = ρ_β and σ_αα = σ_ββ, potentials must be symmetric.
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.1, 0.05, 0.1);
    EXPECT_NEAR(static_cast<double>(pbe.vrho_alpha),
                static_cast<double>(pbe.vrho_beta), 1e-8);
    EXPECT_NEAR(static_cast<double>(pbe.vsigma_aa),
                static_cast<double>(pbe.vsigma_bb), 1e-8);
}

// ── PBEsol GGA ──────────────────────────────────────────────────────────

TEST(XCFunctional, PBEsolMetadata) {
    EXPECT_TRUE(is_gga(XCType::PBEsol));
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::PBEsol), 0.0);
    EXPECT_DOUBLE_EQ(hybrid_pt2_fraction(XCType::PBEsol), 0.0);
}

TEST(XCFunctional, PBEsolMatchesPBEAtZeroGradient) {
    // At σ=0, Fx=1 and H=0 → PBEsol and PBE share the PW92 LDA base.
    for (double rho : {0.01, 0.1, 0.5, 2.0}) {
        auto pbe = evaluate_xc(XCType::PBE, rho / 2, rho / 2, 0, 0, 0);
        auto sol = evaluate_xc(XCType::PBEsol, rho / 2, rho / 2, 0, 0, 0);
        EXPECT_NEAR(sol.exc, pbe.exc, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(sol.vrho_alpha, pbe.vrho_alpha, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(sol.vrho_beta, pbe.vrho_beta, 1e-12) << "rho=" << rho;
    }
}

TEST(XCFunctional, PBEsolDiffersFromPBEAtFiniteGradient) {
    // μ_sol = 10/81 < μ_PBE → different enhancement at finite s.
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto sol = evaluate_xc(XCType::PBEsol, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(sol.exc - pbe.exc), 1e-8);
    EXPECT_TRUE(std::isfinite(sol.exc));
    EXPECT_TRUE(std::isfinite(sol.vsigma_aa));
}

// ── revPBE GGA (κ=1.245; Zhang & Yang 1998) ──────────────────────────────

TEST(XCFunctional, RevPBEMetadata) {
    EXPECT_TRUE(is_gga(XCType::RevPBE));
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::RevPBE), 0.0);
    EXPECT_DOUBLE_EQ(hybrid_pt2_fraction(XCType::RevPBE), 0.0);
}

TEST(XCFunctional, RevPBEMatchesPBEAtZeroGradient) {
    // At σ=0, Fx=1 and H=0 → revPBE and PBE share the PW92 LDA base.
    for (double rho : {0.01, 0.1, 0.5, 2.0}) {
        auto pbe = evaluate_xc(XCType::PBE, rho / 2, rho / 2, 0, 0, 0);
        auto rev = evaluate_xc(XCType::RevPBE, rho / 2, rho / 2, 0, 0, 0);
        EXPECT_NEAR(rev.exc, pbe.exc, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(rev.vrho_alpha, pbe.vrho_alpha, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(rev.vrho_beta, pbe.vrho_beta, 1e-12) << "rho=" << rho;
    }
}

TEST(XCFunctional, RevPBEDiffersFromPBEAtFiniteGradient) {
    // κ_rev = 1.245 > κ_PBE → different enhancement at finite s.
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto rev = evaluate_xc(XCType::RevPBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(rev.exc - pbe.exc), 1e-8);
    EXPECT_TRUE(std::isfinite(rev.exc));
    EXPECT_TRUE(std::isfinite(rev.vsigma_aa));
}

// ── RPBE GGA (exponential F_x; Hammer–Hansen–Nørskov 1999) ──────────────

TEST(XCFunctional, RPBEMetadata) {
    EXPECT_TRUE(is_gga(XCType::RPBE));
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::RPBE), 0.0);
    EXPECT_DOUBLE_EQ(hybrid_pt2_fraction(XCType::RPBE), 0.0);
}

TEST(XCFunctional, RPBEMatchesPBEAtZeroGradient) {
    // At σ=0, F_x=1 and H=0 → RPBE and PBE share the PW92 LDA base.
    for (double rho : {0.01, 0.1, 0.5, 2.0}) {
        auto pbe = evaluate_xc(XCType::PBE, rho / 2, rho / 2, 0, 0, 0);
        auto rpbe = evaluate_xc(XCType::RPBE, rho / 2, rho / 2, 0, 0, 0);
        EXPECT_NEAR(rpbe.exc, pbe.exc, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(rpbe.vrho_alpha, pbe.vrho_alpha, 1e-12) << "rho=" << rho;
        EXPECT_NEAR(rpbe.vrho_beta, pbe.vrho_beta, 1e-12) << "rho=" << rho;
    }
}

TEST(XCFunctional, RPBEDiffersFromPBEAtFiniteGradient) {
    // Exponential F_x vs rational PBE form → different enhancement at finite s.
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto rpbe = evaluate_xc(XCType::RPBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(rpbe.exc - pbe.exc), 1e-8);
    EXPECT_TRUE(std::isfinite(rpbe.exc));
    EXPECT_TRUE(std::isfinite(rpbe.vsigma_aa));
}

TEST(XCFunctional, RPBEDiffersFromRevPBEAtFiniteGradient) {
    // RPBE ≠ revPBE (different exchange form / κ).
    auto rev = evaluate_xc(XCType::RevPBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto rpbe = evaluate_xc(XCType::RPBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(rpbe.exc - rev.exc), 1e-8);
}

TEST(XCFunctional, OptB88VdWMetadata) {
    EXPECT_TRUE(is_gga(XCType::OptB88VdW));
    EXPECT_DOUBLE_EQ(hybrid_exchange_fraction(XCType::OptB88VdW), 0.0);
    EXPECT_DOUBLE_EQ(hybrid_pt2_fraction(XCType::OptB88VdW), 0.0);
}

TEST(XCFunctional, OptB88VdWDiffersFromB88AtFiniteGradient) {
    // optB88 beta/gamma != B88; pure OptB88VdW uses PW92 not LYP.
    auto blyp = evaluate_xc(XCType::BLYP, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto opt = evaluate_xc(XCType::OptB88VdW, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(opt.exc - blyp.exc), 1e-6);
    EXPECT_TRUE(std::isfinite(opt.exc));
    EXPECT_TRUE(std::isfinite(opt.vsigma_aa));
    EXPECT_NEAR(opt.vsigma_ab, 0.0, 1e-15); // LDA-C only
}

TEST(XCFunctional, OptB88VdWDiffersFromPBEAtFiniteGradient) {
    auto pbe = evaluate_xc(XCType::PBE, 0.3, 0.3, 0.2, 0.1, 0.2);
    auto opt = evaluate_xc(XCType::OptB88VdW, 0.3, 0.3, 0.2, 0.1, 0.2);
    EXPECT_GT(std::abs(opt.exc - pbe.exc), 1e-6);
}

TEST(XCFunctional, OptB88VdWClosedShellSymmetry) {
    auto out = evaluate_xc(XCType::OptB88VdW, 0.25, 0.25, 0.1, 0.05, 0.1);
    EXPECT_NEAR(out.vrho_alpha, out.vrho_beta, 1e-12);
    EXPECT_NEAR(out.vsigma_aa, out.vsigma_bb, 1e-12);
}

// ── Numerical stability ─────────────────────────────────────────────────

TEST(XCFunctional, NoNaNAtVariousDensities) {
    // Sweep over a range of densities to check for NaN/Inf.
    for (double rho : {1e-11, 1e-6, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0}) {
        auto lda = evaluate_xc(XCType::LSDA, rho / 2, rho / 2);
        EXPECT_FALSE(std::isnan(static_cast<double>(lda.exc)))
            << "NaN at rho=" << rho;
        EXPECT_FALSE(std::isinf(static_cast<double>(lda.exc)))
            << "Inf at rho=" << rho;
    }
}

TEST(XCFunctional, FullyPolarizedNoNaN) {
    // ρ_β = 0 (within cutoff) should not cause NaN.
    auto out = evaluate_xc(XCType::LSDA, 0.5, double(0));
    EXPECT_FALSE(std::isnan(static_cast<double>(out.exc)));
    EXPECT_FALSE(std::isnan(static_cast<double>(out.vrho_alpha)));
    // vrho_beta for zero beta density: should still be finite
    EXPECT_FALSE(std::isnan(static_cast<double>(out.vrho_beta)));
}

#else

#include <gtest/gtest.h>
TEST(XCFunctional, SkippedNoBuildRHF) {
    GTEST_SKIP()
        << "ATV_BUILD_HF=OFF — XC functional tests require -DATV_BUILD_HF=ON";
}

#endif // ATV_HAS_HF
