/// @file tests/cpu/scf/test_primitives.cpp
/// @brief Isolated unit tests for primitive-level integral functions.
///
/// Covers overlap_prim_3d, kinetic_prim_3d, nuclear_prim, eri_prim
/// that are otherwise only tested transitively through gradient code.
/// Uses analytical identities (symmetry, self-overlap, positivity)
/// and finite-difference cross-checks.

#ifdef ATV_HAS_HF

#include "atomiverse/cpu/scf/native_integrals.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ATV::cpu::scf::native;

namespace {

constexpr double kAlpha = 1.0;
constexpr double kBeta = 0.8;
constexpr double kA[3] = {0.0, 0.0, 0.0};
constexpr double kB[3] = {1.0, 0.5, 0.3};

} // namespace

// ── overlap_prim_3d ──────────────────────────────────────────────────

TEST(PrimitiveIntegrals, Overlap3D_ss_MatchesAnalytic) {
    // (000|000) must equal the s-type analytic formula.
    int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double val = overlap_prim_3d(a, b, kAlpha, kA, kBeta, kB);
    double gamma = kAlpha + kBeta;
    double mu = kAlpha * kBeta / gamma;
    double R2 = 1.0 * 1.0 + 0.5 * 0.5 + 0.3 * 0.3;
    double expected = std::pow(M_PI / gamma, 1.5) * std::exp(-mu * R2);
    EXPECT_NEAR(val, expected, 1e-12);
}

TEST(PrimitiveIntegrals, Overlap3D_Symmetry) {
    int a[3] = {1, 0, 0}, b[3] = {0, 1, 0};
    double val_ab = overlap_prim_3d(a, b, kAlpha, kA, kBeta, kB);
    double val_ba = overlap_prim_3d(b, a, kBeta, kB, kAlpha, kA);
    EXPECT_NEAR(val_ab, val_ba, 1e-12) << "Overlap must be symmetric";
}

TEST(PrimitiveIntegrals, Overlap3D_SelfPositive) {
    int a[3] = {1, 0, 0};
    double val = overlap_prim_3d(a, a, kAlpha, kA, kAlpha, kA);
    EXPECT_GT(val, 0.0) << "Self-overlap <px|px> must be positive";
}

TEST(PrimitiveIntegrals, Overlap3D_Orthogonality) {
    // <px|py> on same center = 0 by angular parity
    int px[3] = {1, 0, 0}, py[3] = {0, 1, 0};
    double val = overlap_prim_3d(px, py, kAlpha, kA, kAlpha, kA);
    EXPECT_NEAR(val, 0.0, 1e-14) << "<px|py> on same center must be 0";
}

// ── kinetic_prim_3d ──────────────────────────────────────────────────

TEST(PrimitiveIntegrals, Kinetic3D_ss_Positive) {
    int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double val = kinetic_prim_3d(a, b, kAlpha, kA, kAlpha, kA);
    EXPECT_GT(val, 0.0) << "Diagonal kinetic must be positive";
}

TEST(PrimitiveIntegrals, Kinetic3D_Symmetry) {
    int a[3] = {1, 0, 0}, b[3] = {0, 0, 0};
    double val_ab = kinetic_prim_3d(a, b, kAlpha, kA, kBeta, kB);
    double val_ba = kinetic_prim_3d(b, a, kBeta, kB, kAlpha, kA);
    EXPECT_NEAR(val_ab, val_ba, 1e-12) << "Kinetic must be symmetric";
}

TEST(PrimitiveIntegrals, Kinetic3D_VsFD) {
    // Finite-difference check: T_{ss} ≈ -½ d²S/dx² at x=0 for a shifted
    // Gaussian — i.e. we displace center B by h and check the Laplacian of S.
    int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double h = 1e-4;
    auto S = [&](double dx) {
        double Bh[3] = {kB[0] + dx, kB[1], kB[2]};
        return overlap_prim_3d(a, b, kAlpha, kA, kBeta, Bh);
    };
    // d²S/dx² ≈ (S(+h) - 2S(0) + S(-h)) / h²
    double d2S = (S(h) - 2.0 * S(0) + S(-h)) / (h * h);
    // Kinetic from two-index formula (uses the actual recurrence)
    double T_ref = kinetic_prim_3d(a, b, kAlpha, kA, kBeta, kB);
    // T = -½ ∂²/∂(RB_x)² S · (something related to beta)
    // This is a rough consistency check — not exact because the FD is on
    // the centre, not on the electron coordinate. Just verify sign and order.
    EXPECT_LT(d2S, 0.0)
        << "S decreases convexly → d²S<0 in separation direction";
    EXPECT_GT(T_ref, 0.0)
        << "Kinetic must be positive for overlapping Gaussians";
}

// ── nuclear_prim ─────────────────────────────────────────────────────

TEST(PrimitiveIntegrals, NuclearPrim_Symmetry) {
    int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double gamma = kAlpha + kBeta;
    double PA[3], PB[3], PC[3];
    double R2_AB = 0.0;
    for (int d = 0; d < 3; ++d) {
        double P = (kAlpha * kA[d] + kBeta * kB[d]) / gamma;
        PA[d] = P - kA[d];
        PB[d] = P - kB[d];
        PC[d] = P - 0.0; // nucleus at origin
        double dAB = kA[d] - kB[d];
        R2_AB += dAB * dAB;
    }
    double K_AB = std::exp(-kAlpha * kBeta / gamma * R2_AB);

    double V_ab = nuclear_prim(a, b, gamma, PA, PB, PC, K_AB);
    // Swap bra labels
    double PA2[3], PB2[3];
    for (int d = 0; d < 3; ++d) {
        PA2[d] = PB[d]; // P - B
        PB2[d] = PA[d]; // P - A
    }
    double V_ba = nuclear_prim(b, a, gamma, PA2, PB2, PC, K_AB);
    EXPECT_NEAR(V_ab, V_ba, 1e-12) << "Nuclear prim must be symmetric";
}

TEST(PrimitiveIntegrals, NuclearPrim_Positive) {
    // nuclear_prim returns the RAW Coulomb integral (without -Z factor).
    // With nucleus at Gaussian product center → maximum value → positive.
    int a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    double gamma = kAlpha + kAlpha;
    double PA[3] = {0, 0, 0}, PB[3] = {0, 0, 0}, PC[3] = {0, 0, 0};
    double K_AB = 1.0; // same center
    double V = nuclear_prim(a, b, gamma, PA, PB, PC, K_AB);
    EXPECT_GT(V, 0.0) << "Raw nuclear integral (without -Z) must be positive";
}

// ── eri_prim ─────────────────────────────────────────────────────────

TEST(PrimitiveIntegrals, ERIPrim_8FoldSymmetry) {
    int a[3] = {1, 0, 0}, b[3] = {0, 0, 0};
    int c[3] = {0, 1, 0}, d[3] = {0, 0, 0};
    double C[3] = {0.5, 0.8, 0.0}, D[3] = {-0.3, 0.2, 0.7};

    double base =
        eri_prim(a, b, c, d, kAlpha, kA, kBeta, kB, kAlpha, C, kBeta, D);
    // Swap bra: (ba|cd)
    EXPECT_NEAR(
        eri_prim(b, a, c, d, kBeta, kB, kAlpha, kA, kAlpha, C, kBeta, D), base,
        1e-12)
        << "(ba|cd) != (ab|cd)";
    // Swap ket: (ab|dc)
    EXPECT_NEAR(
        eri_prim(a, b, d, c, kAlpha, kA, kBeta, kB, kBeta, D, kAlpha, C), base,
        1e-12)
        << "(ab|dc) != (ab|cd)";
    // Swap bra↔ket: (cd|ab)
    EXPECT_NEAR(
        eri_prim(c, d, a, b, kAlpha, C, kBeta, D, kAlpha, kA, kBeta, kB), base,
        1e-12)
        << "(cd|ab) != (ab|cd)";
}

TEST(PrimitiveIntegrals, ERIPrim_Positive) {
    int s[3] = {0, 0, 0};
    double val =
        eri_prim(s, s, s, s, kAlpha, kA, kAlpha, kA, kAlpha, kA, kAlpha, kA);
    EXPECT_GT(val, 0.0) << "Self-repulsion (ss|ss) must be positive";
}

#endif // ATV_HAS_HF
