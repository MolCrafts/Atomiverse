/// @file tests/cpu/md/test_pe_ff_constants.cpp
/// @brief Canonical Dreiding-UA polyethylene FF constants pin test.
///
/// the published MSU LAMMPS deck (`lammps_msu_paper.in`) is the
/// ground truth. This file states the same values in C++ form and
/// asserts they are what we think they are. Any future drift back to
/// OPLS-like values (θ₀ = 112.7° / k = 58.35 / OPLS dihedral) must
/// touch this file, forcing a code-review conversation.
///
/// Cross-file consistency between this canonical block and the 15
/// `benchmarks/md/pe/performance/polyethylene_tension.cpp` +
/// `tests/{cpu,mpi}/test_pe_*.cpp` files is **not** enforced here —
/// each of those files carries its own constants block (the
/// well-known smell). The deduplication refactor is `pe-ff-
/// constants-extract`, intentionally out of scope.
///
/// Spec: `.claude/specs/pe-ff-migrate-dreiding-ua.md` ac-002.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/units.h"

namespace {

// ── Dreiding-UA polyethylene FF (Tschopp 2010, MSU CAVS ICME) ────────
// Verbatim from the published MSU deck (lammps_msu_paper.in):
//   bond_coeff      1  350     1.53
//   angle_coeff     1   60   109.5
//   dihedral_coeff  1  1.73 -4.49 0.776 6.99 0.0
//   pair_coeff      1 1  0.112  4.01
constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;

constexpr double kBondK_kcal = 350.0;
constexpr double kBondR0 = 1.53;                                // Å
constexpr double kAngleK_kcal = 60.0;                           // kcal/mol/rad²
constexpr double kAngleTheta0 = 109.5 * ATV::units::pi / 180.0; // rad (sp³)
// Direct multi-harmonic coefficients (no OPLS Fourier conversion).
constexpr double kDihC0_kcal = 1.73;
constexpr double kDihC1_kcal = -4.49;
constexpr double kDihC2_kcal = 0.776;
constexpr double kDihC3_kcal = 6.99;
constexpr double kDihC4_kcal = 0.0;

constexpr double kLjEps_kcal = 0.112;
constexpr double kLjSigma = 4.01;  // Å
constexpr double kLjCutoff = 10.5; // Å

std::vector<double> dihedral_Cn_eV() {
    return {
        kDihC0_kcal * kKcalPerMolToEv, kDihC1_kcal * kKcalPerMolToEv,
        kDihC2_kcal * kKcalPerMolToEv, kDihC3_kcal * kKcalPerMolToEv,
        kDihC4_kcal * kKcalPerMolToEv,
    };
}

} // namespace

TEST(PEFFConstants, AngleMatchesMSUPaper) {
    EXPECT_DOUBLE_EQ(kAngleK_kcal, 60.0)
        << "Dreiding-UA angle k must be 60 kcal/mol/rad² (MSU paper)";
    EXPECT_NEAR(kAngleTheta0, 109.5 * ATV::units::pi / 180.0, 1e-12)
        << "Dreiding-UA θ0 must be 109.5° (sp³ tetrahedral, MSU paper)";
}

TEST(PEFFConstants, BondMatchesMSUPaper) {
    EXPECT_DOUBLE_EQ(kBondK_kcal, 350.0)
        << "Dreiding-UA bond k must be 350 kcal/mol/Å² (MSU paper)";
    EXPECT_DOUBLE_EQ(kBondR0, 1.53)
        << "Dreiding-UA r0 must be 1.53 Å (MSU paper)";
}

TEST(PEFFConstants, DihedralMatchesMSUPaper) {
    EXPECT_DOUBLE_EQ(kDihC0_kcal, 1.73);
    EXPECT_DOUBLE_EQ(kDihC1_kcal, -4.49);
    EXPECT_DOUBLE_EQ(kDihC2_kcal, 0.776);
    EXPECT_DOUBLE_EQ(kDihC3_kcal, 6.99);
    EXPECT_DOUBLE_EQ(kDihC4_kcal, 0.0);

    // dihedral_Cn_eV() must produce the same coefficients converted to eV.
    const auto Cn = dihedral_Cn_eV();
    ASSERT_EQ(Cn.size(), 5u);
    EXPECT_DOUBLE_EQ(Cn[0], 1.73 * kKcalPerMolToEv);
    EXPECT_DOUBLE_EQ(Cn[1], -4.49 * kKcalPerMolToEv);
    EXPECT_DOUBLE_EQ(Cn[2], 0.776 * kKcalPerMolToEv);
    EXPECT_DOUBLE_EQ(Cn[3], 6.99 * kKcalPerMolToEv);
    EXPECT_DOUBLE_EQ(Cn[4], 0.0);
}

TEST(PEFFConstants, LJMatchesMSUPaper) {
    EXPECT_DOUBLE_EQ(kLjEps_kcal, 0.112);
    EXPECT_DOUBLE_EQ(kLjSigma, 4.01);
    EXPECT_DOUBLE_EQ(kLjCutoff, 10.5);
}
