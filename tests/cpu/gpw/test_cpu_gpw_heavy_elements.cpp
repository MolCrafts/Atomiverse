/// @file tests/cpu/gpw/test_cpu_gpw_heavy_elements.cpp
/// @brief The in-tree GTH potential and
///        MOLOPT basis parsers ingest the Zn/Na/Cl heavy elements of the CP2K
///        ZnO electric-double-layer system, with projector counts and basis
///        angular momenta read straight from the data files (never hardcoded
///        into the parser). Zn-q12 is the first transition-metal path to be
///        exercised: it carries a nonlocal d (l=2) projector channel and an
///        f shell in DZVP-MOLOPT-SR-GTH.
///
/// All reference numbers below were read by hand from the CP2K data files
///   GTH_POTENTIALS:  Zn GTH-PBE-q12, Na GTH-PBE-q9, Cl GTH-PBE-q7, O/H GTH-PBE.
///   BASIS_MOLOPT:    <El> DZVP-MOLOPT-SR-GTH  (the SR variant the ZnO input uses).

#include <string>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GthPotential;

namespace {

/// @brief Highest angular momentum across all sets of a parsed basis.
int basis_lmax(const GaussianBasis& b) {
    int lmax = -1;
    for (const auto& s : b.sets)
        lmax = std::max(lmax, s.lmax);
    return lmax;
}

/// @brief Total contracted shells for angular momentum `l` across all sets.
int basis_nshell_l(const GaussianBasis& b, int l) {
    int n = 0;
    for (const auto& s : b.sets)
        if (l >= s.lmin && l <= s.lmax)
            n += s.nshell_l(l);
    return n;
}
} // namespace

// ── GTH potentials ──────────────────────────────────────────────────────────

// Zn GTH-PBE-q12: occupation 2 0 10 -> zion 12; local has nexp=0 (no C_i);
// three nonlocal channels l=0 (3 proj), l=1 (2 proj), l=2 (1 proj). The l=2
// (d) channel is the transition-metal path no prior gpw test covered.
TEST(GpwHeavyElements, ZincPbeQ12HasDChannel) {
    GthPotential p = GthPotential(molrs::Element::Zn, "GTH-PBE-q12");
    EXPECT_EQ(p.zion, 12);
    EXPECT_DOUBLE_EQ(p.rloc, 0.51000000);
    EXPECT_TRUE(p.cloc.empty()); // nexp = 0 in the entry

    ASSERT_EQ(p.channels.size(), 3u);
    EXPECT_EQ(p.channels[0].l, 0);
    EXPECT_EQ(p.channels[0].nproj, 3);
    EXPECT_DOUBLE_EQ(p.channels[0].r, 0.40031644);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(0, 0), 11.53004133);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(0, 1), -8.79189815);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(1, 0), -8.79189815); // symmetric fill

    EXPECT_EQ(p.channels[1].l, 1);
    EXPECT_EQ(p.channels[1].nproj, 2);
    EXPECT_DOUBLE_EQ(p.channels[1].r, 0.54318233);

    // The d channel.
    EXPECT_EQ(p.channels[2].l, 2);
    EXPECT_EQ(p.channels[2].nproj, 1);
    EXPECT_DOUBLE_EQ(p.channels[2].r, 0.25095885);
    EXPECT_DOUBLE_EQ(p.channels[2].hij(0, 0), -14.46695795);
}

// Na GTH-PBE-q9 (semicore): occupation 3 6 -> zion 9; two channels l=0,1 each
// with a single projector. No d channel.
TEST(GpwHeavyElements, SodiumPbeQ9) {
    GthPotential p = GthPotential(molrs::Element::Na, "GTH-PBE-q9");
    EXPECT_EQ(p.zion, 9);
    EXPECT_DOUBLE_EQ(p.rloc, 0.18000000);
    ASSERT_EQ(p.cloc.size(), 2u);
    EXPECT_DOUBLE_EQ(p.cloc[0], -8.87992393);
    EXPECT_DOUBLE_EQ(p.cloc[1], 3.94443828);

    ASSERT_EQ(p.channels.size(), 2u);
    EXPECT_EQ(p.channels[0].l, 0);
    EXPECT_EQ(p.channels[0].nproj, 1);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(0, 0), 38.69439882);
    EXPECT_EQ(p.channels[1].l, 1);
    EXPECT_EQ(p.channels[1].nproj, 1);
    EXPECT_DOUBLE_EQ(p.channels[1].hij(0, 0), -7.53326705);
}

// Cl GTH-PBE-q7: occupation 2 5 -> zion 7; l=0 has a 2x2 h, l=1 a single proj.
TEST(GpwHeavyElements, ChlorinePbeQ7) {
    GthPotential p = GthPotential(molrs::Element::Cl, "GTH-PBE-q7");
    EXPECT_EQ(p.zion, 7);
    EXPECT_DOUBLE_EQ(p.rloc, 0.41000000);
    ASSERT_EQ(p.cloc.size(), 1u);
    EXPECT_DOUBLE_EQ(p.cloc[0], -6.39208181);

    ASSERT_EQ(p.channels.size(), 2u);
    EXPECT_EQ(p.channels[0].l, 0);
    ASSERT_EQ(p.channels[0].nproj, 2);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(0, 0), 15.21898983);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(0, 1), -4.93452321);
    EXPECT_DOUBLE_EQ(p.channels[0].hij(1, 0), -4.93452321); // symmetric fill
    EXPECT_DOUBLE_EQ(p.channels[0].hij(1, 1), 6.37044208);
    EXPECT_EQ(p.channels[1].l, 1);
    EXPECT_EQ(p.channels[1].nproj, 1);
    EXPECT_DOUBLE_EQ(p.channels[1].hij(0, 0), 4.33877527);
}

// O/H still parse (regression: the light elements already in the slab + water).
TEST(GpwHeavyElements, OxygenAndHydrogenStillParse) {
    EXPECT_EQ(GthPotential(molrs::Element::O, "GTH-PBE-q6").zion, 6);
    EXPECT_EQ(GthPotential(molrs::Element::H, "GTH-PBE-q1").zion, 1);
}

// No fallback: an unknown potential name throws.
TEST(GpwHeavyElements, MissingPotentialThrows) {
    EXPECT_THROW(GthPotential(molrs::Element::Zn, "GTH-NOPE-q12"),
                 std::runtime_error);
}

// ── DZVP-MOLOPT-SR-GTH basis ────────────────────────────────────────────────

// Zn DZVP-MOLOPT-SR-GTH: set descriptor "2 0 3 6 2 2 2 1" -> lmax 3 with a d
// shell (nshell_l(2)=2) and an f shell (nshell_l(3)=1). The d shell is the
// transition-metal basis path.
TEST(GpwHeavyElements, ZincMoloptSrHasDShell) {
    GaussianBasis b = GaussianBasis(molrs::Element::Zn, "DZVP-MOLOPT-SR-GTH");
    ASSERT_FALSE(b.sets.empty());
    EXPECT_EQ(basis_lmax(b), 3);
    EXPECT_EQ(basis_nshell_l(b, 2), 2); // d
    EXPECT_EQ(basis_nshell_l(b, 3), 1); // f
}

// The remaining four elements parse with their file-stated max angular momenta.
TEST(GpwHeavyElements, MoloptSrAngularMomenta) {
    EXPECT_EQ(
        basis_lmax(GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH")), 1);
    EXPECT_EQ(
        basis_lmax(GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH")), 2);
    EXPECT_EQ(
        basis_lmax(GaussianBasis(molrs::Element::Cl, "DZVP-MOLOPT-SR-GTH")), 2);
    // Na-q9 SR-MOLOPT is s,p only (no d).
    EXPECT_EQ(
        basis_lmax(GaussianBasis(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH")), 1);

    // O carries a single d polarization shell.
    EXPECT_EQ(basis_nshell_l(
                  GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"), 2),
              1);
}

// No fallback: an unknown basis name throws.
TEST(GpwHeavyElements, MissingBasisThrows) {
    EXPECT_THROW(GaussianBasis(molrs::Element::Zn, "DZVP-NOPE-GTH"),
                 std::runtime_error);
}
