/// @file tests/cpu/gpw/test_cpu_gpw_data_table.cpp
/// @brief Unit gate for the in-tree GTH + MOLOPT + D3 tables (in-tree data
///        tables).
///
/// These tables replace a runtime read of a CP2K installation's data directory.
/// The values asserted below are spot-checks against the published sources
/// (Krack, TCA 114, 145 (2005) for GTH-PBE; VandeVondele & Hutter, JCP 127,
/// 114105 (2007) for MOLOPT), pinned so a bad transcription cannot pass. The
/// deeper check that the tables are *physically* right is the finite-difference
/// force oracle in benchmarks/qc/gpw/validation (ac-005).
///
/// Unit-scale: table lookups only, no SCF, no parsing anywhere.

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

#include "atomiverse/cpu/gpw/basis_data/gpw_tables.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

using namespace ATV::cpu::gpw;

namespace {
constexpr double kTol = 1e-12; // the tables are exact decimal transcriptions

static_assert(!std::is_default_constructible_v<GaussianBasis>);
static_assert(!std::is_default_constructible_v<GthPotential>);
static_assert(!std::is_default_constructible_v<Kind>);
} // namespace

// ac-003: the five supported elements {Zn,O,H,Na,Cl} resolve, with the right valence charge.
TEST(GpwDataTable, EdlElementsHaveCorrectZion) {
    EXPECT_EQ(GthPotential(molrs::Element::H, "GTH-PBE-q1").zion, 1);
    EXPECT_EQ(GthPotential(molrs::Element::O, "GTH-PBE-q6").zion, 6);
    EXPECT_EQ(GthPotential(molrs::Element::Na, "GTH-PBE-q9").zion, 9);
    EXPECT_EQ(GthPotential(molrs::Element::Cl, "GTH-PBE-q7").zion, 7);
    EXPECT_EQ(GthPotential(molrs::Element::Zn, "GTH-PBE-q12").zion, 12);
}

// ac-003: Zn-q12 is the element that forces arbitrary-l support — it carries a
// d (l=2) nonlocal projector channel. A table that silently dropped it would
// still "work" and be physically wrong, so assert the channel structure.
TEST(GpwDataTable, ZnPbeQ12HasSPDProjectorChannels) {
    const GthPotential zn = GthPotential(molrs::Element::Zn, "GTH-PBE-q12");
    ASSERT_EQ(zn.channels.size(), 3u); // l = 0, 1, 2
    EXPECT_EQ(zn.channels[0].l, 0);
    EXPECT_EQ(zn.channels[1].l, 1);
    EXPECT_EQ(zn.channels[2].l, 2); // the d channel

    EXPECT_EQ(zn.channels[0].nproj, 3);
    EXPECT_EQ(zn.channels[1].nproj, 2);
    EXPECT_EQ(zn.channels[2].nproj, 1);

    // Spot-check against the published GTH-PBE Zn entry.
    EXPECT_NEAR(zn.rloc, 0.51000000, kTol);
    EXPECT_NEAR(zn.channels[0].r, 0.40031644, kTol);
    EXPECT_NEAR(zn.channels[0].hij(0, 0), 11.53004133, kTol);
    EXPECT_NEAR(zn.channels[2].r, 0.25095885, kTol);
    EXPECT_NEAR(zn.channels[2].hij(0, 0), -14.46695795, kTol);
}

// ac-003: local part of a light element, digit-for-digit.
TEST(GpwDataTable, OxygenPbeQ6LocalPart) {
    const GthPotential o = GthPotential(molrs::Element::O, "GTH-PBE-q6");
    EXPECT_EQ(o.zion, 6);
    EXPECT_GT(o.cloc.size(), 0u);
    EXPECT_GT(o.rloc, 0.0);
    EXPECT_FALSE(o.channels.empty());
}

// ac-003: MOLOPT basis for every supported element, with a pinned H contraction.
TEST(GpwDataTable, MoloptSrBasisForEdlElements) {
    for (molrs::Element el :
         {molrs::Element::H, molrs::Element::O, molrs::Element::Na,
          molrs::Element::Cl, molrs::Element::Zn}) {
        const GaussianBasis b = GaussianBasis(el, "DZVP-MOLOPT-SR-GTH");
        EXPECT_EQ(b.element, el);
        ASSERT_FALSE(b.sets.empty()) << static_cast<int>(el);
        EXPECT_GT(b.sets[0].nexp(), 0) << static_cast<int>(el);
    }

    const GaussianBasis h =
        GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    ASSERT_EQ(h.sets.size(), 1u);
    const GaussianBasisSet& s = h.sets[0];
    EXPECT_EQ(s.nexp(), 5);
    EXPECT_EQ(s.lmin, 0);
    EXPECT_EQ(s.lmax, 1);
    EXPECT_EQ(s.ncoef, 3); // 2 s-shells + 1 p-shell
    EXPECT_NEAR(s.exps[0], 10.068468228533, kTol);
    EXPECT_NEAR(s.exps[4], 0.082193184441, kTol);
    EXPECT_NEAR(s.coef_at(0, 0), -0.033917444900, kTol);
    EXPECT_NEAR(s.coef_at(4, 2), 0.869031854000, kTol);
}

// ac-003: Zn's MOLOPT set carries the f (l=3) shell that broke harmonics().
TEST(GpwDataTable, ZnMoloptCarriesFShell) {
    const GaussianBasis zn =
        GaussianBasis(molrs::Element::Zn, "DZVP-MOLOPT-SR-GTH");
    int lmax = 0;
    for (const GaussianBasisSet& s : zn.sets)
        lmax = std::max(lmax, s.lmax);
    EXPECT_GE(lmax, 3) << "Zn DZVP-MOLOPT-SR-GTH must include the f shell";
}

// ac-004 / no-fallback rule: an unsupported element or name THROWS. A fallback
// value here would be a silently wrong pseudopotential.
TEST(GpwDataTable, UnsupportedEntryThrows) {
    EXPECT_THROW(GthPotential(molrs::Element::Zn, "GTH-NOPE-q12"),
                 std::runtime_error);
    EXPECT_THROW(GthPotential(molrs::Element::Pu, "GTH-PBE-q1"),
                 std::runtime_error);
    EXPECT_THROW(GaussianBasis(molrs::Element::Zn, "DZVP-NOPE-GTH"),
                 std::runtime_error);
    EXPECT_THROW(GaussianBasis(molrs::Element::Pu, "DZVP-MOLOPT-SR-GTH"),
                 std::runtime_error);
}

// ac-013: the in-tree D3 table covers the supported set and only that set.
TEST(GpwDataTable, D3CoversSupportedElementsOnly) {
    for (int z : {1, 8, 11, 17, 18, 30})
        EXPECT_TRUE(D3Params::supports_element(z)) << z;
    for (int z : {6, 7, 26, 94})
        EXPECT_FALSE(D3Params::supports_element(z)) << z;

    const D3Params p{};
    EXPECT_FALSE(p.c6ab[1][1].empty());   // H-H
    EXPECT_FALSE(p.c6ab[30][8].empty());  // Zn-O
    EXPECT_FALSE(p.c6ab[11][17].empty()); // Na-Cl
    EXPECT_GT(p.rcov[30], 0.0);
    EXPECT_GT(p.r2r4[30], 0.0);
    EXPECT_GT(p.r0ab[30][8], 0.0);
}
