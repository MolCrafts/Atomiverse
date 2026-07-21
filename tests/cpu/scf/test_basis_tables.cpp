/// @file tests/cpu/scf/test_basis_tables.cpp
/// @brief Unit: static basis-table structure for the third-period extension
///        (hf-capability-gaps-05-basis) — no SCF is ever run here.
///
/// Pinned oracle: PySCF 2.14.0 `gto.basis.load(<name>, <El>)` (the same
/// `mol._basis` dictionary the generator consumes), run externally per
/// docs/tutorial/appendix-pyscf-oracle.md:
///
///   from pyscf import gto
///   b = gto.basis.load("6-31+g*", "O")     # and "aug-cc-pvdz", per element
///
/// General contractions (multi-column blocks, e.g. the cc-pVDZ s block) are
/// split into one ContractedShell per column by tools/gen_basis_data.py;
/// the l sequences / shell counts below are pinned AFTER that split (the
/// spanned AO space is identical).

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_data/aug_ccpvdz.h"
#include "atomiverse/cpu/scf/basis_data/def2_svp.h"
#include "atomiverse/cpu/scf/basis_data/def2_tzvp.h"
#include "atomiverse/cpu/scf/basis_data/pople_631g.h"
#include "atomiverse/cpu/scf/basis_data/pople_631gs.h"
#include "atomiverse/cpu/scf/basis_data/pople_631pgs.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"

using namespace ATV;
using cpu::scf::AtomList;
using cpu::scf::BasisSet;
using cpu::scf::ShellList;
using cpu::scf::angular_convention;
using cpu::scf::basis_set_name;
using cpu::scf::basis_data::AngularConvention;
using cpu::scf::basis_data::AtomBasis;

namespace {

std::vector<int> l_seq(const AtomBasis& ab) {
    std::vector<int> out;
    for (const auto& sh : ab.shells)
        out.push_back(sh.l);
    return out;
}

int n_cart(const AtomBasis& ab) {
    int n = 0;
    for (const auto& sh : ab.shells)
        n += cpu::scf::n_cart_of(sh.l);
    return n;
}

int n_sph(const AtomBasis& ab) {
    int n = 0;
    for (const auto& sh : ab.shells)
        n += 2 * sh.l + 1;
    return n;
}

cpu::Frame one_atom(int Z) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = Z;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

// ── 6-31+G* table structure vs PySCF gto.basis.load("6-31+g*", El) ────

TEST(BasisTables, Pople631pGsStructure) {
    const auto& t = cpu::scf::basis_data::pople_631pgs();
    ASSERT_GE(static_cast<int>(t.size()), 21); // Z=0..20 (K/Ca included)

    // H: 2 s shells; exponents 18.731137 .. 0.1612778.
    const auto& h = t[1];
    EXPECT_EQ(l_seq(h), (std::vector<int>{0, 0}));
    EXPECT_EQ(n_cart(h), 2);
    EXPECT_DOUBLE_EQ(h.shells.front().alpha.front(), 18.731137);
    EXPECT_DOUBLE_EQ(h.shells.back().alpha.back(), 0.1612778);

    // O: 4s3p1d (diffuse sp as the 4th s + 3rd p); 19 Cartesian AOs.
    const auto& o = t[8];
    EXPECT_EQ(l_seq(o), (std::vector<int>{0, 0, 0, 0, 1, 1, 1, 2}));
    EXPECT_EQ(n_cart(o), 19);
    EXPECT_DOUBLE_EQ(o.shells.front().alpha.front(), 5484.6717);
    EXPECT_DOUBLE_EQ(o.shells.front().coeff.front(), 0.0018311);
    EXPECT_DOUBLE_EQ(o.shells.back().alpha.back(), 0.8);

    // Na and Cl (third period): 5s4p1d; 23 Cartesian AOs.
    const std::vector<int> p3{0, 0, 0, 0, 0, 1, 1, 1, 1, 2};
    const auto& na = t[11];
    EXPECT_EQ(l_seq(na), p3);
    EXPECT_EQ(n_cart(na), 23);
    EXPECT_DOUBLE_EQ(na.shells.front().alpha.front(), 9993.2);
    EXPECT_DOUBLE_EQ(na.shells.back().alpha.back(), 0.175);
    const auto& cl = t[17];
    EXPECT_EQ(l_seq(cl), p3);
    EXPECT_EQ(n_cart(cl), 23);
    EXPECT_DOUBLE_EQ(cl.shells.front().alpha.front(), 25180.1);
    EXPECT_DOUBLE_EQ(cl.shells.front().coeff.front(), 0.001833);
    EXPECT_DOUBLE_EQ(cl.shells.back().alpha.back(), 0.75);

    // He is NOT in this table (kept to the sibling 6-31G* coverage).
    EXPECT_TRUE(t[2].shells.empty());
}

// ── aug-cc-pVDZ table structure vs PySCF (after column split) ──────────

TEST(BasisTables, AugCcPVDZStructure) {
    const auto& t = cpu::scf::basis_data::aug_ccpvdz();
    ASSERT_GE(static_cast<int>(t.size()), 19); // Z=0..18

    // H: 3s2p; 9 spherical AOs.
    const auto& h = t[1];
    EXPECT_EQ(l_seq(h), (std::vector<int>{0, 0, 0, 1, 1}));
    EXPECT_EQ(n_sph(h), 9);
    EXPECT_DOUBLE_EQ(h.shells.front().alpha.front(), 13.01);
    EXPECT_DOUBLE_EQ(h.shells.front().coeff.front(), 0.019685);
    EXPECT_DOUBLE_EQ(h.shells.back().alpha.back(), 0.141);

    // O: 4s3p2d after splitting the 2-column 8-primitive s block;
    // 23 spherical / 25 Cartesian AOs (PySCF mol.nao = 23).
    const auto& o = t[8];
    EXPECT_EQ(l_seq(o), (std::vector<int>{0, 0, 0, 0, 1, 1, 1, 2, 2}));
    EXPECT_EQ(n_sph(o), 23);
    EXPECT_EQ(n_cart(o), 25);
    // Split columns share the primitive set: shells 0 and 1 have the same
    // 8 exponents (general contraction), different coefficient columns.
    ASSERT_EQ(o.shells[0].alpha.size(), 8u);
    EXPECT_EQ(o.shells[0].alpha, o.shells[1].alpha);
    EXPECT_NE(o.shells[0].coeff, o.shells[1].coeff);
    EXPECT_DOUBLE_EQ(o.shells.front().alpha.front(), 11720.0);
    EXPECT_DOUBLE_EQ(o.shells.front().coeff.front(), 0.00071);
    EXPECT_DOUBLE_EQ(o.shells.back().alpha.back(), 0.332);

    // Cl (third period): 6s5p2d after splitting (3-column s, 2-column p);
    // 27 spherical AOs (PySCF mol.nao = 27).
    const auto& cl = t[17];
    EXPECT_EQ(l_seq(cl), (std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2}));
    EXPECT_EQ(n_sph(cl), 27);
    EXPECT_DOUBLE_EQ(cl.shells.front().alpha.front(), 127900.0);
    EXPECT_DOUBLE_EQ(cl.shells.front().coeff.front(), 0.000241153);
    EXPECT_DOUBLE_EQ(cl.shells.back().alpha.back(), 0.196);

    // K/Ca are NOT in aug-cc-pVDZ (Kendall-Dunning-Harrison stop at Ar).
    EXPECT_TRUE(static_cast<int>(t.size()) <= 19 || t[19].shells.empty());
}

// ── Four regenerated tables carry the third period (Z=11–18) ──────────

TEST(BasisTables, ThirdPeriodRowsNonEmpty) {
    const auto& g = cpu::scf::basis_data::pople_631g();
    const auto& gs = cpu::scf::basis_data::pople_631gs();
    const auto& pgs = cpu::scf::basis_data::pople_631pgs();
    const auto& svp = cpu::scf::basis_data::def2_svp();
    const auto& tzvp = cpu::scf::basis_data::def2_tzvp();
    const auto& aug = cpu::scf::basis_data::aug_ccpvdz();
    for (int z = 11; z <= 18; ++z) {
        EXPECT_FALSE(g[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(gs[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(pgs[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(svp[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(tzvp[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(aug[static_cast<std::size_t>(z)].shells.empty()) << z;
    }
    // Pople sets additionally carry K/Ca (defined in PySCF's 6-31G family).
    for (int z = 19; z <= 20; ++z) {
        EXPECT_FALSE(g[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(gs[static_cast<std::size_t>(z)].shells.empty()) << z;
        EXPECT_FALSE(pgs[static_cast<std::size_t>(z)].shells.empty()) << z;
    }
}

// ── Enum registration: name, convention, ShellList dispatch ───────────

TEST(BasisTables, NewEnumNamesAndConvention) {
    EXPECT_STREQ(basis_set_name(BasisSet::Pople631pGs), "6-31+G*");
    EXPECT_STREQ(basis_set_name(BasisSet::AugCcPVDZ), "aug-cc-pVDZ");
    EXPECT_EQ(angular_convention(BasisSet::Pople631pGs),
              AngularConvention::Cartesian);
    EXPECT_EQ(angular_convention(BasisSet::AugCcPVDZ),
              AngularConvention::Spherical);
}

TEST(BasisTables, ShellListConstructsForNewSets) {
    // Cl: 6-31+G* → 23 Cartesian AOs; aug-cc-pVDZ → 29 Cartesian shells
    // (spherical collapse to 27 happens downstream in IntegralPack).
    const auto f = one_atom(17);
    const AtomList atoms{f, -1};
    EXPECT_EQ(atoms.n_electrons, 18);
    const ShellList pgs{atoms, BasisSet::Pople631pGs};
    EXPECT_EQ(pgs.n_basis, 23);
    const ShellList aug{atoms, BasisSet::AugCcPVDZ};
    EXPECT_EQ(aug.n_basis, 29);
}

TEST(BasisTables, UnsupportedZThrowsWithBasisName) {
    const auto he = one_atom(2); // He: absent from 6-31+G* table
    const AtomList he_atoms{he, 0};
    try {
        ShellList{he_atoms, BasisSet::Pople631pGs};
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("6-31+G*"), std::string::npos)
            << e.what();
    }

    const auto k = one_atom(19); // K: absent from aug-cc-pVDZ
    const AtomList k_atoms{k, 0};
    try {
        ShellList{k_atoms, BasisSet::AugCcPVDZ};
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("aug-cc-pVDZ"), std::string::npos)
            << e.what();
    }
}
