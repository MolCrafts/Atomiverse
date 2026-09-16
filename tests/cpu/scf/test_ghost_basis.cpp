/// @file tests/cpu/scf/test_ghost_basis.cpp
/// @brief Workspace-level ghost-atom semantics (BSSE plumbing).
///
/// No full SCF loop — exercises AtomList / ShellList / internal::IntegralWorkspace only:
/// ghost Z→0 for nuclear + electron counting, basis still from true Z via
/// basis_z, out-of-range ghost indices throw.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

using namespace ATV;
using cpu::scf::AtomList;
using cpu::scf::BasisSet;
using cpu::scf::internal::IntegralWorkspace;
using cpu::scf::ShellList;

namespace {

/// He–He dimer at 2.5 Å (two closed-shell He atoms).
cpu::Frame make_he2(double r_ang = 2.5) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = r_ang;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

// Ghost atom i: n_electrons excludes Z_i, V_nuc excludes pairs with i,
// n_basis still counts i's shells, V matrix has no Z_i nuclear term.
TEST(GhostBasis, AtomListGhostZerosNuclearAndElectrons) {
    auto frame = make_he2();
    AtomList real{frame, 0, {}};
    EXPECT_EQ(real.n_electrons, 4); // 2 He × Z=2
    EXPECT_EQ(real.atoms[0].Z, 2);
    EXPECT_EQ(real.atoms[1].Z, 2);
    EXPECT_EQ(real.basis_z[0], 2);
    EXPECT_EQ(real.basis_z[1], 2);

    // Ghost atom 1: only atom 0 contributes electrons / nuclear charge.
    AtomList ghosted{frame, 0, {1}};
    EXPECT_EQ(ghosted.n_electrons, 2);
    EXPECT_EQ(ghosted.atoms[0].Z, 2);
    EXPECT_EQ(ghosted.atoms[1].Z, 0);
    EXPECT_EQ(ghosted.basis_z[0], 2);
    EXPECT_EQ(ghosted.basis_z[1], 2); // true Z preserved for basis lookup
}

TEST(GhostBasis, ShellListUsesBasisZ) {
    auto frame = make_he2();
    AtomList both{frame, 0, {}};
    ShellList shells_both{both, BasisSet::STO3G};

    AtomList one_ghost{frame, 0, {1}};
    ShellList shells_ghost{one_ghost, BasisSet::STO3G};

    // STO-3G He = 1 s shell per atom → 2 AO for dimer, same with ghost.
    EXPECT_EQ(shells_both.n_basis, 2);
    EXPECT_EQ(shells_ghost.n_basis, 2);
}

TEST(GhostBasis, IntegralWorkspaceGhostSemantics) {
    auto frame = make_he2(2.5);
    cpu::scf::internal::IntegralWorkspace pack_real{frame, BasisSet::STO3G, 0, false, {}, {}};
    cpu::scf::internal::IntegralWorkspace pack_ghost{frame, BasisSet::STO3G, 0, false, {}, {1}};

    EXPECT_EQ(pack_real.n_electrons, 4);
    EXPECT_EQ(pack_ghost.n_electrons, 2);
    EXPECT_EQ(pack_real.n_basis, 2);
    EXPECT_EQ(pack_ghost.n_basis, 2); // basis still present

    // V_nuc: real He2 has Z1*Z2/R; ghosted has one Z=0 → V_nuc = 0.
    const double r_bohr = 2.5 * units::kAngstromInBohr;
    EXPECT_NEAR(pack_real.V_nuc, 4.0 / r_bohr, 1e-10);
    EXPECT_NEAR(pack_ghost.V_nuc, 0.0, 1e-12);

    // Nuclear attraction V: ghost removes the Z_ghost term. For a one-center
    // diagonal V_00 (shell on atom 0), ghosting atom 1 must make |V_00|
    // smaller in magnitude than the real-dimer value (atom 1 no longer
    // attracts density on atom 0).
    EXPECT_LT(std::abs(pack_ghost.V(0, 0)), std::abs(pack_real.V(0, 0)));

    // Ghost atom has Z=0 in the published atom list.
    EXPECT_EQ(pack_ghost.atoms[1].Z, 0);
    EXPECT_EQ(pack_ghost.atoms[0].Z, 2);
}

TEST(GhostBasis, OutOfRangeGhostIndexThrows) {
    auto frame = make_he2();
    EXPECT_THROW((AtomList{frame, 0, {2}}), std::out_of_range);
    EXPECT_THROW((AtomList{frame, 0, {-1}}), std::out_of_range);
    EXPECT_THROW((cpu::scf::internal::IntegralWorkspace{frame, BasisSet::STO3G, 0, false, {}, {99}}),
                 std::out_of_range);
}

TEST(GhostBasis, EmptyGhostIsNoOp) {
    auto frame = make_he2();
    cpu::scf::internal::IntegralWorkspace a{frame, BasisSet::STO3G};
    cpu::scf::internal::IntegralWorkspace b{frame, BasisSet::STO3G, 0, false, {}, {}};
    EXPECT_EQ(a.n_electrons, b.n_electrons);
    EXPECT_EQ(a.n_basis, b.n_basis);
    EXPECT_NEAR(a.V_nuc, b.V_nuc, 1e-14);
}
