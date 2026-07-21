/// @file tests/cpu/scf/test_counterpoise.cpp
/// @brief Boys–Bernardi Counterpoise orchestration identities (He–He/STO-3G).
///
/// Second-level unit test: five-SCF identities, locked BSSE sign (bsse ≤ 0),
/// and ghost_atoms restoration. Minimal system so the suite stays seconds.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/counterpoise.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;

namespace {

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
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(Counterpoise, HeHeSto3gIdentitiesAndRestore) {
    auto dimer = make_he2(2.5);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.max_iter = 50;
    // Seed a non-empty ghost list so restoration is meaningful.
    rhf.ghost_atoms = {7, 8};

    cpu::scf::Counterpoise cp({0}, {1});
    auto r = cp.run(dimer, rhf);

    // Identity: CP interaction = dimer − A(ghost B) − B(ghost A).
    EXPECT_NEAR(r.e_int_cp, r.e_dimer - r.e_a_ghost - r.e_b_ghost, 1e-12);

    // Identity: raw interaction = dimer − free A − free B.
    EXPECT_NEAR(r.e_int_raw, r.e_dimer - r.e_a - r.e_b, 1e-12);

    // Locked sign convention: bsse = (E_A^AB − E_A^A) + (E_B^AB − E_B^B) ≤ 0.
    EXPECT_NEAR(r.bsse, (r.e_a_ghost - r.e_a) + (r.e_b_ghost - r.e_b), 1e-12);
    EXPECT_LE(r.bsse, 0.0);
    EXPECT_TRUE(std::isfinite(r.bsse));
    // |bsse| is finite and (for He–He/STO-3G) non-zero at finite separation.
    EXPECT_GT(std::abs(r.bsse), 0.0);

    // Method state restored after run.
    ASSERT_EQ(rhf.ghost_atoms.size(), 2u);
    EXPECT_EQ(rhf.ghost_atoms[0], 7);
    EXPECT_EQ(rhf.ghost_atoms[1], 8);
}

TEST(Counterpoise, KsWithGhostThrows) {
    auto dimer = make_he2(2.5);
    cpu::scf::RKS rks;
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.ghost_atoms = {1};
    // Direct SCF with ghosts on KS must throw (not Counterpoise-specific).
    EXPECT_THROW(rks.run(dimer), std::invalid_argument);
}

TEST(Counterpoise, EmptyMonomerThrows) {
    EXPECT_THROW((cpu::scf::Counterpoise{{}, {0}}), std::invalid_argument);
    EXPECT_THROW((cpu::scf::Counterpoise{{0}, {}}), std::invalid_argument);
}
