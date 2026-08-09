/// @file tests/cpu/planewave/test_pw_method.cpp
/// Concrete PW method API: members + build/run(frame), no PWConfig bag.
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

TEST(PWMethod, DefaultMembersConstructible) {
    PW pw;
    pw.ecut_ry = 12.0;
    pw.k_mesh = 2;
    pw.tol = 1e-5;
    pw.max_iter = 30;
    pw.mix_beta = 0.5;
    pw.pulay_depth = 4;
    EXPECT_DOUBLE_EQ(pw.ecut_ry, 12.0);
    EXPECT_EQ(pw.k_mesh, 2);
    // No default pseudopotential: both sources start empty, and a run with
    // neither set throws rather than substituting a bare-Coulomb potential.
    EXPECT_TRUE(pw.upf_path.empty());
    EXPECT_TRUE(pw.gth_name.empty());
}

TEST(PWMethod, RunGthFiniteEnergy) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 14;
    f.span(kAtom_type)[1] = 14;
    f.span(kAtom_mass)[0] = 28.0855;
    f.span(kAtom_mass)[1] = 28.0855;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 1.3575;
    f.span(kAtom_y)[1] = 1.3575;
    f.span(kAtom_z)[1] = 1.3575;
    f.box = cpu::Box::cubic(5.43);

    // API gate only (build → run → finite energy → result identity), so the
    // settings are the cheapest that still exercise the real path: Γ-only,
    // low cutoff, short budget. A converged Si number is pinned by
    // regressions/pw/pw-lda-si, not here — an 8-k-point tol=1e-6 solve is
    // a 55 s run and has no place in a unit suite.
    PW pw;
    pw.gth_name = "GTH-PBE-q4"; // Si, Z = 14
    pw.ecut_ry = 8.0;
    pw.k_mesh = 1;
    pw.tol = 1e-4;
    pw.max_iter = 6;
    pw.build(f);
    const auto& r = pw.run(f);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_TRUE(std::isfinite(r.energy_ha));
    EXPECT_EQ(&r, &pw.last_result());
}
