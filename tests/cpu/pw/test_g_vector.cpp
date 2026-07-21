/// @file test_g_vector.cpp
#include "atomiverse/core/box.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/pw/internal/g_vector.h"
#include "atomiverse/cpu/pw/qe_units.h"
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(GVector, SiFCC_30Ry_GCount) {
    AccReal a =
        AccReal(5.43) * cpu::pw::qe_units::angstrom_to_bohr; // 10.26 Bohr
    auto box = Box::cubic(a);
    AccReal ecut = AccReal(15); // 30 Ry = 15 Hartree
    auto gv = enumerate_g_vectors(box, ecut, Vec3::zeros());
    EXPECT_GE(gv.count(), 1000); // ~3000 for cubic 10.26Bohr cell at 15Ht
    // All G vectors satisfy |G|^2/2 <= ecut
    for (int i = 0; i < gv.count(); ++i)
        EXPECT_LE(gv.ekin[i], ecut + AccReal(1e-10));
}

TEST(GVector, ZeroEcut) {
    auto box = Box::cubic(AccReal(10));
    auto gv = enumerate_g_vectors(box, AccReal(0), Vec3::zeros());
    EXPECT_EQ(gv.count(), 1); // only G=0
}

TEST(GVector, SortedByNorm) {
    auto box = Box::cubic(AccReal(10));
    auto gv = enumerate_g_vectors(box, AccReal(2), Vec3::zeros());
    for (int i = 1; i < gv.count(); ++i)
        EXPECT_LE(gv.g[i - 1].norm_squared(),
                  gv.g[i].norm_squared() + AccReal(1e-12));
}
