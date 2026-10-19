/// @file test_g_vector.cpp
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include <gtest/gtest.h>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(GVector, SiFCC_30Ry_GCount) {
    AccReal a =
        AccReal(5.43) * cpu::pw::qe_units::angstrom_to_bohr; // 10.26 Bohr
    auto box = cpu::Box::cubic(a);
    AccReal ecut = AccReal(15); // 30 Ry = 15 Hartree
    auto gv = enumerate_g_vectors(box, ecut, Vec3::zeros());
    EXPECT_GE(gv.count(), 1000); // ~3000 for cubic 10.26Bohr cell at 15Ht
    // All G vectors satisfy |G|^2/2 <= ecut
    for (int i = 0; i < gv.count(); ++i)
        EXPECT_LE(gv.ekin[i], ecut + AccReal(1e-10));
}

TEST(GVector, ZeroEcut) {
    auto box = cpu::Box::cubic(AccReal(10));
    auto gv = enumerate_g_vectors(box, AccReal(0), Vec3::zeros());
    EXPECT_EQ(gv.count(), 1); // only G=0
}

TEST(GVector, SortedByNorm) {
    auto box = cpu::Box::cubic(AccReal(10));
    auto gv = enumerate_g_vectors(box, AccReal(2), Vec3::zeros());
    for (int i = 1; i < gv.count(); ++i)
        EXPECT_LE(gv.g[i - 1].norm_squared(),
                  gv.g[i].norm_squared() + AccReal(1e-12));
}

TEST(GVector, NonSymmetricCellUsesColumnReciprocalAndCompleteCutoffSphere) {
    Mat3 h = Mat3::diagonal(4.0, 3.0, 5.0);
    h(0, 1) = 7.0;
    h(1, 2) = 1.2;
    const cpu::Box box(h);
    const Vec3 k{0.17, -0.11, 0.23};
    const AccReal cutoff = 2.0;
    const auto gv = enumerate_g_vectors(box, cutoff, k);
    // Independent integer cube oracle: this cell/cutoff fits within +/- 12.
    const Mat3 b = h.inverse().transpose() * AccReal(2 * units::pi);
    int expected = 0;
    for (int i = -12; i <= 12; ++i)
        for (int j = -12; j <= 12; ++j)
            for (int l = -12; l <= 12; ++l) {
                const Vec3 g = b * Vec3{AccReal(i), AccReal(j), AccReal(l)};
                const Vec3 q = g + k;
                if (0.5 * q.norm_squared() > cutoff + 1e-12)
                    continue;
                ++expected;
                bool found = false;
                for (int n = 0; n < gv.count(); ++n)
                    if (gv.h[n] == i && gv.k[n] == j && gv.l[n] == l) {
                        found = true;
                        for (int d = 0; d < 3; ++d)
                            EXPECT_NEAR(gv.g[n][d], g[d], 1e-12);
                    }
                EXPECT_TRUE(found) << i << "," << j << "," << l;
            }
    EXPECT_EQ(gv.count(), expected);
    for (int n = 0; n < gv.count(); ++n) {
        EXPECT_LT(2 * std::abs(gv.h[n]), gv.nx);
        EXPECT_LT(2 * std::abs(gv.k[n]), gv.ny);
        EXPECT_LT(2 * std::abs(gv.l[n]), gv.nz);
    }
}
