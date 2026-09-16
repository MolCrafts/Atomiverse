/// @file tests/cpu/bias/test_dihedral_cv.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/dihedral_cv.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

// Planar cis: i(1,0,0), j(0,0,0), k(0,1,0), l(-1,1,0) → φ = π (or −π)
// Planar trans-ish: l at (1,1,0) → φ = 0
cpu::Frame planar_dihedral(double lx, double ly) {
    cpu::Frame f(4);
    f.span(kAtom_x)[0] = 1;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 1;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_x)[3] = lx;
    f.span(kAtom_y)[3] = ly;
    f.span(kAtom_z)[3] = 0;
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

TEST(DihedralCV, TransIsZero) {
    auto f = planar_dihedral(1.0, 1.0);
    cpu::bias::DihedralCV cv(1, 2, 3, 4);
    cv.build(f);
    EXPECT_NEAR(cv.evaluate(f, nullptr), 0.0, 1e-12);
}

TEST(DihedralCV, CisIsPi) {
    auto f = planar_dihedral(-1.0, 1.0);
    cpu::bias::DihedralCV cv(1, 2, 3, 4);
    cv.build(f);
    const double phi = cv.evaluate(f, nullptr);
    EXPECT_NEAR(std::fabs(phi), units::pi, 1e-12);
}

TEST(DihedralCV, GradientFiniteDifference) {
    auto f = planar_dihedral(0.5, 1.0);
    // Lift l slightly out of plane for nonzero sin.
    f.span(kAtom_z)[3] = 0.3;
    cpu::bias::DihedralCV cv(1, 2, 3, 4);
    cv.build(f);
    std::vector<double> g(12, 0.0);
    cv.evaluate(f, g.data());
    const double h = 1e-6;
    for (int a = 0; a < 4; ++a) {
        for (int d = 0; d < 3; ++d) {
            auto fp = f;
            auto fm = f;
            StorageReal* xp = (d == 0)   ? fp.span(kAtom_x)
                              : (d == 1) ? fp.span(kAtom_y)
                                         : fp.span(kAtom_z);
            StorageReal* xm = (d == 0)   ? fm.span(kAtom_x)
                              : (d == 1) ? fm.span(kAtom_y)
                                         : fm.span(kAtom_z);
            xp[a] += h;
            xm[a] -= h;
            const double gp =
                (cv.evaluate(fp, nullptr) - cv.evaluate(fm, nullptr))
                / (2.0 * h);
            EXPECT_NEAR(g[3 * a + d], gp, 2e-5) << "atom " << a << " dim " << d;
        }
    }
}
