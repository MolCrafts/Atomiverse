/// @file tests/cpu/bias/test_angle_cv.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/angle_cv.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame right_angle_trimer() {
    // i at (1,0,0), j at (0,0,0), k at (0,1,0) → angle π/2
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 1;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 1;
    f.span(kAtom_z)[2] = 0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

TEST(AngleCV, RightAngle) {
    auto f = right_angle_trimer();
    cpu::bias::AngleCV cv(1, 2, 3);
    cv.build(f);
    const double th = cv.evaluate(f, nullptr);
    EXPECT_NEAR(th, 0.5 * ATV::units::pi, 1e-12);
}

TEST(AngleCV, GradientFiniteDifference) {
    auto f = right_angle_trimer();
    // Perturb slightly off right angle for nonzero sin.
    f.span(kAtom_x)[2] = 0.1;
    f.span(kAtom_y)[2] = 1.0;
    cpu::bias::AngleCV cv(1, 2, 3);
    cv.build(f);
    std::vector<double> g(9, 0.0);
    const double th0 = cv.evaluate(f, g.data());
    const double h = 1e-6;
    for (int a = 0; a < 3; ++a) {
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
            EXPECT_NEAR(g[3 * a + d], gp, 1e-5)
                << "atom " << a << " dim " << d << " th0=" << th0;
        }
    }
}
