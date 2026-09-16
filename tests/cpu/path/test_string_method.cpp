/// @file tests/cpu/path/test_string_method.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/string_method.h"

using namespace ATV;

namespace {

cpu::Frame at(double x, double y) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = x;
    f.span(kAtom_y)[0] = y;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(100.0);
    return f;
}

} // namespace

TEST(StringMethod, ReparameterizeEqualSpacing) {
    cpu::path::StringMethod sm(/*n_images*/ 5, /*step*/ 1e-9, /*max_iter*/ 1,
                               /*f_tol*/ 1e30);
    sm.use_muller_brown();
    // Endpoints of a straight line
    sm.run(at(0, 0), at(4, 0));
    // Arc lengths between consecutive images should be nearly equal
    std::vector<double> s;
    for (int i = 1; i < sm.n_images; ++i) {
        cpu::Frame a(1), b(1);
        sm.image(i - 1, a);
        sm.image(i, b);
        const double dx = b.span(kAtom_x)[0] - a.span(kAtom_x)[0];
        const double dy = b.span(kAtom_y)[0] - a.span(kAtom_y)[0];
        s.push_back(std::hypot(dx, dy));
    }
    const double mean = (s[0] + s[1] + s[2] + s[3]) / 4.0;
    for (double d : s)
        EXPECT_NEAR(d, mean, 1e-6 * mean + 1e-9);
}
