/// @file tests/cpu/frame/test_pbc.cpp
/// @brief md::wrap_owned keeps r + H·image invariant (LAMMPS Domain::pbc).
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/pbc.h"

using namespace ATV;

namespace {

// Unwrapped coordinate a of row i: r_a + Σ_b H_ab n_b.
double unwrapped(const cpu::Frame& f, int i, int a) {
    const auto image = f.span(kAtom_image);
    const double r[3] = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                         f.span(kAtom_z)[i]};
    double u = r[a];
    for (int b = 0; b < 3; ++b)
        u += f.box.cell()(a, b) * static_cast<double>(image(i, b));
    return u;
}

} // namespace

TEST(WrapOwned, OrthorhombicWrapCountsImages) {
    cpu::Frame f(3);
    f.box = cpu::Box::orthorhombic(10.0, 12.0, 8.0);
    const double x[] = {10.5, -0.25, 3.0};
    const double y[] = {1.0, 25.0, 3.0};
    const double z[] = {-17.0, 4.0, 3.0};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
    }
    f.span(kAtom_image)(2, 0) = 4; // untouched row
    double before[3][3];
    for (int i = 0; i < 3; ++i)
        for (int a = 0; a < 3; ++a)
            before[i][a] = unwrapped(f, i, a);
    cpu::md::wrap_owned(f, 3);

    EXPECT_NEAR(f.span(kAtom_x)[0], 0.5, 1e-12);
    EXPECT_EQ(f.span(kAtom_image)(0, 0), 1);
    EXPECT_EQ(f.span(kAtom_image)(0, 2), -3);
    EXPECT_EQ(f.span(kAtom_image)(1, 0), -1);
    EXPECT_EQ(f.span(kAtom_image)(1, 1), 2);
    EXPECT_EQ(f.span(kAtom_image)(2, 0), 4);
    for (int i = 0; i < 3; ++i)
        for (int a = 0; a < 3; ++a)
            EXPECT_NEAR(unwrapped(f, i, a), before[i][a], 1e-12)
                << "row " << i << " axis " << a;
}

TEST(WrapOwned, TriclinicWrapKeepsUnwrappedPosition) {
    cpu::Frame f(2);
    Mat3 cell = Mat3::diagonal(10.0, 9.0, 8.0);
    cell(0, 1) = 2.5;  // xy tilt
    cell(0, 2) = -1.0; // xz tilt
    cell(1, 2) = 1.5;  // yz tilt
    f.box = cpu::Box(cell);
    // Row 0: one b-vector beyond the cell along y; row 1 below every face.
    f.span(kAtom_x)[0] = 4.0;
    f.span(kAtom_y)[0] = 9.5;
    f.span(kAtom_z)[0] = 2.0;
    f.span(kAtom_x)[1] = -3.0;
    f.span(kAtom_y)[1] = -1.0;
    f.span(kAtom_z)[1] = -0.5;
    double before[2][3];
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            before[i][a] = unwrapped(f, i, a);
    cpu::md::wrap_owned(f, 2);
    EXPECT_EQ(f.span(kAtom_image)(0, 1), 1);
    EXPECT_EQ(f.span(kAtom_image)(1, 2), -1);
    for (int i = 0; i < 2; ++i) {
        const auto s = f.box.to_fractional(
            {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]});
        for (int a = 0; a < 3; ++a) {
            EXPECT_GE(s[a], 0.0);
            EXPECT_LT(s[a], 1.0);
            EXPECT_NEAR(unwrapped(f, i, a), before[i][a], 1e-12);
        }
    }
}

// An int8 counter wrapped to -128 here; the int32 ImageInt must count past
// ±127 crossings exactly (long transport / NEMD-drift runs).
TEST(WrapOwned, ImageCountsPastTheInt8Range) {
    cpu::Frame f(2);
    f.box = cpu::Box::orthorhombic(10.0, 10.0, 10.0);
    f.span(kAtom_image)(0, 0) = 127;
    f.span(kAtom_image)(1, 2) = -128;
    f.span(kAtom_x)[0] = 10.5;   // +1 crossing → 128
    f.span(kAtom_z)[1] = -0.5;   // -1 crossing → -129
    f.span(kAtom_y)[1] = 4005.0; // +400 crossings in one wrap
    cpu::md::wrap_owned(f, 2);
    EXPECT_EQ(f.span(kAtom_image)(0, 0), 128);
    EXPECT_EQ(f.span(kAtom_image)(1, 2), -129);
    EXPECT_EQ(f.span(kAtom_image)(1, 1), 400);
    EXPECT_NEAR(f.span(kAtom_y)[1], 5.0, 1e-9);
}
