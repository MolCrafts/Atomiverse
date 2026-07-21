/// @file test_crystal.cpp
/// @brief Manual Frame construction for PW (no magic builders).
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ATV;

namespace {
cpu::Frame make_si_diamond(double a) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_type)[0] = 14;
    f.span(ATV::kAtom_type)[1] = 14;
    f.span(ATV::kAtom_mass)[0] = 28.0855;
    f.span(ATV::kAtom_mass)[1] = 28.0855;
    f.span(ATV::kAtom_x)[0] = 0;
    f.span(ATV::kAtom_y)[0] = 0;
    f.span(ATV::kAtom_z)[0] = 0;
    f.span(ATV::kAtom_x)[1] = a / 4.0;
    f.span(ATV::kAtom_y)[1] = a / 4.0;
    f.span(ATV::kAtom_z)[1] = a / 4.0;
    double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.simbox.set_cell(cell);
    return f;
}
} // namespace

TEST(Crystal, ManualSiDiamond) {
    auto f = make_si_diamond(5.43);
    EXPECT_EQ(f.nlocal(), 2);
    EXPECT_EQ(f.span(ATV::kAtom_type)[0], 14);
    EXPECT_EQ(f.span(ATV::kAtom_type)[1], 14);
    EXPECT_NEAR(f.span(ATV::kAtom_mass)[0], 28.0855, 1e-4);
}

TEST(Crystal, ManualPositions) {
    auto f = make_si_diamond(5.43);
    double a4 = 5.43 / 4.0;
    constexpr double position_tol = 1.0e-12;
    EXPECT_NEAR(f.span(ATV::kAtom_x)[0], 0.0, position_tol);
    EXPECT_NEAR(f.span(ATV::kAtom_y)[0], 0.0, position_tol);
    EXPECT_NEAR(f.span(ATV::kAtom_z)[0], 0.0, position_tol);
    EXPECT_NEAR(f.span(ATV::kAtom_x)[1], a4, position_tol);
    EXPECT_NEAR(f.span(ATV::kAtom_y)[1], a4, position_tol);
    EXPECT_NEAR(f.span(ATV::kAtom_z)[1], a4, position_tol);
}
