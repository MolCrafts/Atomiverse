/// @file tests/cpu/dynamics/test_fix_wall_lj1043.cpp
/// @brief Unit tests for cpu::md::FixWallLJ1043 (Steele 10-4-3).
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_wall_lj1043.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kTwoPi = 6.28318530717958647692528676655900576;

struct Coeffs {
    double c1, c2, c3, c4, offset;
};

Coeffs make_coeffs(double eps, double sig, double rc) {
    const double sqrt2 = std::sqrt(2.0);
    Coeffs c;
    c.c1 = kTwoPi * 2.0 / 5.0 * eps * std::pow(sig, 10);
    c.c2 = kTwoPi * eps * std::pow(sig, 4);
    c.c3 = kTwoPi * sqrt2 / 3.0 * eps * std::pow(sig, 3);
    c.c4 = 0.61 / sqrt2 * sig;
    c.offset = c.c1 / std::pow(rc, 10) - c.c2 / std::pow(rc, 4)
               - c.c3 / std::pow(rc + c.c4, 3);
    return c;
}

double U_shift(const Coeffs& c, double r) {
    return c.c1 / std::pow(r, 10) - c.c2 / std::pow(r, 4)
           - c.c3 / std::pow(r + c.c4, 3) - c.offset;
}

// Fz for zlo wall (side=-1): F = -fwall with fwall = side*(...)
// → Fz = +(10 c1/r^11 - 4 c2/r^5 - 3 c3/(r+c4)^4)  when side=-1? Wait:
// fwall = side * (...), force[dim] -= fwall
// side=-1 → fwall = -(...), force -= fwall = +(...) when (...)>0 repulsive
// Analytic dU/dr (unshifted, r = z - z_wall for lo):
// dU/dr = -10 c1/r^11 + 4 c2/r^5 + 3 c3/(r+c4)^4
// Force on atom Fz = -dU/dz = -dU/dr = 10 c1/r^11 - 4 c2/r^5 - 3 c3/(r+c4)^4
// = - (side) * (10 c1/r^11 - 4 c2/r^5 - 3 c3/...) with side=-1
// = - fwall. Yes Fz = -fwall matches.
double F_lo(const Coeffs& c, double r) {
    const double dUdr = -10.0 * c.c1 / std::pow(r, 11)
                        + 4.0 * c.c2 / std::pow(r, 5)
                        + 3.0 * c.c3 / std::pow(r + c.c4, 4);
    return -dUdr; // Fz for lo wall
}

cpu::Frame one_atom_z(double z) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 10.0;
    f.span(kAtom_y)[0] = 10.0;
    f.span(kAtom_z)[0] = z;
    f.span(kAtom_mass)[0] = 39.948;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::orthorhombic(20.0, 20.0, 30.0);
    f.meta.get(kEnergy) = 0.0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0.0;
    return f;
}

} // namespace

TEST(FixWallLJ1043, AnalyticEnergyAndForceAtSeveralR) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    auto c = make_coeffs(eps, sig, rc);
    cpu::Context ctx;

    for (double r : {3.0, 3.5, 4.0, 5.0, 8.0}) {
        auto f = one_atom_z(r);
        cpu::md::FixWallLJ1043 wall(/*axis=*/2, /*side=*/-1, /*wall_pos=*/0.0,
                                    eps, sig, rc);
        wall.launch({ctx, f, 0, 0});
        EXPECT_NEAR(f.meta.get(kEnergy), U_shift(c, r), 1e-12) << "r=" << r;
        EXPECT_NEAR(f.span(kAtom_fz)[0], F_lo(c, r), 1e-12) << "r=" << r;
        EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-15);
        EXPECT_NEAR(f.span(kAtom_fy)[0], 0.0, 1e-15);
        EXPECT_NEAR(wall.last_energy(), U_shift(c, r), 1e-12);
    }
}

// The 10-4-3 form is defined only on 0 < r < r_c: it diverges as r->0+ and its
// third term turns attractive past the pole at r = -(0.61/sqrt2)*sigma. An atom
// at r <= 0 therefore has no wall energy, and skipping it would delete the wall
// for that atom permanently (F = 0 => it can never re-enter).
TEST(FixWallLJ1043, ThrowsOnWallSurface) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    cpu::Context ctx;
    auto f = one_atom_z(0.0); // exactly on the lo wall => delta == 0
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    EXPECT_THROW(wall.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(FixWallLJ1043, ThrowsBehindLoWallNamingAxisAndCoord) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    cpu::Context ctx;
    auto f = one_atom_z(-0.5);
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    try {
        wall.launch({ctx, f, 0, 0});
        FAIL() << "expected std::runtime_error for an atom behind the wall";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("axis z"), std::string::npos) << msg;
        EXPECT_NE(msg.find("-0.5"), std::string::npos) << msg;
    }
}

TEST(FixWallLJ1043, ThrowsBehindHiWall) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    cpu::Context ctx;
    auto f = one_atom_z(26.0);
    cpu::md::FixWallLJ1043 wall(2, +1, /*wall_pos=*/25.0, eps, sig, rc);
    EXPECT_THROW(wall.launch({ctx, f, 0, 0}), std::runtime_error);
}

// The boundary is r > 0 exactly, not "r beyond some tolerance".
TEST(FixWallLJ1043, StrictlyPositiveSeparationIsLegal) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    cpu::Context ctx;
    auto f = one_atom_z(1e-12);
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    EXPECT_NO_THROW(wall.launch({ctx, f, 0, 0}));
}

// A configuration with no defined wall energy must not deposit a partial one.
TEST(FixWallLJ1043, RejectedConfigurationLeavesFrameUntouched) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    cpu::Context ctx;
    auto f = one_atom_z(-0.5);
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    EXPECT_THROW(wall.launch({ctx, f, 0, 0}), std::runtime_error);
    EXPECT_EQ(f.meta.get(kEnergy), 0.0);
    EXPECT_EQ(f.span(kAtom_fz)[0], 0.0);
}

TEST(FixWallLJ1043, ZeroBeyondCutoff) {
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    auto f = one_atom_z(rc + 0.1);
    cpu::Context ctx;
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    wall.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
}

TEST(FixWallLJ1043, ZeroAtCutoffAfterShift) {
    // Just inside cutoff: energy → 0 from the shift.
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    auto c = make_coeffs(eps, sig, rc);
    const double r = rc - 1e-4;
    auto f = one_atom_z(r);
    cpu::Context ctx;
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, eps, sig, rc);
    wall.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U_shift(c, r), 1e-12);
    EXPECT_LT(std::fabs(f.meta.get(kEnergy)), 1e-5);
}

TEST(FixWallLJ1043, HiWallMirror) {
    // Wall at z=30 (zhi), atom at z=26 → delta = 4. Same E as lo at r=4.
    const double eps = 0.010338, sig = 3.40, rc = 12.0;
    auto c = make_coeffs(eps, sig, rc);
    auto f = one_atom_z(26.0);
    cpu::Context ctx;
    cpu::md::FixWallLJ1043 wall(2, /*side=*/+1, /*wall_pos=*/30.0, eps, sig,
                                rc);
    wall.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U_shift(c, 4.0), 1e-12);
    // Hi wall: force pushes atom down when repulsive → Fz opposite of lo.
    EXPECT_NEAR(f.span(kAtom_fz)[0], -F_lo(c, 4.0), 1e-12);
}

TEST(FixWallLJ1043, RejectsInvalidParams) {
    EXPECT_THROW(cpu::md::FixWallLJ1043(3, -1, 0.0, 0.01, 3.4, 12.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::md::FixWallLJ1043(2, 0, 0.0, 0.01, 3.4, 12.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::md::FixWallLJ1043(2, -1, 0.0, 0.0, 3.4, 12.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::md::FixWallLJ1043(2, -1, 0.0, 0.01, 0.0, 12.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::md::FixWallLJ1043(2, -1, 0.0, 0.01, 3.4, 0.0),
                 std::invalid_argument);
}

TEST(FixWallLJ1043, Name) {
    cpu::md::FixWallLJ1043 wall(2, -1, 0.0, 0.01, 3.4, 12.0);
    EXPECT_EQ(wall.name(), "fix_wall_lj1043");
}
