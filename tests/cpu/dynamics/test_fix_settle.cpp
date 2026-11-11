/// @file tests/cpu/md/test_fix_settle.cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_settle.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kDOH = 0.9572;
constexpr double kDHH = 1.5139;

/// Ideal TIP3P triangle, centred on `(cx, cy, cz)`, tags 1/2/3.
cpu::Frame tip3p_ideal(double cx = 0.0, double cy = 0.0, double cz = 0.0) {
    const double half = 0.5 * kDHH;
    const double h = std::sqrt(kDOH * kDOH - half * half);
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = cx;
    f.span(kAtom_y)[0] = cy + h;
    f.span(kAtom_z)[0] = cz;
    f.span(kAtom_x)[1] = cx - half;
    f.span(kAtom_y)[1] = cy;
    f.span(kAtom_z)[1] = cz;
    f.span(kAtom_x)[2] = cx + half;
    f.span(kAtom_y)[2] = cy;
    f.span(kAtom_z)[2] = cz;
    f.span(kAtom_mass)[0] = 16.0;
    f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_mass)[2] = 1.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = i == 0 ? 1 : 2;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = 0.01 * (i + 1);
        f.span(kAtom_vy)[i] = -0.02 * (i + 1);
        f.span(kAtom_vz)[i] = 0.005;
    }
    f.box = cpu::Box::cubic(20.0);
    f.rebuild_tag_index(0, 3);
    return f;
}

/// Ideal TIP3P then perturb the hydrogens off the constraint manifold.
cpu::Frame tip3p_perturbed() {
    cpu::Frame f = tip3p_ideal();
    f.span(kAtom_x)[1] += 0.05;
    f.span(kAtom_y)[2] += 0.03;
    return f;
}

double dist(const cpu::Frame& f, int i, int j) {
    const double dx = f.span(kAtom_x)[i] - f.span(kAtom_x)[j];
    const double dy = f.span(kAtom_y)[i] - f.span(kAtom_y)[j];
    const double dz = f.span(kAtom_z)[i] - f.span(kAtom_z)[j];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

TEST(FixSettle, RestoresGeometry) {
    auto f = tip3p_perturbed();
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    EXPECT_NEAR(dist(f, 0, 1), kDOH, 1e-10);
    EXPECT_NEAR(dist(f, 0, 2), kDOH, 1e-10);
    EXPECT_NEAR(dist(f, 1, 2), kDHH, 1e-10);
}

TEST(FixSettle, VelocityConstraint) {
    auto f = tip3p_perturbed();
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::final_integrate), 0});
    auto bond_rv = [&](int i, int j) {
        const double dx = f.span(kAtom_x)[i] - f.span(kAtom_x)[j];
        const double dy = f.span(kAtom_y)[i] - f.span(kAtom_y)[j];
        const double dz = f.span(kAtom_z)[i] - f.span(kAtom_z)[j];
        const double dvx = f.span(kAtom_vx)[i] - f.span(kAtom_vx)[j];
        const double dvy = f.span(kAtom_vy)[i] - f.span(kAtom_vy)[j];
        const double dvz = f.span(kAtom_vz)[i] - f.span(kAtom_vz)[j];
        return dx * dvx + dy * dvy + dz * dvz;
    };
    // One pass projection — should be much smaller than unconstrained.
    EXPECT_NEAR(bond_rv(0, 1), 0.0, 1e-6);
}

/// SETTLE is the closed-form SHAKE solution, not a reshaping of the
/// unconstrained triangle. On a molecule that is ALREADY rigid and has moved
/// as a rigid body, the constraint has nothing to do and must return exactly
/// the input.
///
/// Reconstructing the ideal triangle from the unconstrained geometry alone
/// passes every bond-length check and still fails this one: with no memory of
/// the reference orientation the reconstruction is free to mirror the
/// molecule, which is what the pre-2026-07 implementation did — a 1e-4 Å
/// perturbation teleported a hydrogen 1.8 Å across the molecule.
TEST(FixSettle, RigidBodyMotionIsUntouched) {
    auto f = tip3p_ideal(5.0, 5.0, 5.0);
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});

    // Rigidly rotate about the centre of mass and translate.
    const double m[3] = {16.0, 1.0, 1.0};
    const double mass_total = m[0] + m[1] + m[2];
    double cx = 0, cy = 0;
    for (int i = 0; i < 3; ++i) {
        cx += m[i] * f.span(kAtom_x)[i] / mass_total;
        cy += m[i] * f.span(kAtom_y)[i] / mass_total;
    }
    const double th = 0.05, ct = std::cos(th), st = std::sin(th);
    double moved[3][3];
    for (int i = 0; i < 3; ++i) {
        const double dx = f.span(kAtom_x)[i] - cx;
        const double dy = f.span(kAtom_y)[i] - cy;
        moved[i][0] = cx + ct * dx - st * dy + 0.01;
        moved[i][1] = cy + st * dx + ct * dy - 0.02;
        moved[i][2] = f.span(kAtom_z)[i] + 0.03;
        f.span(kAtom_x)[i] = moved[i][0];
        f.span(kAtom_y)[i] = moved[i][1];
        f.span(kAtom_z)[i] = moved[i][2];
    }

    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(f.span(kAtom_x)[i], moved[i][0], 1e-12);
        EXPECT_NEAR(f.span(kAtom_y)[i], moved[i][1], 1e-12);
        EXPECT_NEAR(f.span(kAtom_z)[i], moved[i][2], 1e-12);
    }
}

/// The constraint may not move the centre of mass: it is an internal force.
TEST(FixSettle, PreservesCentreOfMass) {
    auto f = tip3p_perturbed();
    const double m[3] = {16.0, 1.0, 1.0};
    const double mass_total = m[0] + m[1] + m[2];
    auto com = [&](int d) {
        double c = 0;
        for (int i = 0; i < 3; ++i)
            c += m[i]
                 * (d == 0   ? f.span(kAtom_x)[i]
                    : d == 1 ? f.span(kAtom_y)[i]
                             : f.span(kAtom_z)[i])
                 / mass_total;
        return c;
    };
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    const double before[3] = {com(0), com(1), com(2)};
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(com(d), before[d], 1e-12);
}

/// Rows are a per-rebuild accident (`atom_modify sort` reorders owned atoms);
/// tags are the identity. A component that caches rows at build() constrains
/// three unrelated atoms after the first spatial sort.
TEST(FixSettle, FollowsTagsAcrossAReorder) {
    auto f = tip3p_perturbed();
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});

    // Permute the rows and republish the tag index, exactly as a spatial sort
    // does.
    auto swap_rows = [&](int a, int b) {
        std::swap(f.span(kAtom_x)[a], f.span(kAtom_x)[b]);
        std::swap(f.span(kAtom_y)[a], f.span(kAtom_y)[b]);
        std::swap(f.span(kAtom_z)[a], f.span(kAtom_z)[b]);
        std::swap(f.span(kAtom_vx)[a], f.span(kAtom_vx)[b]);
        std::swap(f.span(kAtom_vy)[a], f.span(kAtom_vy)[b]);
        std::swap(f.span(kAtom_vz)[a], f.span(kAtom_vz)[b]);
        std::swap(f.span(kAtom_mass)[a], f.span(kAtom_mass)[b]);
        std::swap(f.span(kAtom_type)[a], f.span(kAtom_type)[b]);
        std::swap(f.span(kAtom_tag)[a], f.span(kAtom_tag)[b]);
    };
    swap_rows(0, 2); // O is now row 2, H2 is row 0
    f.clear_tag_index();
    f.rebuild_tag_index(0, 3);

    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});

    const auto& index = f.meta.get(cpu::kTagIndex);
    const int o = index.at(1), h1 = index.at(2), h2 = index.at(3);
    EXPECT_NEAR(dist(f, o, h1), kDOH, 1e-10);
    EXPECT_NEAR(dist(f, o, h2), kDOH, 1e-10);
    EXPECT_NEAR(dist(f, h1, h2), kDHH, 1e-10);
}

/// Every component resolves its Frame columns once at launch entry and then
/// iterates raw pointers; resolving inside the per-atom loops blows the
/// engine's per-launch resolution budget and the driver aborts the run.
TEST(FixSettle, StaysWithinTheResolutionBudget) {
    auto f = tip3p_perturbed();
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    for (auto phase : {md::Phase::initial_integrate, md::Phase::post_integrate,
                       md::Phase::final_integrate}) {
        f.begin_resolution_window(cpu::kResolveBudgetPerLaunch);
        settle.launch({ctx, f, ATV::slot_of(phase), 0});
        EXPECT_NO_THROW(f.end_resolution_window("settle"));
    }
}

/// SETTLE's closed form assumes an isosceles triangle. An asymmetric triple is
/// a job for the iterative solver, and saying so beats mis-constraining it.
TEST(FixSettle, RejectsUnequalHydrogenMasses) {
    auto f = tip3p_ideal();
    f.span(kAtom_mass)[2] = 2.014; // deuterium on one arm only
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    EXPECT_THROW(settle.build({ctx, f}), std::runtime_error);
}

/// A water that straddles a periodic face must still be constrained: the
/// integrator wraps coordinates, so the hydrogens can be an image away from
/// their own oxygen.
TEST(FixSettle, HandlesAMoleculeAcrossAPeriodicFace) {
    auto f = tip3p_ideal(0.0, 0.0, 0.0);
    const double L = 20.0;
    for (int i = 0; i < 3; ++i) {
        if (f.span(kAtom_x)[i] < 0)
            f.span(kAtom_x)[i] += L;
        if (f.span(kAtom_y)[i] < 0)
            f.span(kAtom_y)[i] += L;
    }
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    f.span(kAtom_x)[1] += 0.01;
    f.span(kAtom_y)[2] += 0.01;
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});

    auto dmi = [&](int i, int j) {
        double dx = f.span(kAtom_x)[i] - f.span(kAtom_x)[j];
        double dy = f.span(kAtom_y)[i] - f.span(kAtom_y)[j];
        double dz = f.span(kAtom_z)[i] - f.span(kAtom_z)[j];
        f.box.min_image(dx, dy, dz);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_NEAR(dmi(0, 1), kDOH, 1e-10);
    EXPECT_NEAR(dmi(0, 2), kDOH, 1e-10);
    EXPECT_NEAR(dmi(1, 2), kDHH, 1e-10);
}

/// The analytic velocity stage zeroes all three bond-direction relative
/// velocities in one solve, for every water of a threaded batch.
TEST(FixSettle, VelocityStageIsExactForEveryWater) {
    constexpr int kWaters = 64;
    const cpu::Frame one = tip3p_ideal();
    cpu::Frame f(3 * kWaters);
    for (int k = 0; k < kWaters; ++k)
        for (int a = 0; a < 3; ++a) {
            const int r = 3 * k + a;
            f.span(kAtom_x)[r] = one.span(kAtom_x)[a] + 3.0 * (k % 4);
            f.span(kAtom_y)[r] = one.span(kAtom_y)[a] + 3.0 * ((k / 4) % 4);
            f.span(kAtom_z)[r] = one.span(kAtom_z)[a] + 3.0 * (k / 16);
            f.span(kAtom_mass)[r] = one.span(kAtom_mass)[a];
            f.span(kAtom_tag)[r] = r + 1;
            f.span(kAtom_vx)[r] = 0.01 * std::sin(1.3 * r);
            f.span(kAtom_vy)[r] = 0.02 * std::cos(0.7 * r);
            f.span(kAtom_vz)[r] = -0.015 * std::sin(2.1 * r + 0.4);
        }
    f.box = cpu::Box::cubic(20.0);
    f.rebuild_tag_index(0, f.nlocal());
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    for (int k = 0; k < kWaters; ++k)
        settle.add_water(3 * k + 1, 3 * k + 2, 3 * k + 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::post_integrate), 0});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::final_integrate), 0});
    auto rv = [&](int i, int j) {
        return (f.span(kAtom_x)[i] - f.span(kAtom_x)[j])
                   * (f.span(kAtom_vx)[i] - f.span(kAtom_vx)[j])
               + (f.span(kAtom_y)[i] - f.span(kAtom_y)[j])
                     * (f.span(kAtom_vy)[i] - f.span(kAtom_vy)[j])
               + (f.span(kAtom_z)[i] - f.span(kAtom_z)[j])
                     * (f.span(kAtom_vz)[i] - f.span(kAtom_vz)[j]);
    };
    for (int k = 0; k < kWaters; ++k) {
        const int o = 3 * k, h1 = o + 1, h2 = o + 2;
        EXPECT_NEAR(rv(o, h1), 0.0, 1e-15);
        EXPECT_NEAR(rv(o, h2), 0.0, 1e-15);
        EXPECT_NEAR(rv(h1, h2), 0.0, 1e-15);
    }
}

TEST(FixSettle, RejectsAnAtomInTwoWaters) {
    cpu::Frame f(5);
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = i == 0 ? 16.0 : 1.0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.rebuild_tag_index(0, 5);
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    settle.add_water(1, 4, 5);
    cpu::Context ctx;
    EXPECT_THROW(settle.build({ctx, f}), std::invalid_argument);
}

/// The forward-comm position shift is the lattice translation H·pbc: in a
/// tilted cell the image across y also moves by the xy tilt.
TEST(FixSettle, ForwardShiftFollowsTheTriclinicLattice) {
    auto f = tip3p_ideal(5.0, 5.0, 5.0);
    Mat3 cell = Mat3::cubic(20.0);
    cell(0, 1) = 3.0;  // xy tilt
    cell(0, 2) = -1.5; // xz tilt
    f.box.set_cell(cell);
    cpu::md::FixSettle settle(2.0, kDOH, kDHH);
    settle.add_water(1, 2, 3);
    cpu::Context ctx;
    settle.build({ctx, f});
    settle.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
    settle.set_comm_mode(cpu::md::FixSettle::CommMode::XSHAKE);
    const int list[1] = {0};
    for (const auto& image :
         {std::array<int, 3>{0, 1, 0}, std::array<int, 3>{0, 0, -1},
          std::array<int, 3>{1, 1, 1}}) {
        double buf[3] = {};
        ASSERT_EQ(settle.pack_forward_comm(1, list, buf, 1, image.data()), 3);
        const Vec3 shift = f.box.to_cartesian(
            {double(image[0]), double(image[1]), double(image[2])});
        EXPECT_DOUBLE_EQ(buf[0], f.span(kAtom_x)[0] + shift[0]);
        EXPECT_DOUBLE_EQ(buf[1], f.span(kAtom_y)[0] + shift[1]);
        EXPECT_DOUBLE_EQ(buf[2], f.span(kAtom_z)[0] + shift[2]);
    }
    // The tilt must matter in this check, or it proves nothing.
    double buf[3] = {};
    const int up[3] = {0, 1, 0};
    settle.pack_forward_comm(1, list, buf, 1, up);
    EXPECT_DOUBLE_EQ(buf[0], f.span(kAtom_x)[0] + 3.0);
}
