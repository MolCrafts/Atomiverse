/// @file tests/cpu/dynamics/test_fix_lincs.cpp
/// @brief FixLincs: matrix-expansion position and velocity stages on a
///        methyl group straddling the boundary, a hydroxyl and a rigid
///        water triangle (constraints sharing atoms at both ends).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_lincs.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr int kAtoms = 9;
constexpr double kBox = 20.0;
constexpr double kCH = 1.09;
constexpr double kOH = 0.96;
constexpr double kWaterOH = 0.9572;
const double kHalfAngle = 0.5 * 104.52 * std::acos(-1.0) / 180.0;
const double kWaterHH = 2.0 * kWaterOH * std::sin(kHalfAngle);

/// Methyl straddling the x boundary (rows 0-3), hydroxyl O-H (rows 4-5),
/// water O H H (rows 6-8). Bond type 0 = C-H, 1 = O-H, 2 = water O-H;
/// angle type 0 = water H-O-H.
cpu::Frame make_clusters() {
    cpu::Frame f(kAtoms);
    const double t = kCH / std::sqrt(3.0);
    const double wx = kWaterOH * std::sin(kHalfAngle);
    const double wy = kWaterOH * std::cos(kHalfAngle);
    const double xyz[kAtoms][3] = {{19.8, 9.0, 9.0},
                                   {19.8 + t - kBox, 9.0 + t, 9.0 + t},
                                   {19.8 + t - kBox, 9.0 - t, 9.0 - t},
                                   {19.8 - t, 9.0 + t, 9.0 - t},
                                   {12.0, 14.0, 3.0},
                                   {12.0 + kOH, 14.0, 3.0},
                                   {5.0, 5.0, 5.0},
                                   {5.0 + wx, 5.0 + wy, 5.0},
                                   {5.0 - wx, 5.0 + wy, 5.0}};
    const double mass[kAtoms] = {12.011, 1.008,  1.008, 1.008, 15.999,
                                 1.008,  15.999, 1.008, 1.008};
    for (int i = 0; i < kAtoms; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_type)[i] = 0;
    }
    f.box = cpu::Box::cubic(kBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, kAtoms);
    for (int h = 1; h <= 3; ++h)
        f.add_bond(0, h, 0);
    f.add_bond(4, 5, 1);
    f.add_bond(6, 7, 2);
    f.add_bond(6, 8, 2);
    f.add_angle(7, 6, 8, 0);
    return f;
}

struct Held {
    int i, j;
    double d;
};
const Held kHeld[7] = {{0, 1, kCH},     {0, 2, kCH},      {0, 3, kCH},
                       {4, 5, kOH},     {6, 7, kWaterOH}, {6, 8, kWaterOH},
                       {7, 8, kWaterHH}};

double bond(const cpu::Frame& f, int i, int j, double* r = nullptr) {
    double d[3] = {f.span(kAtom_x)[i] - f.span(kAtom_x)[j],
                   f.span(kAtom_y)[i] - f.span(kAtom_y)[j],
                   f.span(kAtom_z)[i] - f.span(kAtom_z)[j]};
    for (double& v : d)
        v -= kBox * std::nearbyint(v / kBox);
    if (r)
        for (int k = 0; k < 3; ++k)
            r[k] = d[k];
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

template <class Fix>
Fix constrained(Fix fix) {
    fix.constrain_bond(0, kCH);
    fix.constrain_bond(1, kOH);
    fix.constrain_bond(2, kWaterOH);
    fix.constrain_angle(0, kWaterHH);
    return fix;
}

cpu::md::FixLincs make_lincs(int order = 4) {
    return constrained(cpu::md::FixLincs(2.0, 1e-10, 200, order));
}

void drift(cpu::Frame& f) {
    // An unconstrained step: every atom moves differently.
    for (int i = 0; i < kAtoms; ++i) {
        f.span(kAtom_x)[i] += 0.013 * std::sin(1.7 * i);
        f.span(kAtom_y)[i] += 0.011 * std::cos(2.3 * i);
        f.span(kAtom_z)[i] -= 0.009 * std::sin(0.7 * i + 1.0);
    }
}

void kick(cpu::Frame& f) {
    for (int i = 0; i < kAtoms; ++i) {
        f.span(kAtom_vx)[i] = 0.01 * std::cos(3.1 * i);
        f.span(kAtom_vy)[i] = -0.02 * std::sin(1.3 * i);
        f.span(kAtom_vz)[i] = 0.015 * std::cos(0.4 * i);
    }
}

} // namespace

TEST(FixLincs, IsADistinctAlgorithmWithRattleBindings) {
    cpu::md::FixLincs lincs(1.0);
    EXPECT_EQ(lincs.name(), "lincs");
    EXPECT_EQ(lincs.default_bindings().size(), 4u);
    cpu::Context ctx;
    auto f = make_clusters();
    EXPECT_EQ(make_lincs().n_constraints(ctx, f), 7);
    EXPECT_THROW((cpu::md::FixLincs{1.0, 1e-10, 500, 0}),
                 std::invalid_argument);
}

TEST(FixLincs, PositionStageRestoresEveryHeldDistance) {
    auto f = make_clusters();
    auto lincs = make_lincs();
    cpu::Context ctx;
    lincs.build({ctx, f});
    lincs.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    EXPECT_GT(std::abs(bond(f, 0, 1) - kCH), 1e-3);
    lincs.launch({ctx, f, slot_of(md::Phase::post_integrate), 0});
    for (const Held& h : kHeld)
        EXPECT_NEAR(bond(f, h.i, h.j), h.d, 1e-9) << h.i << "-" << h.j;
    EXPECT_GE(lincs.last_position_iters(), 1);
}

TEST(FixLincs, PositionCorrectionConservesMomentum) {
    auto f = make_clusters();
    auto lincs = make_lincs();
    cpu::Context ctx;
    lincs.build({ctx, f});
    lincs.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    lincs.launch({ctx, f, slot_of(md::Phase::post_integrate), 0});
    double p[3] = {0, 0, 0};
    for (int i = 0; i < kAtoms; ++i) {
        const double m = f.span(kAtom_mass)[i];
        p[0] += m * f.span(kAtom_vx)[i];
        p[1] += m * f.span(kAtom_vy)[i];
        p[2] += m * f.span(kAtom_vz)[i];
    }
    for (double v : p)
        EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(FixLincs, VelocityStageRemovesBondComponents) {
    auto f = make_clusters();
    auto lincs = make_lincs();
    cpu::Context ctx;
    lincs.build({ctx, f});
    kick(f);
    lincs.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    lincs.launch({ctx, f, slot_of(md::Phase::final_integrate), 0});
    for (const Held& h : kHeld) {
        double r[3];
        bond(f, h.i, h.j, r);
        const double rv =
            r[0] * (f.span(kAtom_vx)[h.i] - f.span(kAtom_vx)[h.j])
            + r[1] * (f.span(kAtom_vy)[h.i] - f.span(kAtom_vy)[h.j])
            + r[2] * (f.span(kAtom_vz)[h.i] - f.span(kAtom_vz)[h.j]);
        // The tolerance is relative: |r.v| / d^2.
        EXPECT_NEAR(rv / (h.d * h.d), 0.0, 1e-10) << h.i << "-" << h.j;
    }
    EXPECT_GE(lincs.last_velocity_iters(), 1);
}

// Both solvers displace along the reference bond vectors until every length
// is met, so the converged positions, velocities and virial are one
// solution whatever the linear algebra.
TEST(FixLincs, ConvergedStagesEqualRattle) {
    auto a = make_clusters();
    auto b = a;
    auto lincs = constrained(cpu::md::FixLincs(2.0, 1e-13, 500));
    auto rattle = constrained(cpu::md::FixRattle(2.0, 1e-13, 500));
    cpu::Context ctx;
    lincs.build({ctx, a});
    rattle.build({ctx, b});
    const auto stage = [&](md::Phase phase, bool stress = false) {
        cpu::LaunchContext la{ctx, a, slot_of(phase), 0};
        cpu::LaunchContext lb{ctx, b, slot_of(phase), 0};
        la.stress_requested = lb.stress_requested = stress;
        lincs.launch(la);
        rattle.launch(lb);
    };
    stage(md::Phase::initial_integrate);
    drift(a);
    drift(b);
    kick(a);
    kick(b);
    stage(md::Phase::post_integrate);
    a.meta.get(kStress) = {};
    b.meta.get(kStress) = {};
    stage(md::Phase::post_force, true);
    stage(md::Phase::final_integrate);
    for (auto key : {kAtom_x, kAtom_y, kAtom_z, kAtom_vx, kAtom_vy, kAtom_vz})
        for (int i = 0; i < kAtoms; ++i)
            EXPECT_NEAR(a.span(key)[i], b.span(key)[i], 1e-11) << i;
    double trace = 0.0;
    for (int d = 0; d < 6; ++d) {
        EXPECT_NEAR(a.meta.get(kStress)[d], b.meta.get(kStress)[d], 1e-9) << d;
        trace += std::abs(b.meta.get(kStress)[d]);
    }
    EXPECT_GT(trace, 1e-6);
}

TEST(FixLincs, HigherOrderNeedsFewerCorrectionSweeps) {
    int sweeps[2];
    const int orders[2] = {1, 8};
    for (int n = 0; n < 2; ++n) {
        auto f = make_clusters();
        auto lincs = make_lincs(orders[n]);
        cpu::Context ctx;
        lincs.build({ctx, f});
        lincs.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
        drift(f);
        lincs.launch({ctx, f, slot_of(md::Phase::post_integrate), 0});
        sweeps[n] = lincs.last_position_iters();
    }
    EXPECT_LT(sweeps[1], sweeps[0]);
}

TEST(FixLincs, NonConvergenceThrows) {
    auto f = make_clusters();
    auto lincs = constrained(cpu::md::FixLincs(2.0, 1e-12, 1, 1));
    cpu::Context ctx;
    lincs.build({ctx, f});
    lincs.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    EXPECT_THROW(lincs.launch({ctx, f, slot_of(md::Phase::post_integrate), 0}),
                 std::runtime_error);
}
