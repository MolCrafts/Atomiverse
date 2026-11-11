/// @file tests/cpu/dynamics/test_fix_rattle.cpp
/// @brief FixRattle: per-cluster SHAKE position stage and RATTLE velocity
///        stage on GAFF-style X-H clusters (CH3 groups, an O-H bond).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kBox = 20.0;
constexpr double kCH = 1.09;
constexpr double kOH = 0.96;

/// Methyl A (rows 0-3), methyl B straddling the x boundary (rows 4-7),
/// hydroxyl O-H (rows 8-9). Bond type 0 = C-H, 1 = O-H.
cpu::Frame make_clusters() {
    cpu::Frame f(10);
    const double t = kCH / std::sqrt(3.0);
    const double xyz[10][3] = {{5.0, 5.0, 5.0},
                               {5.0 + t, 5.0 + t, 5.0 + t},
                               {5.0 + t, 5.0 - t, 5.0 - t},
                               {5.0 - t, 5.0 + t, 5.0 - t},
                               {19.8, 9.0, 9.0},
                               {19.8 + t - kBox, 9.0 + t, 9.0 + t},
                               {19.8 + t - kBox, 9.0 - t, 9.0 - t},
                               {19.8 - t, 9.0 + t, 9.0 - t},
                               {12.0, 14.0, 3.0},
                               {12.0 + kOH, 14.0, 3.0}};
    const double mass[10] = {12.011, 1.008, 1.008, 1.008,  12.011,
                             1.008,  1.008, 1.008, 15.999, 1.008};
    for (int i = 0; i < 10; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_type)[i] = 0;
    }
    f.box = cpu::Box::cubic(kBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, 10);
    for (int c : {0, 4})
        for (int h = 1; h <= 3; ++h)
            f.add_bond(c, c + h, 0);
    f.add_bond(8, 9, 1);
    return f;
}

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

const int kBonds[7][2] = {{0, 1}, {0, 2}, {0, 3}, {4, 5},
                          {4, 6}, {4, 7}, {8, 9}};

cpu::md::FixRattle make_rattle(double tol = 1e-10) {
    cpu::md::FixRattle rattle(2.0, tol, 200);
    rattle.constrain_bond(0, kCH);
    rattle.constrain_bond(1, kOH);
    return rattle;
}

void drift(cpu::Frame& f) {
    // An unconstrained step: every atom moves differently.
    for (int i = 0; i < 10; ++i) {
        f.span(kAtom_x)[i] += 0.013 * std::sin(1.7 * i);
        f.span(kAtom_y)[i] += 0.011 * std::cos(2.3 * i);
        f.span(kAtom_z)[i] -= 0.009 * std::sin(0.7 * i + 1.0);
    }
}

} // namespace

TEST(FixRattle, CountsOneConstraintPerHeldBond) {
    auto f = make_clusters();
    auto rattle = make_rattle();
    cpu::Context ctx;
    EXPECT_EQ(rattle.n_constraints(ctx, f), 7);
}

TEST(FixRattle, PositionStageRestoresEveryClusterAcrossTheBoundary) {
    auto f = make_clusters();
    auto rattle = make_rattle();
    cpu::Context ctx;
    rattle.build({ctx, f});
    rattle.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    EXPECT_GT(std::abs(bond(f, 4, 5) - kCH), 1e-3);
    rattle.launch({ctx, f, slot_of(md::Phase::post_integrate), 0});
    for (const auto& b : kBonds) {
        const double d = b[1] == 9 ? kOH : kCH;
        // Relative tolerance on |r|^2/d^2 - 1 => |r| within ~tol/2 relative.
        EXPECT_NEAR(bond(f, b[0], b[1]), d, 1e-9) << b[0] << "-" << b[1];
    }
    EXPECT_GE(rattle.last_position_iters(), 1);
}

TEST(FixRattle, PositionCorrectionConservesMomentum) {
    auto f = make_clusters();
    auto rattle = make_rattle();
    cpu::Context ctx;
    rattle.build({ctx, f});
    rattle.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    rattle.launch({ctx, f, slot_of(md::Phase::post_integrate), 0});
    double p[3] = {0, 0, 0};
    for (int i = 0; i < 10; ++i) {
        const double m = f.span(kAtom_mass)[i];
        p[0] += m * f.span(kAtom_vx)[i];
        p[1] += m * f.span(kAtom_vy)[i];
        p[2] += m * f.span(kAtom_vz)[i];
    }
    for (double v : p)
        EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(FixRattle, VelocityStageRemovesBondComponents) {
    auto f = make_clusters();
    auto rattle = make_rattle();
    cpu::Context ctx;
    rattle.build({ctx, f});
    for (int i = 0; i < 10; ++i) {
        f.span(kAtom_vx)[i] = 0.01 * std::cos(3.1 * i);
        f.span(kAtom_vy)[i] = -0.02 * std::sin(1.3 * i);
        f.span(kAtom_vz)[i] = 0.015 * std::cos(0.4 * i);
    }
    rattle.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    rattle.launch({ctx, f, slot_of(md::Phase::final_integrate), 0});
    for (const auto& b : kBonds) {
        double r[3];
        bond(f, b[0], b[1], r);
        const double rv =
            r[0] * (f.span(kAtom_vx)[b[0]] - f.span(kAtom_vx)[b[1]])
            + r[1] * (f.span(kAtom_vy)[b[0]] - f.span(kAtom_vy)[b[1]])
            + r[2] * (f.span(kAtom_vz)[b[0]] - f.span(kAtom_vz)[b[1]]);
        EXPECT_NEAR(rv, 0.0, 1e-10) << b[0] << "-" << b[1];
    }
}

TEST(FixRattle, NonConvergenceThrows) {
    auto f = make_clusters();
    cpu::md::FixRattle rattle(2.0, 1e-12, 1);
    rattle.constrain_bond(0, kCH);
    cpu::Context ctx;
    rattle.build({ctx, f});
    rattle.launch({ctx, f, slot_of(md::Phase::initial_integrate), 0});
    drift(f);
    EXPECT_THROW(rattle.launch({ctx, f, slot_of(md::Phase::post_integrate), 0}),
                 std::runtime_error);
}
