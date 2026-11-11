/// @file tests/cpu/md/test_brownian.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/brownian.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

TEST(Brownian, FreeParticleStepVariance) {
    const double D = 0.01; // Å²/fs
    const double dt = 1.0; // fs
    cpu::md::Brownian bd(dt, D, 300.0, /*seed=*/42);
    cpu::Frame f(1);
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(1e6); // free sandbox — Brownian skips wrap

    cpu::Context ctx;
    const int N = 20000;
    double acc = 0;
    for (int i = 0; i < N; ++i) {
        f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
        f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
        bd.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), i});
        const double dx = f.span(kAtom_x)[0];
        acc += dx * dx;
    }
    const double var = acc / N;
    // Per-coordinate variance = 2 D dt
    const double expect = 2.0 * D * dt;
    EXPECT_NEAR(var, expect, 0.08 * expect);
}

TEST(Brownian, NoiseFollowsTagsAcrossPermutationAndChunks) {
    cpu::md::Brownian bd(0.5, 0.01, 300.0, 42);
    cpu::Context context;
    cpu::Frame ordered(3), permuted(3);
    ordered.box = permuted.box = cpu::Box::cubic(1e6);
    const int permutation[3] = {2, 0, 1};
    for (int i = 0; i < 3; ++i) {
        ordered.span(kAtom_tag)[i] = 100 + i;
        permuted.span(kAtom_tag)[i] = 100 + permutation[i];
    }
    bd.launch({context, ordered, slot_of(md::Phase::initial_integrate), 7});
    bd.launch({context, permuted, slot_of(md::Phase::initial_integrate), 7});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(permuted.span(kAtom_x)[i],
                         ordered.span(kAtom_x)[permutation[i]]);
        EXPECT_DOUBLE_EQ(permuted.span(kAtom_y)[i],
                         ordered.span(kAtom_y)[permutation[i]]);
        EXPECT_DOUBLE_EQ(permuted.span(kAtom_z)[i],
                         ordered.span(kAtom_z)[permutation[i]]);
    }
    cpu::Frame single(1);
    single.box = cpu::Box::cubic(1e6);
    single.span(kAtom_tag)[0] = 101;
    bd.launch({context, single, slot_of(md::Phase::initial_integrate), 7});
    EXPECT_DOUBLE_EQ(single.span(kAtom_x)[0], ordered.span(kAtom_x)[1]);
}

TEST(Brownian, NonfiniteForceFailsBeforeMovingAnyAtom) {
    cpu::md::Brownian bd(0.5, 0.01);
    cpu::Frame frame(2);
    cpu::Context context;
    frame.span(kAtom_x)[0] = 1.0;
    frame.span(kAtom_x)[1] = 2.0;
    frame.span(kAtom_fy)[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        bd.launch({context, frame, slot_of(md::Phase::initial_integrate), 0}),
        std::runtime_error);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[1], 2.0);
}

TEST(Brownian, InfiniteParametersAreRejected) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((cpu::md::Brownian{inf, 0.01}), std::invalid_argument);
    EXPECT_THROW((cpu::md::Brownian{0.5, inf}), std::invalid_argument);
    EXPECT_THROW((cpu::md::Brownian{0.5, 0.01, inf}), std::invalid_argument);
}
