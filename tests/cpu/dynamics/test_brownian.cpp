/// @file tests/cpu/md/test_brownian.cpp
#include <gtest/gtest.h>

#include <cmath>
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
        bd.launch({ctx, f, ATV::slot_of(md::Phase::initial_integrate), 0});
        const double dx = f.span(kAtom_x)[0];
        acc += dx * dx;
    }
    const double var = acc / N;
    // Per-coordinate variance = 2 D dt
    const double expect = 2.0 * D * dt;
    EXPECT_NEAR(var, expect, 0.08 * expect);
}
