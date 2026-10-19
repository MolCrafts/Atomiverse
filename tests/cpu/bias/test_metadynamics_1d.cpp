#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/metadynamics_1d.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
using namespace ATV;
TEST(Metadynamics1D, HillsGrow) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 2;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.box = cpu::Box::cubic(20);
    cpu::Context ctx;
    auto cv = std::make_unique<cpu::bias::DistanceCV>(1, 2);
    cpu::bias::Metadynamics1D meta(std::move(cv), 0.1, 0.2, 300.0, 10.0, 1);
    meta.build({ctx, f});
    for (int i = 0; i < 5; ++i) {
        f.meta.get(kEnergy) = 0;
        meta.launch({ctx, f, 0, i});
    }
    EXPECT_GE(meta.n_hills(), 3);
    EXPECT_GT(meta.last_bias_energy(), 0.0);
}

TEST(Metadynamics1D,
     ForceInitializationPreservesHistoryAndEvaluatesExistingHills) {
    cpu::Frame f(2);
    f.span(kAtom_x)[1] = 2.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20);
    cpu::Context ctx;
    cpu::bias::Metadynamics1D meta(
        std::make_unique<cpu::bias::DistanceCV>(1, 2), 0.1, 0.2, 300.0, 1.0, 2);
    meta.build({ctx, f});
    const auto evaluate = [&](bool setup) {
        ctx.exec.initializing_forces = setup;
        f.meta.get(kEnergy) = 0;
        for (int i = 0; i < 2; ++i)
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        meta.launch({ctx, f, 0, 0});
    };
    for (int i = 0; i < 3; ++i) {
        evaluate(true);
        EXPECT_EQ(meta.n_hills(), 0);
        EXPECT_DOUBLE_EQ(f.meta.get(kEnergy), 0);
    }
    // Setup does not count toward the two-launch deposition cadence.
    for (int i = 0; i < 4; ++i) {
        evaluate(false);
        EXPECT_EQ(meta.n_hills(), i < 2 ? 0 : 1);
    }
    f.span(kAtom_x)[1] = 2.1;
    const double expected_energy = 0.1 * std::exp(-0.125);
    for (int i = 0; i < 3; ++i) {
        evaluate(
            true); // Production counter is exactly at a deposition boundary.
        EXPECT_EQ(meta.n_hills(), 1);
        EXPECT_NEAR(f.meta.get(kEnergy), expected_energy, 1e-13);
        EXPECT_NEAR(f.span(kAtom_fx)[1], 2.5 * expected_energy, 1e-13);
        EXPECT_NEAR(f.span(kAtom_fx)[0], -2.5 * expected_energy, 1e-13);
    }
    evaluate(false);
    EXPECT_EQ(meta.n_hills(), 2);
    EXPECT_NEAR(f.meta.get(kEnergy), expected_energy + 0.1, 1e-13);
}
