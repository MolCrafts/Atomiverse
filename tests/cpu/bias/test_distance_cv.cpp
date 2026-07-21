/// @file tests/cpu/bias/test_distance_cv.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

TEST(DistanceCV, ValueAndGradient) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.simbox = cpu::Box::cubic(20.0);

    cpu::bias::DistanceCV cv(1, 2);
    cv.build(f);
    std::vector<double> g(6, 0.0);
    const double r = cv.evaluate(f, g.data());
    EXPECT_NEAR(r, 3.0, 1e-12);
    // ∇ξ on i = (r_i−r_j)/r; here r_i−r_j = (−3,0,0), r=3 → (−1,0,0).
    EXPECT_NEAR(g[0], -1.0, 1e-12);
    EXPECT_NEAR(g[3], 1.0, 1e-12);
}

TEST(HarmonicBias, EnergyAndForce) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 2.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_type)[i] = 0;
    }
    f.simbox = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;

    auto cv = std::make_unique<cpu::bias::DistanceCV>(1, 2);
    const double k = 2.0, center = 3.0;
    cpu::bias::HarmonicBias bias(std::move(cv), k, center);
    cpu::Context ctx;
    bias.build({ctx, f});
    bias.launch({ctx, f, 0, 0});

    // ξ=2, d=-1, U=0.5*k*1=1
    EXPECT_NEAR(bias.last_cv(), 2.0, 1e-12);
    EXPECT_NEAR(f.meta.get(kEnergy), 1.0, 1e-12);
    // F = −k d ∇ξ; d=ξ−ξ0=−1 → −k d = +2; ∇ξ_0=−1 → fx0=−2; ∇ξ_1=+1 → fx1=+2.
    EXPECT_NEAR(f.span(kAtom_fx)[0], -2.0, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], 2.0, 1e-12);
}

TEST(HarmonicBias, EnergyOnlySkipsForce) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 2.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_fx)[0] = f.span(kAtom_fx)[1] = 0;
    f.simbox = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;

    cpu::bias::HarmonicBias bias(std::make_unique<cpu::bias::DistanceCV>(1, 2),
                                 2.0, 3.0);
    cpu::Context ctx;
    bias.build({ctx, f});
    cpu::LaunchContext lctx{ctx, f, 0, 0};
    lctx.forces_requested = false;
    bias.launch(lctx);
    EXPECT_NEAR(f.meta.get(kEnergy), 1.0, 1e-12);
    EXPECT_EQ(f.span(kAtom_fx)[0], 0.0);
    EXPECT_EQ(f.span(kAtom_fx)[1], 0.0);
}
