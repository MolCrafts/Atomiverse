/// @file tests/cpu/potentials/test_pair_wca.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_wca.h"

using namespace ATV;

TEST(PairWCA, ZeroAtCutoff) {
    const double eps = 1.0, sigma = 1.0;
    const double rc = std::pow(2.0, 1.0 / 6.0) * sigma;
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = rc;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairWCA wca(eps, sigma);
    wca.build({ctx, f});
    wca.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-10);
    wca.destroy({ctx, f});
}

TEST(PairWCA, RepulsiveInside) {
    const double eps = 1.0, sigma = 1.0, r = 0.95;
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairWCA wca(eps, sigma);
    wca.build({ctx, f});
    wca.launch({ctx, f, 0, 0});
    EXPECT_GT(f.meta.get(kEnergy), 0.0);
    EXPECT_LT(f.span(kAtom_fx)[0], 0.0); // atom 0 pushed left
    EXPECT_GT(f.span(kAtom_fx)[1], 0.0);
    wca.destroy({ctx, f});
}
