/// @file tests/cpu/bias/test_metadynamics_nd.cpp
/// @brief Unit tests for multi-CV well-tempered metadynamics.
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/metadynamics.h"
#include "atomiverse/cpu/bias/metadynamics_1d.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame make_dimer() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 2.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 2.0;
    f.span(kAtom_z)[2] = 0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20);
    return f;
}

} // namespace

TEST(MetadynamicsND, TwoDimHillsAndForces) {
    auto f = make_dimer();
    cpu::Context ctx;
    std::vector<std::unique_ptr<cpu::bias::CollectiveVariable>> cvs;
    cvs.push_back(std::make_unique<cpu::bias::DistanceCV>(1, 2));
    cvs.push_back(std::make_unique<cpu::bias::DistanceCV>(1, 3));
    cpu::bias::Metadynamics meta(std::move(cvs), /*h0=*/0.1, {0.2, 0.2}, 300.0,
                                 10.0, /*pace=*/1);
    meta.build({ctx, f});
    for (int i = 0; i < 6; ++i) {
        // Move slightly so hills are not all at the same s (∇V=0 at centres).
        f.span(kAtom_x)[1] = 2.0 + 0.05 * i;
        f.span(kAtom_y)[2] = 2.0 - 0.03 * i;
        f.meta.get(kEnergy) = 0;
        for (int a = 0; a < 3; ++a)
            f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;
        cpu::LaunchContext lc{ctx, f, 0, i};
        lc.forces_requested = true;
        meta.launch(lc);
    }
    EXPECT_EQ(meta.ndim(), 2);
    EXPECT_GE(meta.n_hills(), 4);
    EXPECT_EQ(meta.last_cv().size(), 2u);
    EXPECT_GT(meta.last_bias_energy(), 0.0);
    // Move off the last hill centre so ∇V is nonzero, re-evaluate once more.
    f.span(kAtom_x)[1] += 0.4;
    f.span(kAtom_y)[2] += 0.4;
    f.meta.get(kEnergy) = 0;
    for (int a = 0; a < 3; ++a)
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;
    cpu::LaunchContext lc2{ctx, f, 0, 100};
    lc2.forces_requested = true;
    meta.launch(lc2);
    double f2 = 0;
    for (int a = 0; a < 3; ++a)
        f2 += f.span(kAtom_fx)[a] * f.span(kAtom_fx)[a]
              + f.span(kAtom_fy)[a] * f.span(kAtom_fy)[a]
              + f.span(kAtom_fz)[a] * f.span(kAtom_fz)[a];
    EXPECT_TRUE(std::isfinite(f2));
    EXPECT_GT(f2, 0.0);
}

TEST(MetadynamicsND, Matches1DWhenNEqualsOne) {
    auto f = make_dimer();
    cpu::Context ctx;
    // Shared trajectory of launches
    std::vector<std::unique_ptr<cpu::bias::CollectiveVariable>> cvs;
    cvs.push_back(std::make_unique<cpu::bias::DistanceCV>(1, 2));
    cpu::bias::Metadynamics meta_n(std::move(cvs), 0.15, {0.25}, 300.0, 8.0, 1);
    cpu::bias::Metadynamics1D meta1(
        std::make_unique<cpu::bias::DistanceCV>(1, 2), 0.15, 0.25, 300.0, 8.0,
        1);
    meta_n.build({ctx, f});
    meta1.build({ctx, f});
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_x)[1] = 1.5 + 0.1 * i;
        f.meta.get(kEnergy) = 0;
        meta_n.launch({ctx, f, 0, i});
        const double un = meta_n.last_bias_energy();
        f.meta.get(kEnergy) = 0;
        meta1.launch({ctx, f, 0, i});
        const double u1 = meta1.last_bias_energy();
        EXPECT_NEAR(un, u1, 1e-12) << "step " << i;
        EXPECT_EQ(meta_n.n_hills(), meta1.n_hills());
    }
}
