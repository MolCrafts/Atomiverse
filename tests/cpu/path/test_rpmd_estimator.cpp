/// @file tests/cpu/path/test_rpmd_estimator.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/pimd.h"

using namespace ATV;

TEST(RpmdEstimator, AnalyticPositiveAndClassicalLimit) {
    cpu::path::Pimd pimd(4, 300.0, 0.25, 5, 1, 0.0);

    cpu::Frame f(1);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(50.0);

    // Trigger ensure_built via short run so mass/spring set.
    pimd.run(f, 2);
    const double rg2_an = pimd.free_particle_rg2_analytic();
    EXPECT_GT(rg2_an, 0.0);
    EXPECT_TRUE(std::isfinite(rg2_an));
    EXPECT_GT(pimd.radius_of_gyration_sq(), 0.0);

    // Direct formula check (P=4 classical free-polymer Rg²):
    const double hbar = units::h_planck_ev_fs / (2.0 * units::pi);
    const double beta = 1.0 / (units::kB_ev_per_k * 300.0);
    const double m_eff = units::mass_v2_to_energy * 1.0;
    const double P = 4.0;
    const double expect =
        (hbar * beta) * (hbar * beta) / (4.0 * m_eff) * (1.0 - 1.0 / (P * P));
    EXPECT_NEAR(rg2_an, expect, 1e-12);
}
