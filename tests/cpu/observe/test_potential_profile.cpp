/// @file tests/cpu/observe/test_potential_profile.cpp
/// @brief Unit: PotentialProfile planar Poisson mid-gap E and Ψ(z).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/potential_profile.h"

using namespace ATV;
using cpu::PotentialProfile;

namespace {

/// Two-sheet parallel plate: +Q at z=5.5, −Q at z=14.5; box 10×10×20.
cpu::Frame two_sheet_plate(double Q = 1.0) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 5.0;
    f.span(kAtom_y)[0] = 5.0;
    f.span(kAtom_z)[0] = 5.5;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_charge)[0] = Q;
    f.span(kAtom_mass)[0] = 12.0;
    f.span(kAtom_tag)[0] = 1;

    f.span(kAtom_x)[1] = 5.0;
    f.span(kAtom_y)[1] = 5.0;
    f.span(kAtom_z)[1] = 14.5;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_charge)[1] = -Q;
    f.span(kAtom_mass)[1] = 12.0;
    f.span(kAtom_tag)[1] = 2;

    f.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);
    return f;
}

} // namespace

TEST(PotentialProfile, ParallelPlateMidGapFieldAndDrop) {
    // Offline pin (REFERENCE.md): σ = 0.01 e/Å², E = 4π ke σ
    constexpr double ke = units::coulomb;
    constexpr double four_pi_ke = 4.0 * units::pi * ke;
    constexpr double A = 100.0;
    constexpr double sigma = 1.0 / A;
    const double E_gap = four_pi_ke * sigma;
    // Discrete right-edge Ψ with n=20, dz=1, sheets in bins 5 and 14:
    // ΔΨ = E_gap * 9.0 (center-to-center separation)
    const double psi_drop = -E_gap * 9.0;

    auto frame = two_sheet_plate(1.0);
    PotentialProfile obs(/*n_bins=*/20, /*z_min=*/0.0, /*z_max=*/20.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);

    EXPECT_EQ(obs.n_bins(), 20);
    EXPECT_NEAR(obs.dz(), 1.0, 1e-15);

    // Mid-gap bin 9 (z=9.5): continuum field
    EXPECT_NEAR(obs.field()[9], E_gap, 1e-12);
    // Outside left / right vacuum
    EXPECT_NEAR(obs.field()[0], 0.0, 1e-12);
    EXPECT_NEAR(obs.field()[19], 0.0, 1e-12);

    // Potential: left vacuum zero-ish, right vacuum = psi_drop
    EXPECT_NEAR(obs.psi()[0], 0.0, 1e-12);
    EXPECT_NEAR(obs.psi()[19], psi_drop, 1e-12);
    EXPECT_NEAR(obs.potential_drop(), psi_drop, 1e-12);
    EXPECT_NEAR(frame.meta.get(kPotentialDrop), psi_drop, 1e-12);

    // Charge density in electrode bins
    EXPECT_NEAR(obs.rho()[5], sigma / obs.dz(), 1e-12);
    EXPECT_NEAR(obs.rho()[14], -sigma / obs.dz(), 1e-12);
}

TEST(PotentialProfile, InvalidCtorThrows) {
    EXPECT_THROW(PotentialProfile(0, 0.0, 10.0), std::invalid_argument);
    EXPECT_THROW(PotentialProfile(-1, 0.0, 10.0), std::invalid_argument);
    EXPECT_THROW(PotentialProfile(10, 5.0, 5.0), std::invalid_argument);
    EXPECT_THROW(PotentialProfile(10, 5.0, 4.0), std::invalid_argument);
}

TEST(PotentialProfile, EmptyWindowNoChargeZeroProfile) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 50.0; // outside window
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(10.0);

    PotentialProfile obs(10, 0.0, 10.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, f, 0, 0};
    obs.launch(lc);
    for (double e : obs.field())
        EXPECT_NEAR(e, 0.0, 1e-15);
    for (double p : obs.psi())
        EXPECT_NEAR(p, 0.0, 1e-15);
    EXPECT_NEAR(obs.potential_drop(), 0.0, 1e-15);
}
