/// @file tests/cpu/dynamics/test_fix_dfield.cpp
/// @brief Unit tests for cpu::md::FixDfield — vacuum foothold (E ≡ D) and
///        Zhang–Sprik constant-D polarization feedback (E_z = D_z − 4πk_e P_z).
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_dfield.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kFourPiK = 4.0 * units::pi * units::coulomb;

/// ±1 e pair on the z axis: +1 at z=6, −1 at z=4, L=10 (V=1000).
cpu::Frame charge_pair() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 5.0;
        f.span(kAtom_y)[i] = 5.0;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
        for (int d = 0; d < 3; ++d)
            f.span(kAtom_image)[3 * i + d] = 0;
    }
    f.span(kAtom_z)[0] = 6.0;
    f.span(kAtom_z)[1] = 4.0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0.0;
    return f;
}

double launch_once(cpu::md::FixDfield& d, cpu::Frame& f) {
    cpu::Context ctx;
    d.launch({ctx, f, /*stage=*/0, /*step=*/0});
    return d.last_energy();
}

} // namespace

TEST(FixDfield, FootholdForceIsQTimesD) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.1, -0.05, 0.2);
    launch_once(d, f);
    EXPECT_NEAR(double(f.span(kAtom_fx)[0]), 0.1, 1e-15);
    EXPECT_NEAR(double(f.span(kAtom_fy)[0]), -0.05, 1e-15);
    EXPECT_NEAR(double(f.span(kAtom_fz)[0]), 0.2, 1e-15);
    EXPECT_NEAR(double(f.span(kAtom_fz)[1]), -0.2, 1e-15);
}

TEST(FixDfield, FootholdEnergyIsMinusQDdotR) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    const double U = launch_once(d, f);
    // U = -q0 D z0 - q1 D z1 = -0.5*6 + 0.5*4 = -1.0
    EXPECT_NEAR(U, -1.0, 1e-14);
    EXPECT_NEAR(f.meta.get(kEnergy), -1.0, 1e-14);
}

TEST(FixDfield, FootholdFieldStaysD) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    launch_once(d, f);
    EXPECT_EQ(d.last_Ez(), 0.5);
}

TEST(FixDfield, FeedbackPolarizationIsQdOverV) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    d.polarization_feedback = true;
    launch_once(d, f);
    EXPECT_NEAR(d.last_Pz(), 2.0e-3, 1e-18); // q*d/V = 1*2/1000
}

TEST(FixDfield, FeedbackFieldIsDMinus4PiKP) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    d.polarization_feedback = true;
    launch_once(d, f);
    EXPECT_NEAR(d.last_Ez(), 0.5 - kFourPiK * 2.0e-3, 1e-15);
}

TEST(FixDfield, FeedbackForceIsQTimesE) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    d.polarization_feedback = true;
    launch_once(d, f);
    const double Ez = d.last_Ez();
    EXPECT_NEAR(double(f.span(kAtom_fz)[0]), Ez, 1e-15);
    EXPECT_NEAR(double(f.span(kAtom_fz)[1]), -Ez, 1e-15);
    EXPECT_NEAR(double(f.span(kAtom_fx)[0]), 0.0, 1e-15);
}

TEST(FixDfield, FeedbackEnergyIsElectricEnthalpy) {
    auto f = charge_pair();
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    d.polarization_feedback = true;
    const double U = launch_once(d, f);
    const double Ez = 0.5 - kFourPiK * 2.0e-3;
    const double U_want = 1000.0 * Ez * Ez / (8.0 * units::pi * units::coulomb);
    EXPECT_NEAR(U, U_want, 1e-14);
}

TEST(FixDfield, FeedbackForceIsMinusEnergyGradient) {
    // Central FD of the reported energy vs the applied force on atom 0.
    const double h = 1e-5;
    auto fp = charge_pair();
    auto fm = charge_pair();
    fp.span(kAtom_z)[0] = 6.0 + h;
    fm.span(kAtom_z)[0] = 6.0 - h;
    cpu::md::FixDfield dp(0.0, 0.0, 0.5), dm(0.0, 0.0, 0.5),
        d0(0.0, 0.0, 0.5);
    dp.polarization_feedback = dm.polarization_feedback =
        d0.polarization_feedback = true;
    const double Up = launch_once(dp, fp);
    const double Um = launch_once(dm, fm);
    auto f0 = charge_pair();
    launch_once(d0, f0);
    const double fd = -(Up - Um) / (2.0 * h);
    EXPECT_NEAR(double(f0.span(kAtom_fz)[0]), fd, 1e-8);
}

TEST(FixDfield, FeedbackUsesUnwrappedZ) {
    auto f = charge_pair();
    f.span(kAtom_image)[3 * 0 + 2] = 1; // +q one box up: z^u = 16
    cpu::md::FixDfield d(0.0, 0.0, 0.0);
    d.polarization_feedback = true;
    launch_once(d, f);
    EXPECT_NEAR(d.last_Pz(), (16.0 - 4.0) / 1000.0, 1e-16);
}

TEST(FixDfield, FeedbackThrowsOnZeroVolume) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0.0;
    for (int d = 0; d < 3; ++d)
        f.span(kAtom_image)[d] = 0;
    f.meta.get(kEnergy) = 0.0; // box left default: V = 0
    cpu::md::FixDfield d(0.0, 0.0, 0.5);
    d.polarization_feedback = true;
    cpu::Context ctx;
    EXPECT_THROW(d.launch({ctx, f, 0, 0}), std::runtime_error);
}
