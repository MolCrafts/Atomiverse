/// @file tests/cpu/potentials/test_improper_harmonic.cpp
/// @brief Unit tests for harmonic improper energy and force balance.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"

using namespace ATV;

namespace {

/// Orthogonal bonds → χ = π/2.
cpu::Frame make_right_angle_improper() {
    cpu::Frame frame(4);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 1.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.0;
    frame.span(ATV::kAtom_x)[2] = 1.0;
    frame.span(ATV::kAtom_y)[2] = 1.0;
    frame.span(ATV::kAtom_z)[2] = 0.0;
    frame.span(ATV::kAtom_x)[3] = 1.0;
    frame.span(ATV::kAtom_y)[3] = 1.0;
    frame.span(ATV::kAtom_z)[3] = 1.0;
    for (int i = 0; i < 4; ++i) {
        frame.span(ATV::kAtom_mass)[i] = 12.0;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.box = cpu::Box::cubic(40.0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_improper(0, 1, 2, 3, 0);
    return frame;
}

double eval_energy(cpu::Frame& frame, cpu::ImproperHarmonic& pot) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    frame.meta.get(ATV::kEnergyImproper) = 0;
    cpu::Context ctx;
    pot.build({ctx, frame});
    pot.launch({ctx, frame, 0, 0});
    pot.destroy({ctx, frame});
    return frame.meta.get(ATV::kEnergy);
}

} // namespace

TEST(ImproperHarmonic, RightAngleEnergy) {
    // χ = π/2. V = K (χ − χ0)²
    // χ0 = π/2 → V = 0
    // χ0 = 0 → V = K (π/2)²
    constexpr double K = 0.4;
    auto f0 = make_right_angle_improper();
    cpu::ImproperHarmonic p0({K}, {M_PI / 2.0});
    EXPECT_NEAR(eval_energy(f0, p0), 0.0, 1e-12);

    auto f1 = make_right_angle_improper();
    cpu::ImproperHarmonic p1({K}, {0.0});
    const double want = K * (M_PI / 2.0) * (M_PI / 2.0);
    EXPECT_NEAR(eval_energy(f1, p1), want, 1e-12);
    EXPECT_NEAR(f1.meta.get(ATV::kEnergyImproper), want, 1e-12);
}

TEST(ImproperHarmonic, RejectsBadParams) {
    EXPECT_THROW(cpu::ImproperHarmonic({}, {0.0}), std::invalid_argument);
    EXPECT_THROW(cpu::ImproperHarmonic({1.0}, {0.0, 1.0}),
                 std::invalid_argument);
}

TEST(ImproperHarmonic, TranslationalForceBalance) {
    auto frame = make_right_angle_improper();
    frame.span(ATV::kAtom_x)[3] = 1.15;
    frame.span(ATV::kAtom_y)[3] = 0.85;
    frame.span(ATV::kAtom_z)[3] = 0.95;

    cpu::ImproperHarmonic pot({0.5}, {0.0});
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    cpu::Context ctx;
    pot.build({ctx, frame});
    pot.launch({ctx, frame, 0, 0});
    pot.destroy({ctx, frame});

    double fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        fx += frame.span(ATV::kAtom_fx)[i];
        fy += frame.span(ATV::kAtom_fy)[i];
        fz += frame.span(ATV::kAtom_fz)[i];
    }
    EXPECT_NEAR(fx, 0.0, 1e-10);
    EXPECT_NEAR(fy, 0.0, 1e-10);
    EXPECT_NEAR(fz, 0.0, 1e-10);
    EXPECT_GT(std::abs(frame.meta.get(ATV::kEnergy)), 1e-12);
}

TEST(ImproperHarmonic, FrameTopologyIndependentOfDihedral) {
    auto frame = make_right_angle_improper();
    frame.add_dihedral(0, 1, 2, 3, 0);
    EXPECT_EQ(frame.span(kAtom_num_improper)[1], 1);
    EXPECT_EQ(frame.span(kAtom_num_dihedral)[1], 1);
}
