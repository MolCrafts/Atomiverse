/// @file tests/cpu/potentials/test_dihedral_charmm.cpp
/// @brief Unit tests for CHARMM periodic dihedral energy on fixed geometry.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"

using namespace ATV;

namespace {

/// Orthogonal-bond geometry: φ = π/2 exactly.
/// r1=(0,0,0), r2=(1,0,0), r3=(1,1,0), r4=(1,1,1).
cpu::Frame make_right_angle_dihedral() {
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
    frame.add_dihedral(0, 1, 2, 3, 0);
    return frame;
}

double eval_energy(cpu::Frame frame, cpu::DihedralCharmm& pot) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    frame.meta.get(ATV::kEnergyDihedral) = 0;
    cpu::Context ctx;
    pot.build({ctx, frame});
    pot.launch({ctx, frame, 0, 0});
    pot.destroy({ctx, frame});
    return frame.meta.get(ATV::kEnergy);
}

} // namespace

TEST(DihedralCharmm, RightAngleEnergy) {
    // φ = π/2. V = K (1 + cos(n φ − δ)).
    // n=1, δ=0 → V = K (1 + cos(π/2)) = K
    // n=2, δ=0 → V = K (1 + cos(π)) = 0
    // n=1, δ=π → V = K (1 + cos(π/2 − π)) = K (1 + cos(−π/2)) = K
    constexpr double K = 0.25;
    auto f1 = make_right_angle_dihedral();
    cpu::DihedralCharmm p1({K}, {1}, {0.0});
    EXPECT_NEAR(eval_energy(f1, p1), K, 1e-12);

    auto f2 = make_right_angle_dihedral();
    cpu::DihedralCharmm p2({K}, {2}, {0.0});
    EXPECT_NEAR(eval_energy(f2, p2), 0.0, 1e-12);

    auto f3 = make_right_angle_dihedral();
    cpu::DihedralCharmm p3({K}, {1}, {M_PI});
    EXPECT_NEAR(eval_energy(f3, p3), K, 1e-12);
}

TEST(DihedralCharmm, RejectsBadParams) {
    EXPECT_THROW(cpu::DihedralCharmm({}, {1}, {0.0}), std::invalid_argument);
    EXPECT_THROW(cpu::DihedralCharmm({1.0}, {0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(cpu::DihedralCharmm({1.0}, {1, 2}, {0.0}),
                 std::invalid_argument);
}

TEST(DihedralCharmm, TranslationalForceBalance) {
    auto frame = make_right_angle_dihedral();
    // Displace atom 4 slightly so forces are non-zero.
    frame.span(ATV::kAtom_x)[3] = 1.2;
    frame.span(ATV::kAtom_y)[3] = 0.8;
    frame.span(ATV::kAtom_z)[3] = 0.9;

    cpu::DihedralCharmm pot({0.3}, {3}, {0.4});
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
