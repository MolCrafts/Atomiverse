/// @file tests/cpu/dynamics/test_const_potential_bo.cpp
/// @brief Unit test: ConstPotentialBO Q polarity + neutrality (vacuum capacitor).

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_potential.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Two parallel walls (2×2 atoms each) in vacuum, no electrolyte.
cpu::Frame make_vacuum_capacitor() {
    constexpr int n_wall = 4; // 2×2 per wall
    const int N = 2 * n_wall;
    cpu::Frame f(N);
    const double Lxy = 6.0;
    const double Lz = 12.0;
    const double zL = 1.0;
    const double zR = 11.0;
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? zL : zR;
        const int typ = (wall == 0) ? 2 : 3;
        for (int ix = 0; ix < 2; ++ix)
            for (int iy = 0; iy < 2; ++iy) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / 2.0);
                f.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / 2.0);
                f.span(kAtom_z)[idx] = z;
                f.span(kAtom_type)[idx] = typ;
                f.span(kAtom_charge)[idx] = 0.0;
                f.span(kAtom_mass)[idx] = 12.0;
                f.span(kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    }
    f.box = cpu::Box::orthorhombic(Lxy, Lxy, Lz);
    return f;
}

} // namespace

TEST(ConstPotentialBO, VacuumCapacitorPolarityAndNeutrality) {
    auto frame = make_vacuum_capacitor();
    cpu::Context runtime{};

    // Ψ_L = −0.5 V, Ψ_R = +0.5 V → Q_L < 0 < Q_R
    cpu::ConstPotentialBO cpot(
        /*types=*/std::vector<int>{2, 3},
        /*psi=*/std::vector<double>{-0.5, +0.5},
        /*w=*/0.5);

    cpu::BuildContext bctx{runtime, frame};
    cpot.build(bctx);

    const double QL = cpot.total_charge(0);
    const double QR = cpot.total_charge(1);
    EXPECT_LT(QL, 0.0) << "left (low Ψ) should be negative";
    EXPECT_GT(QR, 0.0) << "right (high Ψ) should be positive";
    EXPECT_NEAR(QL + QR, 0.0, 1e-8);
    EXPECT_LE(cpot.neutrality_residual(), 1e-8);

    // Charges written to Frame.
    double sum = 0.0;
    for (int i = 0; i < frame.nlocal(); ++i)
        sum += frame.span(kAtom_charge)[i];
    EXPECT_NEAR(sum, 0.0, 1e-8);

    // Re-launch is finite.
    cpu::LaunchContext lctx{runtime, frame, ATV::slot_of(md::Phase::pre_force),
                            1};
    cpot.launch(lctx);
    EXPECT_TRUE(std::isfinite(cpot.total_charge(0)));
    EXPECT_TRUE(std::isfinite(cpot.total_charge(1)));
}
