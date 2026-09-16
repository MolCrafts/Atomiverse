/// @file tests/cpu/potentials/test_pair_yukawa.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"

using namespace ATV;

namespace {

cpu::Frame dimer(double r) {
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
    return f;
}

} // namespace

TEST(PairYukawa, AnalyticEnergyAndForce) {
    const double A = 2.0, kappa = 0.5, rc = 10.0, r = 3.0;
    const double U_un = A * std::exp(-kappa * r) / r;
    const double U_rc = A * std::exp(-kappa * rc) / rc;
    const double U = U_un - U_rc;
    const double dUdr = -A * std::exp(-kappa * r) * (kappa / r + 1.0 / (r * r));
    // Force on atom 0 along +x toward 1 when A>0 is repulsive → F0x < 0
    const double F0x = -(-dUdr); // magnitude dU/dr < 0 so -dUdr > 0 ...
    // U decreases with r for A>0? dUdr = -A e (...)<0 so U decreases → attractive?
    // Wait: A>0 Yukawa is typically repulsive: U = +A e/r > 0, dU/dr < 0 means
    // energy decreases as r increases → repulsive force pushing particles apart.
    // F on particle 0: points away from 1 → −x → F0x < 0.
    // F = −∇U; ∇_0 U = dU/dr * ∂r/∂x0; r = x1−x0, ∂r/∂x0 = −1 → ∇_0 U = −dU/dr
    // F0 = −∇_0 U = dU/dr < 0. Good.

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;

    cpu::Context ctx;
    cpu::PairYukawa yuk(A, kappa, rc);
    yuk.build({ctx, f});
    yuk.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], dUdr, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -dUdr, 1e-12);
    (void)F0x;
    yuk.destroy({ctx, f});
}
