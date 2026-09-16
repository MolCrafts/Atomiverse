/// @file tests/cpu/potentials/test_bond_fene.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bond_fene.h"

using namespace ATV;

namespace {

cpu::Frame dimer_with_bond(double r) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 0;
    f.span(kAtom_tag)[1] = 1;
    f.box = cpu::Box::cubic(50.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(0, 1, 0);
    return f;
}

} // namespace

TEST(BondFene, AnalyticEnergy) {
    const double K = 30.0, R0 = 1.5, r = 1.0;
    const double U = -0.5 * K * R0 * R0 * std::log(1.0 - (r / R0) * (r / R0));

    auto f = dimer_with_bond(r);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyBond) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::BondFene bond({K}, {R0});
    bond.build({ctx, f});
    bond.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    // Attractive: atom 0 force +x
    EXPECT_GT(f.span(kAtom_fx)[0], 0.0);
    EXPECT_LT(f.span(kAtom_fx)[1], 0.0);
    bond.destroy({ctx, f});
}

TEST(BondFene, ThrowsAtOrBeyondR0) {
    auto f = dimer_with_bond(1.5);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    cpu::BondFene bond({30.0}, {1.5});
    bond.build({ctx, f});
    EXPECT_THROW(bond.launch({ctx, f, 0, 0}), std::runtime_error);
    bond.destroy({ctx, f});
}
