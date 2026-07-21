/// @file tests/cpu/potentials/test_coul_long_special.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

cpu::Frame two_charge_14() {
    // A-B-C-D; charges on A and D so the only Coulomb pair is 1-4.
    cpu::Frame f(4);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 1.2 * i;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[3] = -1.0;
    f.simbox = cpu::Box::cubic(40.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, 4);
    f.add_bond(0, 1, 0);
    f.add_bond(1, 2, 0);
    f.add_bond(2, 3, 0);
    f.build_special_list();
    return f;
}

cpu::md::PairList only_14() {
    cpu::md::PairList pairs;
    pairs.inum = 4;
    pairs.ilist = {0, 1, 2, 3};
    pairs.numneigh = {1, 0, 0, 0};
    pairs.firstneigh = {0, 1, 1, 1};
    pairs.neighbors = {3};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;
    return pairs;
}

double run_coul(cpu::Frame& f, cpu::md::PairList& pairs, double s14) {
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::CoulLong coul(/*alpha=*/0.3, /*cutoff=*/20.0);
    coul.set_special_bonds(0.0, 0.0, s14);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});
    return f.meta.get(kEnergy);
}

} // namespace

TEST(CoulLongSpecial, Half14RealSpace) {
    auto f = two_charge_14();
    auto pairs = only_14();
    const double e1 = run_coul(f, pairs, 1.0);
    pairs.version = 2;
    const double e05 = run_coul(f, pairs, 0.5);
    ASSERT_NE(e1, 0.0);
    EXPECT_NEAR(e05, 0.5 * e1, 1e-10 * std::abs(e1));
}

TEST(CoulLongSpecial, Zero12Bonded) {
    // Bonded 1-2 pair only.
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 1.5;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 0;
    f.span(kAtom_tag)[1] = 1;
    f.simbox = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, 2);
    f.add_bond(0, 1, 0);
    f.build_special_list();

    cpu::md::PairList pairs;
    pairs.inum = 2;
    pairs.ilist = {0, 1};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 1};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;

    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::CoulLong coul(0.3, 10.0);
    coul.set_special_bonds(0.0, 0.0, 0.5);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-14);
}
