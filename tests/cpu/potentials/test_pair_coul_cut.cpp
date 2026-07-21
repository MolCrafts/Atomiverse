/// @file tests/cpu/potentials/test_pair_coul_cut.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

TEST(PairCoulCut, TwoChargeEnergyAndForce) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 2.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.simbox = cpu::Box::cubic(20.0);

    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    f.meta.get(kEnergy) = 0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;

    cpu::PairCoulCut coul(10.0);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});

    const double r = 2.0;
    const double e_ref = units::coulomb * (1.0) * (-1.0) / r;
    EXPECT_NEAR(f.meta.get(kEnergy), e_ref, 1e-12);
    // Attractive: atom 0 pulled toward +x (positive force).
    EXPECT_GT(f.span(kAtom_fx)[0], 0.0);
    EXPECT_LT(f.span(kAtom_fx)[1], 0.0);
}

TEST(PairCoulCut, SpecialBondsExclude12) {
    // Water-like O-H pair: charges + topology, full 1-2 exclusion → E=0.
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 1.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = -0.834;
    f.span(kAtom_charge)[1] = 0.417;
    f.span(kAtom_mass)[0] = 16.0;
    f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.simbox = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(1, 2, 0); // tags, not local indices
    f.build_special_list();

    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    f.meta.get(kEnergy) = 0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;

    cpu::PairCoulCut coul(10.0);
    coul.set_special_bonds(0.0, 0.0, 0.0);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});

    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-14);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-14);
    EXPECT_NEAR(f.span(kAtom_fx)[1], 0.0, 1e-14);
}

TEST(PairCoulCut, SpecialBondsKeepIntermolecular) {
    // Two O-like charges, no bonds → full Coulomb retained under specials.
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = -0.834;
    f.span(kAtom_charge)[1] = -0.834;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 16.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.simbox = cpu::Box::cubic(20.0);
    f.build_special_list();

    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
    pairs.has_shifts = false;
    pairs.version = 1;

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    f.meta.get(kEnergy) = 0;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;

    cpu::PairCoulCut coul(10.0);
    coul.set_special_bonds(0.0, 0.0, 0.0);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});

    const double r = 3.0;
    const double e_ref = units::coulomb * (-0.834) * (-0.834) / r;
    EXPECT_NEAR(f.meta.get(kEnergy), e_ref, 1e-12);
}
