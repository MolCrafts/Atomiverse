/// @file tests/cpu/potentials/test_pair_coul_cut.cpp
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

TEST(PairCoulCut, FilteredListUsesExplicitOwnerIndices) {
    for (bool filtered : {false, true}) {
        cpu::Frame frame(3);
        frame.box = cpu::Box::cubic(20.0);
        frame.span(kAtom_x)[0] = 7.0;
        frame.span(kAtom_x)[1] = 2.0;
        frame.span(kAtom_x)[2] = 0.0;
        for (int i = 0; i < 3; ++i) {
            frame.span(kAtom_charge)[i] = 1.0;
            frame.span(kAtom_tag)[i] = i + 1;
        }
        cpu::md::PairList pairs;
        pairs.inum = 1;
        pairs.ilist = {2};
        pairs.numneigh = {0, 0, 1};
        pairs.firstneigh = {0, 0, 0};
        pairs.neighbors = {1};
        pairs.total_pairs = 1;
        cpu::Context context;
        context.md.pair_list = &pairs;
        cpu::PairCoulCut coul(4.0);
        if (filtered)
            coul.set_special_bonds(0.0, 0.0, 0.5);
        coul.build({context, frame});
        coul.launch({context, frame, 0, 0, true, true});
        EXPECT_NEAR(frame.meta.get(kEnergy),
                    units::UnitSystem::lj().qqrd2e() / 2.0, 1e-12);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 0.0);
        EXPECT_NEAR(frame.span(kAtom_fx)[2],
                    -units::UnitSystem::lj().qqrd2e() / 4.0, 1e-12);
        coul.destroy({context, frame});
    }
}

TEST(PairCoulCut, FilteredRowsRetainExplicitImageCoordinates) {
    for (bool filtered : {false, true}) {
        for (bool inside : {false, true}) {
            cpu::Frame frame(2);
            frame.box = cpu::Box::orthorhombic(7.6, 20.0, 20.0);
            frame.span(kAtom_x)[0] = 0.2;
            frame.span(kAtom_x)[1] = inside ? 3.8 : -3.8;
            frame.span(kAtom_charge)[0] = 1.0;
            frame.span(kAtom_charge)[1] = -1.0;
            cpu::md::PairList pairs;
            pairs.inum = 1;
            pairs.ilist = {0};
            pairs.numneigh = {1, 0};
            pairs.firstneigh = {0, 1};
            pairs.neighbors = {1};
            pairs.total_pairs = 1;
            cpu::Context context;
            context.md.pair_list = &pairs;
            cpu::PairCoulCut coul(3.77118);
            if (filtered)
                coul.set_special_bonds(0.0, 0.0, 0.5);
            coul.build({context, frame});
            coul.launch({context, frame, 0, 0, true, true});
            EXPECT_NEAR(frame.meta.get(kEnergy),
                        inside ? -units::UnitSystem::lj().qqrd2e() / 3.6 : 0.0,
                        1e-12);
            if (!inside) {
                EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 0.0);
                EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[1], 0.0);
                EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[0], 0.0);
            }
            coul.destroy({context, frame});
        }
    }
}

TEST(PairCoulCut, FilteredCacheTracksSourceAndBuildLifecycle) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(20.0);
    frame.span(kAtom_x)[1] = 2.0;
    frame.span(kAtom_charge)[0] = 1.0;
    frame.span(kAtom_charge)[1] = -1.0;
    frame.span(kAtom_tag)[0] = 1;
    frame.span(kAtom_tag)[1] = 2;
    cpu::md::PairList populated;
    populated.inum = 2;
    populated.ilist = {0, 1};
    populated.numneigh = {1, 0};
    populated.firstneigh = {0, 1};
    populated.neighbors = {1};
    populated.total_pairs = 1;
    auto empty = populated;
    empty.numneigh = {0, 0};
    empty.firstneigh = {0, 0};
    empty.neighbors.clear();
    empty.total_pairs = 0;
    ASSERT_EQ(populated.version, 0u);
    ASSERT_EQ(empty.version, populated.version);
    cpu::Context context;
    context.md.pair_list = &populated;
    cpu::PairCoulCut coul(8.0);
    coul.set_special_bonds(0.0, 0.0, 0.5);
    coul.build({context, frame});
    auto evaluate = [&](cpu::md::PairList& source) {
        context.md.pair_list = &source;
        frame.meta.get(kEnergy) = 0;
        coul.launch({context, frame, 0, 0, false, false});
        return frame.meta.get(kEnergy);
    };
    const double expected = -units::UnitSystem::lj().qqrd2e() / 2.0;
    EXPECT_NEAR(evaluate(populated), expected, 1e-12);
    EXPECT_DOUBLE_EQ(evaluate(empty), 0.0);
    EXPECT_NEAR(evaluate(populated), expected, 1e-12);

    // A new build invalidates the old cache even when its provider and
    // version are unchanged but the provider has been reinitialized.
    populated = empty;
    coul.build({context, frame});
    EXPECT_DOUBLE_EQ(evaluate(populated), 0.0);
    coul.destroy({context, frame});
    populated.numneigh = {1, 0};
    populated.firstneigh = {0, 1};
    populated.neighbors = {1};
    populated.total_pairs = 1;
    coul.build({context, frame});
    EXPECT_NEAR(evaluate(populated), expected, 1e-12);
    coul.destroy({context, frame});
}

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
    f.box = cpu::Box::cubic(20.0);

    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
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
    const double e_ref = units::UnitSystem::lj().qqrd2e() * (1.0) * (-1.0) / r;
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
    f.box = cpu::Box::cubic(20.0);
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
    f.box = cpu::Box::cubic(20.0);
    f.build_special_list();

    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh = {1, 0};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {1};
    pairs.total_pairs = 1;
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
    const double e_ref =
        units::UnitSystem::lj().qqrd2e() * (-0.834) * (-0.834) / r;
    EXPECT_NEAR(f.meta.get(kEnergy), e_ref, 1e-12);
}
