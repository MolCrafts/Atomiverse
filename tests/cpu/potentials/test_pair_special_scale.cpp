/// @file tests/cpu/potentials/test_pair_special_scale.cpp
/// @brief Amber-style special_bonds scales for PairLJCut and PairCoulCut.
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Four atoms: A-B-C-D chain so A–D is 1-4 via build_special_list.
cpu::Frame four_atom_chain() {
    cpu::Frame f(4);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 1.5 * i;
        f.span(kAtom_y)[i] = 0;
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i; // dense tags 0..3
        f.span(kAtom_charge)[i] = (i == 0) ? 1.0 : (i == 3) ? -1.0 : 0.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(40.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, 4);
    f.add_bond(0, 1, 0);
    f.add_bond(1, 2, 0);
    f.add_bond(2, 3, 0);
    f.build_special_list();
    return f;
}

/// Pair list containing only the 1-4 pair (atoms 0 and 3).
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

void zero_forces_energy(cpu::Frame& f) {
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
}

} // namespace

TEST(PairSpecialScale, CoulCutHalf14) {
    auto f = four_atom_chain();
    auto pairs = only_14();
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    zero_forces_energy(f);
    cpu::PairCoulCut full(20.0);
    full.set_special_bonds(1.0, 1.0, 1.0);
    full.build({ctx, f});
    full.launch({ctx, f, 0, 0});
    full.destroy({ctx, f});
    const double e_full = f.meta.get(kEnergy);
    const double fx0_full = f.span(kAtom_fx)[0];
    ASSERT_NE(e_full, 0.0);

    zero_forces_energy(f);
    pairs.version = 2; // force rebuild
    cpu::PairCoulCut half(20.0);
    half.set_special_bonds(0.0, 0.0, 0.5);
    half.build({ctx, f});
    half.launch({ctx, f, 0, 0});
    half.destroy({ctx, f});
    const double e_half = f.meta.get(kEnergy);
    const double fx0_half = f.span(kAtom_fx)[0];

    EXPECT_NEAR(e_half, 0.5 * e_full, 1e-12 * std::abs(e_full));
    EXPECT_NEAR(fx0_half, 0.5 * fx0_full,
                1e-10 * std::max(1.0, std::abs(fx0_full)));
}

TEST(PairSpecialScale, CoulCutZero14) {
    auto f = four_atom_chain();
    auto pairs = only_14();
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    zero_forces_energy(f);

    cpu::PairCoulCut coul(20.0);
    coul.set_special_bonds(0.0, 0.0, 0.0);
    coul.build({ctx, f});
    coul.launch({ctx, f, 0, 0});
    coul.destroy({ctx, f});

    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fx)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fx)[3], 0.0, 1e-15);
}

TEST(PairSpecialScale, LJCutHalf14) {
    auto f = four_atom_chain();
    // Give all atoms LJ sites (type 0).
    auto pairs = only_14();
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    const double eps = 0.01, sig = 3.0, cut = 20.0;

    zero_forces_energy(f);
    cpu::PairLJCut full(eps, sig, cut);
    full.set_special_bonds(1.0, 1.0, 1.0);
    full.build({ctx, f});
    full.launch({ctx, f, 0, 0});
    full.destroy({ctx, f});
    const double e_full = f.meta.get(kEnergy);
    const double fx0_full = f.span(kAtom_fx)[0];
    ASSERT_NE(e_full, 0.0);

    zero_forces_energy(f);
    pairs.version = 2;
    cpu::PairLJCut half(eps, sig, cut);
    half.set_special_bonds(0.0, 0.0, 0.5);
    half.build({ctx, f});
    half.launch({ctx, f, 0, 0});
    half.destroy({ctx, f});
    const double e_half = f.meta.get(kEnergy);
    const double fx0_half = f.span(kAtom_fx)[0];

    EXPECT_NEAR(e_half, 0.5 * e_full, 1e-12 * std::abs(e_full));
    EXPECT_NEAR(fx0_half, 0.5 * fx0_full,
                1e-10 * std::max(1.0, std::abs(fx0_full)));
}
