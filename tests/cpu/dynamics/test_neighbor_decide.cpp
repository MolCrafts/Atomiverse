/// @file tests/cpu/dynamics/test_neighbor_decide.cpp
/// @brief `NeighborList::decide_reneighbor` is LAMMPS `Neighbor::decide`.

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"

using namespace ATV;

namespace {

cpu::Frame two_atoms() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 1.0;
    f.span(kAtom_z)[0] = 1.0;
    f.span(kAtom_x)[1] = 1.5;
    f.span(kAtom_y)[1] = 1.0;
    f.span(kAtom_z)[1] = 1.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_mass)[1] = 1.0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

} // namespace

TEST(TestNeighborDecide, DelayTenEveryOneCheckNoRebuildsOnTenthDecide) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(/*cutoff=*/2.0, /*skin=*/0.4, /*every=*/1,
                               /*delay=*/10, /*check=*/false);
    nl.build({ctx, frame});
    ASSERT_EQ(nl.n_rebuilds(), 1);

    for (int s = 1; s <= 9; ++s) {
        const bool nflag = nl.decide_reneighbor(ctx, frame);
        EXPECT_FALSE(nflag) << "step " << s;
        ctx.md.reneighbor = nflag;
        ctx.md.reneighbor_decided = true;
        nl.launch({ctx, frame, 0, s});
        EXPECT_EQ(nl.n_rebuilds(), 1) << "step " << s;
    }
    const bool tenth = nl.decide_reneighbor(ctx, frame);
    EXPECT_TRUE(tenth);
    ctx.md.reneighbor = tenth;
    ctx.md.reneighbor_decided = true;
    nl.launch({ctx, frame, 0, 10});
    EXPECT_EQ(nl.n_rebuilds(), 2);
}

TEST(TestNeighborDecide, DirectLaunchDecidesEachCall) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 10, false);
    nl.build({ctx, frame});
    const int before = nl.n_rebuilds();
    for (int s = 0; s < 9; ++s)
        nl.launch({ctx, frame, 0, s});
    EXPECT_EQ(nl.n_rebuilds(), before);
    nl.launch({ctx, frame, 0, 9});
    EXPECT_EQ(nl.n_rebuilds(), before + 1);
}
