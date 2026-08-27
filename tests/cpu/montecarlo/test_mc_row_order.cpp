/// @file tests/cpu/montecarlo/test_mc_row_order.cpp
/// @brief An MC plan must not renumber frame rows.
///
/// `mc::Transaction` records a trial's changeset as a set of frame ROWS
/// ("row `r` looked like this"), and the neighbour rebuild that would
/// spatially reorder those rows runs INSIDE the trial — the list is bound at
/// `sampling::Phase::energy`, ahead of the evaluator that scores the proposal.
/// A reorder landing between `record_touch` and `rollback` therefore splices a
/// saved row over an unrelated atom: two atoms end up on top of each other,
/// the sampler's reference energy stops describing the configuration it holds,
/// and every later trial is rejected against an overlap it cannot see. The
/// chain freezes with no error and no NaN.
///
/// So `McBase::install_defaults()` calls `forbid_row_reordering()` on the
/// neighbour provider it was handed. These tests pin both halves: the
/// provider honours the request, and the MC preset makes it.

#include <gtest/gtest.h>

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Spatially SCRAMBLED lattice: consecutive rows are far apart, so a spatial
/// sort has something to do and will visibly permute the tag column.
cpu::Frame scrambled_box(int side, double spacing) {
    const int n = side * side * side;
    cpu::Frame f(n);
    int row = 0;
    for (int k = 0; k < side; ++k)
        for (int j = 0; j < side; ++j)
            for (int i = 0; i < side; ++i) {
                // Interleave the two halves of the z stack.
                const int kk = (k % 2 == 0) ? k / 2 : side - 1 - k / 2;
                f.span(kAtom_x)[row] = (i + 0.5) * spacing;
                f.span(kAtom_y)[row] = (j + 0.5) * spacing;
                f.span(kAtom_z)[row] = (kk + 0.5) * spacing;
                f.span(kAtom_mass)[row] = 39.948;
                f.span(kAtom_type)[row] = 1;
                f.span(kAtom_tag)[row] = row + 1;
                ++row;
            }
    f.box = cpu::Box::cubic(side * spacing);
    f.rebuild_tag_index(0, n);
    return f;
}

std::vector<TagInt> tag_column(const cpu::Frame& f) {
    std::vector<TagInt> t(static_cast<std::size_t>(f.nlocal()));
    for (int i = 0; i < f.nlocal(); ++i)
        t[static_cast<std::size_t>(i)] = f.span(kAtom_tag)[i];
    return t;
}

} // namespace

/// The provider honours the request: after `forbid_row_reordering()` the sort
/// never fires, however often the cadence comes round.
TEST(NeighborListRowOrder, ForbidRowReorderingStopsTheSpatialSort) {
    auto f = scrambled_box(4, 4.0);
    const auto before = tag_column(f);

    // `check no`, delay 0: rebuild on every launch, so the sort cadence is
    // reached immediately. Sort at EVERY rebuild — without the veto the very
    // first launch permutes (see the control below).
    cpu::md::NeighborList<> nl{6.0, /*skin=*/1.0, /*every=*/1, /*delay=*/0,
                               /*check=*/false};
    nl.set_atom_sort(/*nfreq=*/1);
    nl.forbid_row_reordering();

    cpu::Context ctx;
    nl.build({ctx, f});
    for (int step = 0; step < 5; ++step)
        nl.launch({ctx, f, ATV::slot_of(ATV::sampling::Phase::energy), step});

    EXPECT_EQ(tag_column(f), before);
}

/// Control: the same list WITHOUT the veto does reorder, so the test above is
/// not passing because the sort was never going to run.
TEST(NeighborListRowOrder, TheSortDoesReorderWhenItIsAllowed) {
    auto f = scrambled_box(4, 4.0);
    const auto before = tag_column(f);

    cpu::md::NeighborList<> nl{6.0, /*skin=*/1.0, /*every=*/1, /*delay=*/0,
                               /*check=*/false};
    nl.set_atom_sort(/*nfreq=*/1);

    cpu::Context ctx;
    nl.build({ctx, f});
    for (int step = 0; step < 5; ++step)
        nl.launch({ctx, f, ATV::slot_of(ATV::md::Phase::neighbor), step});

    EXPECT_NE(tag_column(f), before);
}

/// And the MC preset makes the request, at `build()`, on the list it was
/// handed — the user does not have to know.
TEST(McRowOrder, BuildForbidsRowReorderingOnTheNeighborList) {
    auto f = scrambled_box(4, 4.0);

    cpu::md::NeighborList<> nl{6.0, /*skin=*/1.0, /*every=*/1, /*delay=*/0,
                               /*check=*/false};
    nl.set_atom_sort(/*nfreq=*/1);

    cpu::mc::MC mc{/*T [K]=*/300.0, /*seed=*/7u};
    mc.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.0});
    mc.set_neighbor_ref(nl);
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    mc.build();

    const auto before = tag_column(f);
    cpu::Context ctx;
    nl.launch({ctx, f, ATV::slot_of(ATV::sampling::Phase::energy), 0});
    EXPECT_EQ(tag_column(f), before);
}
