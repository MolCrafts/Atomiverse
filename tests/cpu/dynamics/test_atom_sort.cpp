/// @file tests/cpu/md/test_atom_sort.cpp
/// @brief Spatial sort reorders owned atoms by bin; tags/positions stay bound.
#include <gtest/gtest.h>

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"

using namespace ATV;

TEST(AtomSort, ReordersBySpatialBin) {
    constexpr int n = 8;
    cpu::Frame f(n);
    // Deliberately scrambled order: far atoms interleaved.
    const double coords[n][3] = {
        {9.0, 0.1, 0.1}, {0.1, 0.1, 0.1}, {9.0, 9.0, 0.1}, {0.1, 9.0, 0.1},
        {0.1, 0.1, 9.0}, {9.0, 0.1, 9.0}, {0.1, 9.0, 9.0}, {9.0, 9.0, 9.0},
    };
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = coords[i][0];
        f.span(kAtom_y)[i] = coords[i][1];
        f.span(kAtom_z)[i] = coords[i][2];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = i;
        f.span(kAtom_mass)[i] = 1.0 + 0.01 * i;
    }
    f.box = cpu::Box::cubic(10.0);

    // Snapshot tag→position
    std::vector<double> xref(n), yref(n), zref(n);
    std::vector<int> typeref(n);
    for (int i = 0; i < n; ++i) {
        const int tag = static_cast<int>(f.span(kAtom_tag)[i]);
        xref[tag - 1] = f.span(kAtom_x)[i];
        yref[tag - 1] = f.span(kAtom_y)[i];
        zref[tag - 1] = f.span(kAtom_z)[i];
        typeref[tag - 1] = f.span(kAtom_type)[i];
    }

    const bool moved = cpu::md::sort_owned_atoms(f, n, /*binsize=*/5.0);
    EXPECT_TRUE(moved);

    // Every tag still carries its original coordinates / type.
    for (int i = 0; i < n; ++i) {
        const int tag = static_cast<int>(f.span(kAtom_tag)[i]);
        EXPECT_NEAR(f.span(kAtom_x)[i], xref[tag - 1], 1e-15);
        EXPECT_NEAR(f.span(kAtom_y)[i], yref[tag - 1], 1e-15);
        EXPECT_NEAR(f.span(kAtom_z)[i], zref[tag - 1], 1e-15);
        EXPECT_EQ(f.span(kAtom_type)[i], typeref[tag - 1]);
    }

    // Idempotent on already-sorted data (same binsize).
    EXPECT_FALSE(cpu::md::sort_owned_atoms(f, n, 5.0));
}

TEST(AtomSort, NotifyBumpsNeighborGeneration) {
    cpu::Frame f(4);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.1 * i;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 0.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = 0;
    }
    f.box = cpu::Box::cubic(4.0);
    cpu::md::mpi::DomainDecomposition dd;
    dd.bind(f);
    const auto g0 = dd.neighbor_generation;
    cpu::md::notify_sorted(&dd, f, /*swaps=*/nullptr);
    EXPECT_EQ(dd.neighbor_generation, g0 + 1);
    // Owner tag map always rebuilt after sort (specials / gather / bonded).
    EXPECT_EQ(dd.tag_to_local.size(), 4u);
}

/// The FRAME's own tag index — not just the decomposition's mirror — must name
/// the post-sort rows. On a single rank that index is the ONLY tag→row map the
/// constraint solvers have (`FixRattle`, `FixSettle`), so a stale entry means
/// they hold three unrelated atoms rigid and the molecules quietly come apart.
TEST(AtomSort, NotifyRepublishesTheFrameTagIndex) {
    constexpr int n = 8;
    cpu::Frame f(n);
    const double coords[n][3] = {
        {9.0, 0.1, 0.1}, {0.1, 0.1, 0.1}, {9.0, 9.0, 0.1}, {0.1, 9.0, 0.1},
        {0.1, 0.1, 9.0}, {9.0, 0.1, 9.0}, {0.1, 9.0, 9.0}, {9.0, 9.0, 9.0},
    };
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = coords[i][0];
        f.span(kAtom_y)[i] = coords[i][1];
        f.span(kAtom_z)[i] = coords[i][2];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = i;
        f.span(kAtom_mass)[i] = 1.0;
    }
    f.box = cpu::Box::cubic(10.0);
    f.rebuild_tag_index(0, n);

    cpu::md::mpi::DomainDecomposition dd;
    dd.bind(f);
    ASSERT_TRUE(cpu::md::sort_owned_atoms(f, n, /*binsize=*/5.0));
    cpu::md::notify_sorted(&dd, f, /*swaps=*/nullptr);

    const auto& index = f.meta.get(cpu::kTagIndex);
    ASSERT_EQ(index.size(), static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row)
        EXPECT_EQ(index.at(f.span(kAtom_tag)[row]), row);
}

/// …and it must do so **without** a decomposition. `notify_sorted` used to
/// early-return on `dd == nullptr` before touching the Frame index, so a
/// pipeline with sorting on and no DD would have kept a map naming the
/// pre-sort rows: no error, no NaN, every tag silently pointing at the wrong
/// atom. Unreachable today (`DomainRuntime::allocate` always makes a `dd`) —
/// which is precisely why it needs a gate rather than a comment.
TEST(AtomSort, NotifyRepublishesTheFrameTagIndexWithoutADecomposition) {
    constexpr int n = 8;
    cpu::Frame f(n);
    const double coords[n][3] = {
        {9.0, 0.1, 0.1}, {0.1, 0.1, 0.1}, {9.0, 9.0, 0.1}, {0.1, 9.0, 0.1},
        {0.1, 0.1, 9.0}, {9.0, 0.1, 9.0}, {0.1, 9.0, 9.0}, {9.0, 9.0, 9.0},
    };
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = coords[i][0];
        f.span(kAtom_y)[i] = coords[i][1];
        f.span(kAtom_z)[i] = coords[i][2];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = i;
        f.span(kAtom_mass)[i] = 1.0;
    }
    f.box = cpu::Box::cubic(10.0);
    f.rebuild_tag_index(0, n);

    ASSERT_TRUE(cpu::md::sort_owned_atoms(f, n, /*binsize=*/5.0));
    cpu::md::notify_sorted(/*dd=*/nullptr, f, /*swaps=*/nullptr);

    const auto& index = f.meta.get(cpu::kTagIndex);
    ASSERT_EQ(index.size(), static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row)
        EXPECT_EQ(index.at(f.span(kAtom_tag)[row]), row);
}

/// The clock is the Driver step, assigned not incremented. A Minimize
/// line-search re-evaluation stamps the same step twice and must not walk
/// `next_sort` — that used to happen because `forward_comm` did
/// `++dd->atom_sort_step` on every force evaluation.
TEST(AtomSort, LineSearchReevalDoesNotAdvanceCadence) {
    cpu::md::AtomSortState s;
    s.configure(/*nfreq=*/2);
    EXPECT_EQ(s.next_sort, 2);

    s.step = 2;
    EXPECT_TRUE(s.due());
    s.step = 2; // same step again — line-search trial
    EXPECT_TRUE(s.due());
    EXPECT_EQ(s.next_sort, 2);
    s.mark_sorted();
    EXPECT_EQ(s.next_sort, 4);
    s.step = 2;
    EXPECT_FALSE(s.due());
}

/// Under plain MD the sort fires on the requested nfreq cadence, not on
/// the first step (LAMMPS `nextsort` init) and not on the in-between steps.
TEST(AtomSort, FiresOnNfreqCadence) {
    cpu::md::AtomSortState s;
    s.configure(/*nfreq=*/3);
    s.step = 0;
    EXPECT_FALSE(s.due());
    s.step = 2;
    EXPECT_FALSE(s.due());
    s.step = 3;
    EXPECT_TRUE(s.due());
    s.mark_sorted();
    s.step = 5;
    EXPECT_FALSE(s.due());
    s.step = 6;
    EXPECT_TRUE(s.due());
}

/// NeighborList publishes the existing AtomSortState; it does not flatten
/// the cadence onto DomainDecomposition.
TEST(AtomSort, NeighborListPublishesTheStateOnDynamicsServices) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_mass)[1] = 1.0;
    f.box = cpu::Box::cubic(10.0);

    cpu::Context ctx;
    cpu::md::NeighborList<> nl(/*cutoff=*/2.0);
    nl.set_atom_sort(/*nfreq=*/7, /*binsize=*/1.5, /*enabled=*/true);
    nl.build({ctx, f});
    ASSERT_NE(ctx.md.atom_sort, nullptr);
    EXPECT_EQ(ctx.md.atom_sort->nfreq, 7);
    EXPECT_DOUBLE_EQ(ctx.md.atom_sort->binsize, 1.5);
    EXPECT_TRUE(ctx.md.atom_sort->enabled);
}
