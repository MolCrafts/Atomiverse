/// @file tests/cpu/md/test_atom_sort.cpp
/// @brief Spatial sort reorders owned atoms by bin; tags/positions stay bound.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/frame.h"

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
