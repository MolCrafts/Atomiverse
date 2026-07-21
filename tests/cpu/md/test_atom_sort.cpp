/// @file tests/cpu/md/test_atom_sort.cpp
/// @brief Spatial sort reorders owned atoms by bin; tags/positions stay bound.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/atom_sort.h"

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
    f.simbox = cpu::Box::cubic(10.0);

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
    f.simbox = cpu::Box::cubic(4.0);
    cpu::md::mpi::DomainDecomposition dd;
    dd.bind(f);
    const auto g0 = dd.neighbor_generation;
    cpu::md::notify_sorted(&dd, f);
    EXPECT_EQ(dd.neighbor_generation, g0 + 1);
    // Owner tag map always rebuilt after sort (specials / gather / bonded).
    EXPECT_EQ(dd.tag_to_local.size(), 4u);
}
