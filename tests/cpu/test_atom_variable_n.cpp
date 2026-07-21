/// @file tests/cpu/test_atom_variable_n.cpp
/// @brief Unit tests for the variable-N Frame structural primitives used by
///        GCMC: `Frame::copy_row` and the global `Frame::acquire_tag`.

#include <gtest/gtest.h>

#include <random>
#include <set>

#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Fill row `i` with distinct, recognizable values keyed by `seed`.
void fill_row(cpu::Frame& frame, int i, double seed) {
    frame.span(kAtom_x)[i] = seed + 0.1;
    frame.span(kAtom_y)[i] = seed + 0.2;
    frame.span(kAtom_z)[i] = seed + 0.3;
    frame.span(kAtom_vx)[i] = seed + 0.4;
    frame.span(kAtom_vy)[i] = seed + 0.5;
    frame.span(kAtom_vz)[i] = seed + 0.6;
    frame.span(kAtom_fx)[i] = seed + 0.7;
    frame.span(kAtom_fy)[i] = seed + 0.8;
    frame.span(kAtom_fz)[i] = seed + 0.9;
    frame.span(kAtom_mass)[i] = seed + 1.0;
    frame.span(kAtom_charge)[i] = seed + 1.1;
    frame.span(kAtom_type)[i] = static_cast<int>(seed);
    frame.span(kAtom_tag)[i] = static_cast<TagInt>(seed) + 100;
    auto image = frame.span(kAtom_image);
    image(i, 0) = 1;
    image(i, 1) = -1;
    image(i, 2) = 2;
}

} // namespace

TEST(AtomCopyRow, CopiesEverySoAField) {
    cpu::Frame frame(4);
    fill_row(frame, 3, 30.0);
    fill_row(frame, 1, 10.0);

    frame.copy_row(1, 3);

#define EXPECT_COLUMN_COPIED(key)                                              \
    EXPECT_EQ(frame.span(key)[1], frame.span(key)[3])
    EXPECT_COLUMN_COPIED(kAtom_x);
    EXPECT_COLUMN_COPIED(kAtom_y);
    EXPECT_COLUMN_COPIED(kAtom_z);
    EXPECT_COLUMN_COPIED(kAtom_vx);
    EXPECT_COLUMN_COPIED(kAtom_vy);
    EXPECT_COLUMN_COPIED(kAtom_vz);
    EXPECT_COLUMN_COPIED(kAtom_fx);
    EXPECT_COLUMN_COPIED(kAtom_fy);
    EXPECT_COLUMN_COPIED(kAtom_fz);
    EXPECT_COLUMN_COPIED(kAtom_mass);
    EXPECT_COLUMN_COPIED(kAtom_charge);
    EXPECT_COLUMN_COPIED(kAtom_type);
    EXPECT_COLUMN_COPIED(kAtom_tag);
#undef EXPECT_COLUMN_COPIED
    const auto image = frame.span(kAtom_image);
    for (int d = 0; d < 3; ++d)
        EXPECT_EQ(image(1, d), image(3, d));
}

TEST(AtomCopyRow, TopologyFollowsTheRow) {
    cpu::Frame frame(4);
    // Tags are identity by construction; bond 3-0 is owned by atom 0
    // (smaller tag), angle 0-3-2 by its center atom 3.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, 4);
    frame.add_bond(/*ti=*/3, /*tj=*/0, /*type=*/7);
    frame.add_angle(/*ti=*/0, /*tj=*/3, /*tk=*/2, /*type=*/5);

    frame.copy_row(1, 3);

    EXPECT_EQ(frame.span(kAtom_num_angle)[1], frame.span(kAtom_num_angle)[3]);
    EXPECT_EQ(frame.span(kAtom_angle_atom)(1, 0),
              frame.span(kAtom_angle_atom)(3, 0));
    EXPECT_EQ(frame.span(kAtom_angle_atom)(1, 1),
              frame.span(kAtom_angle_atom)(3, 1));
    EXPECT_EQ(frame.span(kAtom_angle_type)(1, 0),
              frame.span(kAtom_angle_type)(3, 0));

    frame.copy_row(2, 0); // bond owner row moves too
    EXPECT_EQ(frame.span(kAtom_num_bond)[2], 1);
    EXPECT_EQ(frame.span(kAtom_bond_atom)(2, 0),
              frame.span(kAtom_bond_atom)(0, 0));
    EXPECT_EQ(frame.span(kAtom_bond_type)(2, 0), 7);
}

TEST(AtomCopyRow, SameRowIsANoOp) {
    cpu::Frame frame(2);
    fill_row(frame, 0, 5.0);
    frame.copy_row(0, 0);
    EXPECT_EQ(frame.span(kAtom_x)[0], StorageReal(5.1));
    EXPECT_EQ(frame.span(kAtom_tag)[0], 105);
}

TEST(AtomGrow, ChunkedGrowPreservesEveryLiveRow) {
    // Regression pin for a real data-loss bug: grow() trusted
    // xt::xtensor::resize to keep the prefix, and it does NOT — the first
    // GCMC insertion past the initial capacity (a chunked 64 → 4160 grow)
    // zeroed every live atom, collapsed them onto the origin, and froze the
    // Markov chain on a NaN energy. grow() now reallocates through
    // set_capacity's explicit copy.
    cpu::Frame frame(45); // ctor floors capacity at 64
    for (int i = 0; i < 45; ++i) {
        fill_row(frame, i, 10.0 * i);
        frame.span(kAtom_tag)[i] = 1000 + i;
    }
    const int nmax0 = frame.capacity();
    frame.grow(frame.capacity() + 1); // forces the chunked reallocation
    ASSERT_GT(frame.capacity(), nmax0);
    for (int i = 0; i < 45; ++i) {
        EXPECT_EQ(frame.span(kAtom_x)[i], StorageReal(10.0 * i + 0.1))
            << "row " << i;
        EXPECT_EQ(frame.span(kAtom_vz)[i], StorageReal(10.0 * i + 0.6));
        EXPECT_EQ(frame.span(kAtom_mass)[i], StorageReal(10.0 * i + 1.0));
        EXPECT_EQ(frame.span(kAtom_tag)[i], 1000 + i);
    }
    // The new tail is zeroed, tags identity-seeded (the grow contract).
    EXPECT_EQ(frame.span(kAtom_x)[50], 0.0);
    EXPECT_EQ(frame.span(kAtom_tag)[50], 50);
}

TEST(AtomAcquireTag, SeedsPastTheLiveMaximum) {
    cpu::Frame frame(3);                // identity tags 0,1,2
    frame.span(ATV::kAtom_tag)[1] = 41; // a restart-style sparse tag
    EXPECT_EQ(frame.acquire_tag(), 42);
    EXPECT_EQ(frame.acquire_tag(), 43); // monotonic thereafter
}

TEST(AtomAcquireTag, UniqueAcrossRandomInsertDeleteCycles) {
    // ac-004: 1000 random insert/delete cycles never produce a duplicate
    // tag. Delete = compact-last-into-victim (the GCMC deletion primitive);
    // insert = write a fresh row at nlocal with acquire_tag().
    cpu::Frame frame(8);
    std::mt19937_64 rng(2026u);

    for (int cycle = 0; cycle < 1000; ++cycle) {
        std::uniform_int_distribution<int> coin(0, 1);
        if (frame.nlocal() > 1 && coin(rng)) {
            std::uniform_int_distribution<int> pick(0, frame.nlocal() - 1);
            const int victim = pick(rng);
            frame.copy_row(victim, frame.nlocal() - 1);
            frame.set_nlocal(frame.nlocal() - 1);
        } else {
            const int row = frame.nlocal();
            if (row + 1 > frame.capacity())
                frame.grow(row + 1);
            frame.span(kAtom_tag)[row] = frame.acquire_tag();
            frame.set_nlocal(row + 1);
        }
        std::set<TagInt> seen;
        const auto tags = frame.span(kAtom_tag);
        for (int i = 0; i < frame.nlocal(); ++i)
            EXPECT_TRUE(seen.insert(tags[i]).second)
                << "duplicate tag " << tags[i] << " at cycle " << cycle;
    }
}
