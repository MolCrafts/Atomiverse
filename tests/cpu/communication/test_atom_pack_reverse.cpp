/// @file tests/cpu/communication/test_atom_pack_reverse.cpp
/// @brief Unit tests for the reverse wire codec.
///
/// `comm-maint-02-dedup` moved `pack_reverse_rows` / `accumulate_reverse_rows`
/// out of `comm_brick.cpp`'s anonymous namespace into `atom_pack`, beside the
/// three forward codecs (`pack_comm`, `pack_border`, `pack_exchange`) that
/// were already public there. Reverse had been the odd one out — file-scoped,
/// and therefore only reachable through a multi-rank end-to-end run. These
/// tests are what that move buys.
///
/// Surface pinned (no MPI, no halo, no Driver):
/// @code{.cpp}
///   PackPlan    make_reverse_plan(Frame&);
///   std::size_t plan_row_bytes(const PackPlan&);
///   std::size_t reverse_payload_bytes(const PackPlan&, int);
///   void        pack_reverse_rows(const PackPlan&, int, int,
///                                 std::vector<std::byte>&);
///   void        accumulate_reverse_rows(const PackPlan&,
///                                       const std::vector<int>&,
///                                       const std::vector<std::byte>&);
/// @endcode
///
/// Categories: basics (sizing), domain (the accumulate contract is `+=`, not
/// `=`, because several ghosts of one owner fold in across swaps), edge
/// (empty payload).
///
/// Reference: LAMMPS `src/atom_vec.cpp` `AtomVec::unpack_reverse`
/// (`f[j][0] += buf[m++]`), baseline
/// `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-src` @ 78a7ddf.

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Frame with `n` rows and a known force pattern: row i carries
/// (i + 1, 10 * (i + 1), 100 * (i + 1)) eV/A.
cpu::Frame forces_frame(int n) {
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_fx)[i] = static_cast<AccReal>(i + 1);
        f.span(kAtom_fy)[i] = static_cast<AccReal>(10 * (i + 1));
        f.span(kAtom_fz)[i] = static_cast<AccReal>(100 * (i + 1));
    }
    return f;
}

} // namespace

// ── 1. basics: sizing ─────────────────────────────────────────────────

/// The reverse wire is the three force accumulators, so one row is three
/// `AccReal` and `reverse_payload_bytes` is linear in the row count.
TEST(TestAtomPackReverse, PayloadIsThreeAccumulatorsPerRow) {
    cpu::Frame f = forces_frame(4);
    const cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    EXPECT_EQ(plan.fields.size(), 3u);
    EXPECT_EQ(plan.row_bytes(), 3 * sizeof(AccReal));
    EXPECT_EQ(plan.bytes(0), 0u);
    EXPECT_EQ(plan.bytes(4), 4 * 3 * sizeof(AccReal));
}

// ── 2. domain: the pack/accumulate round trip ─────────────────────────

/// Pack a contiguous ghost run, fold it onto a disjoint set of owner rows,
/// and the owners must end up holding exactly what the ghosts carried.
///
/// This is the property the transport depends on and the one nothing could
/// check while the codec was file-scoped.
TEST(TestAtomPackReverse, RoundTripRecoversTheGhostForces) {
    // Rows 0-2 are owners (zeroed), rows 3-5 are the "ghosts" to fold back.
    cpu::Frame f = forces_frame(6);
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_fx)[i] = 0.0;
        f.span(kAtom_fy)[i] = 0.0;
        f.span(kAtom_fz)[i] = 0.0;
    }
    cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/3, /*count=*/3, wire);
    ASSERT_EQ(wire.size(), plan.bytes(3));

    // Ghost 3 belongs to owner 0, ghost 4 to owner 1, ghost 5 to owner 2.
    const std::vector<int> owners{0, 1, 2};
    plan.accumulate(owners, wire);

    for (int i = 0; i < 3; ++i) {
        SCOPED_TRACE("owner " + std::to_string(i));
        const AccReal src = static_cast<AccReal>(i + 4); // ghost row i+3
        EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[i], src);
        EXPECT_DOUBLE_EQ(f.span(kAtom_fy)[i], 10.0 * src);
        EXPECT_DOUBLE_EQ(f.span(kAtom_fz)[i], 100.0 * src);
    }
}

/// Accumulation is `+=`, not `=`. One owner can receive from several ghosts
/// — different swaps, different periodic images — and every contribution has
/// to survive. Folding the same payload twice must double the result.
TEST(TestAtomPackReverse, AccumulateAddsRatherThanOverwrites) {
    cpu::Frame f = forces_frame(4);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_fx)[i] = 0.0;
        f.span(kAtom_fy)[i] = 0.0;
        f.span(kAtom_fz)[i] = 0.0;
    }
    cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/2, /*count=*/2, wire);

    const std::vector<int> owners{0, 1};
    plan.accumulate(owners, wire);
    plan.accumulate(owners, wire);

    // Ghost row 2 carries 3, row 3 carries 4; folded twice.
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 2.0 * 3.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[1], 2.0 * 4.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fz)[1], 2.0 * 400.0);
}

/// Several ghosts of the SAME owner fold onto that one row — the multi-hop
/// case, where a row can be a ghost of one rank at two periodic images.
TEST(TestAtomPackReverse, ManyGhostsFoldOntoOneOwner) {
    cpu::Frame f = forces_frame(4);
    f.span(kAtom_fx)[0] = 0.0;
    f.span(kAtom_fy)[0] = 0.0;
    f.span(kAtom_fz)[0] = 0.0;
    cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/1, /*count=*/3, wire);

    plan.accumulate(std::vector<int>{0, 0, 0}, wire);

    // Ghost rows 1, 2, 3 carry fx = 2, 3, 4.
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 2.0 + 3.0 + 4.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fy)[0], 10.0 * (2.0 + 3.0 + 4.0));
}

// ── 3. edge ───────────────────────────────────────────────────────────

/// A swap that received nothing folds nothing and touches no owner.
TEST(TestAtomPackReverse, EmptyPayloadIsANoOp) {
    cpu::Frame f = forces_frame(3);
    cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/0, /*count=*/0, wire);
    EXPECT_TRUE(wire.empty());

    plan.accumulate(std::vector<int>{}, wire);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[0], 1.0);
    EXPECT_DOUBLE_EQ(f.span(kAtom_fx)[2], 3.0);
}

/// A row count that disagrees with the payload is a wire-protocol error and
/// must throw rather than read past the buffer.
TEST(TestAtomPackReverse, MismatchedRowCountThrows) {
    cpu::Frame f = forces_frame(4);
    cpu::comm::PackPlan plan = cpu::comm::make_reverse_plan(f);

    std::vector<std::byte> wire;
    plan.pack_rows(/*first=*/0, /*count=*/2, wire);

    // Three owner rows against a two-row payload.
    EXPECT_THROW(plan.accumulate(std::vector<int>{0, 1, 2}, wire),
                 std::exception);
}
