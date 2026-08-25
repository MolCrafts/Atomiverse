/// @file tests/cpu/communication/test_swap_list.cpp
/// @brief Unit tests for `comm::SwapList` — the LAMMPS CommBrick swap table.
///
/// `SwapList` is a **purely local** data structure: it derives the ordered
/// swap table from a `DomainDecomposition` + `Box` + halo cutoff, and owns
/// the per-swap send lists and ghost-slot prefix sum. It performs no MPI,
/// so every case here builds the DD by hand (dims / coords / periods /
/// sublo / subhi / face_neighbors) and never initializes MPI. Multi-rank
/// transport correctness stays in `tests/mpi/test_halo_exchange.cpp`.
///
/// Behaviour pinned here (spec `.claude/specs/comm-topology-01-swaplist.md`):
///   * swap expansion order and count (`nswap = 2 * sum_d maxneed_d`),
///   * `pbc[3]` integer image flags on the periodic edges of the grid,
///   * `sendself` on non-decomposed axes (a **skip**, not a local memcpy),
///   * the **half-open** `slablo/slabhi` eligibility band,
///   * `commit_recv` as the ghost-slot prefix-sum invariant,
///   * the undecomposed-axis exemption from every width check
///     (the `need_d > 1` refusal itself is superseded by
///     `.claude/specs/comm-topology-02-multihop.md`: a decomposed axis now
///     derives multi-hop rounds, and only the minimum-image bound refuses
///     — see `DerivesMultiHopWhenNeedExceedsSub` below and
///     `test_swap_list_maxneed.cpp` for the hop counts),
///   * `invalidate()` clearing indices while keeping the swap geometry.

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

/// Box shared by every fixture: 20 x 10 x 16 Angstrom, orthorhombic.
cpu::Box test_box() { return cpu::Box::orthorhombic(20.0, 10.0, 16.0); }

/// @brief dims = {2,1,1} decomposition of the 20x10x16 box, seen from `rank`.
///
/// Sub-slab along x is 10 A wide; y and z are undecomposed (dims == 1), so
/// on those axes `MPI_Cart_shift` would hand back this rank itself — the
/// `sendself` case. With dims[0] == 2 the -x and +x neighbours are the *same*
/// other rank, which is exactly why the tests below can assert
/// `sendproc`/`recvproc` without pinning the lo/hi pairing convention.
DomainDecomposition split_x_dd(int rank) {
    DomainDecomposition dd;
    dd.dims[0] = 2;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = rank;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
    dd.sublo[0] = 10.0 * rank;
    dd.subhi[0] = 10.0 * rank + 10.0;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 10.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 16.0;
    const int other = 1 - rank;
    // {-x, +x, -y, +y, -z, +z}; undecomposed axes point at self.
    dd.face_neighbors =
        std::array<int, 6>{other, other, rank, rank, rank, rank};
    return dd;
}

/// @brief dims = {2,1,1} over a 30 x 8 x 16 box, seen from `rank`.
///
/// The x sub-slab is 15 A wide while the **undecomposed** y axis is only
/// 8 A long: a 12 A shell fits on x but is wider than the whole y axis.
/// That combination is legal today and must stay legal — see
/// `UndecomposedAxisExemptFromGuard`.
DomainDecomposition thin_y_axis_dd(int rank) {
    DomainDecomposition dd;
    dd.dims[0] = 2;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = rank;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
    dd.sublo[0] = 15.0 * rank;
    dd.subhi[0] = 15.0 * rank + 15.0;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 8.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 16.0;
    const int other = 1 - rank;
    dd.face_neighbors =
        std::array<int, 6>{other, other, rank, rank, rank, rank};
    return dd;
}

/// Box matching `thin_y_axis_dd`: 30 x 8 x 16 Angstrom.
cpu::Box thin_y_axis_box() { return cpu::Box::orthorhombic(30.0, 8.0, 16.0); }

/// @brief dims = {1,1,1}: one rank owns the whole box, every face is self.
DomainDecomposition single_rank_dd() {
    DomainDecomposition dd; // dims {1,1,1}, coords {0,0,0}, periods {1,1,1}
    dd.sublo[0] = 0.0;
    dd.subhi[0] = 20.0;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 10.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 16.0;
    dd.face_neighbors = std::array<int, 6>{0, 0, 0, 0, 0, 0};
    return dd;
}

/// @brief 12-row chain along x: 10 "owners" then 2 appended "ghost" rows.
///
/// Owner x = 0,1,...,9 A; rows 10 and 11 sit at x = -0.5 and x = 10.5 A —
/// where ghosts handed down by an earlier axis stage land for the rank-0
/// slab [0, 10). y and z are constant so only the x column decides.
cpu::Frame x_chain_frame() {
    constexpr int kRows = 12;
    const double xs[kRows] = {0.0, 1.0, 2.0, 3.0, 4.0,  5.0,
                              6.0, 7.0, 8.0, 9.0, -0.5, 10.5};
    cpu::Frame f(kRows);
    for (int i = 0; i < kRows; ++i) {
        f.span(kAtom_x)[i] = xs[i];
        f.span(kAtom_y)[i] = 5.0;
        f.span(kAtom_z)[i] = 8.0;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = test_box();
    return f;
}

void expect_pbc(const cpu::comm::Swap& s, int px, int py, int pz) {
    EXPECT_EQ(s.pbc[0], px);
    EXPECT_EQ(s.pbc[1], py);
    EXPECT_EQ(s.pbc[2], pz);
}

constexpr double kHaloCutoff = 3.0; // [Angstrom]

} // namespace

// ── 1. expansion ──────────────────────────────────────────────────────

/// Happy path: a 3-D periodic grid whose halo fits inside every sub-slab
/// expands to exactly six swaps in face order (0,lo),(0,hi),...,(2,hi) —
/// bit-for-bit the `face_neighbors` order the cascade halo used.
TEST(TestSwapList, HappyPathExpansion) {
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);
    cpu::comm::SwapList list;
    list.setup(dd, test_box(), kHaloCutoff, /*my_rank=*/0);

    // need_d = ceil(r_halo * dims_d / L_d): x -> ceil(0.3), y -> ceil(0.3),
    // z -> ceil(0.1875), all 1 => nswap = 2 * (1 + 1 + 1).
    EXPECT_EQ(list.maxneed(0), 1);
    EXPECT_EQ(list.maxneed(1), 1);
    EXPECT_EQ(list.maxneed(2), 1);
    ASSERT_EQ(static_cast<int>(list.size()), 6);

    const int expect_dim[6] = {0, 0, 1, 1, 2, 2};
    const int expect_sign[6] = {-1, +1, -1, +1, -1, +1};
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_EQ(list[i].dim, expect_dim[i]);
        EXPECT_EQ(list[i].sign, expect_sign[i]);
    }

    // dims[0] == 2: both x neighbours are rank 1, so this holds under either
    // lo/hi pairing convention. Undecomposed axes point back at this rank.
    EXPECT_EQ(list[0].sendproc, 1);
    EXPECT_EQ(list[0].recvproc, 1);
    EXPECT_EQ(list[1].sendproc, 1);
    EXPECT_EQ(list[1].recvproc, 1);
    for (int i = 2; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_EQ(list[i].sendproc, 0);
        EXPECT_EQ(list[i].recvproc, 0);
    }
}

// ── 2. periodic image flags ───────────────────────────────────────────

/// `pbc[3]` carries the *integer* image flag (multiplied by the current
/// `box.length(dim)` at pack time, never a metric offset baked in at setup).
/// Only a rank sitting on the global periodic edge shifts.
TEST(TestSwapList, PbcFlags) {
    cpu::comm::SwapList lo_edge;
    lo_edge.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff,
                  /*my_rank=*/0);
    // coords[0] == 0: the -x partner sits at the far side of the global box.
    expect_pbc(lo_edge[0], +1, 0, 0);
    // coords[0] == 0 != dims[0]-1: the +x partner is an interior neighbour.
    expect_pbc(lo_edge[1], 0, 0, 0);

    cpu::comm::SwapList hi_edge;
    hi_edge.setup(split_x_dd(/*rank=*/1), test_box(), kHaloCutoff,
                  /*my_rank=*/1);
    expect_pbc(hi_edge[0], 0, 0, 0);
    // coords[0] == dims[0]-1: the +x partner wraps to the low side.
    expect_pbc(hi_edge[1], -1, 0, 0);

    // Undecomposed axes transmit nothing (`sendself`), so they record no
    // image shift at all — periodicity there is the pair kernel's min-image.
    for (int i = 2; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        expect_pbc(lo_edge[i], 0, 0, 0);
        expect_pbc(hi_edge[i], 0, 0, 0);
    }
}

// ── 3. sendself is a skip ─────────────────────────────────────────────

/// On an axis with `dims[d] == 1` the Cartesian neighbour *is* this rank.
/// This engine **skips** those swaps rather than doing LAMMPS' local
/// self-image memcpy: `cell_list` still applies min-image on non-extended
/// axes, so a self-ghost there would be double-counted.
TEST(TestSwapList, SendselfSkip) {
    cpu::comm::SwapList list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);
    ASSERT_EQ(static_cast<int>(list.size()), 6);

    EXPECT_FALSE(list[0].sendself);
    EXPECT_FALSE(list[1].sendself);
    for (int i = 2; i < 6; ++i) { // y and z: dims == 1
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_TRUE(list[i].sendself);
        EXPECT_EQ(list[i].nrecv, 0);
    }
    EXPECT_EQ(list.total_recv(), 0);
}

// ── 4. the half-open eligibility band ─────────────────────────────────

/// The band comparison stays **strict** (`q - sublo < cutoff` /
/// `subhi - q < cutoff`), not LAMMPS' `<=`: an atom exactly on the band
/// edge is NOT shipped. x = 3.0 (lo edge) and x = 7.0 (hi edge) are in the
/// chain precisely to catch a slide to `<=`.
TEST(TestSwapList, SlabBandMatchesLegacyPredicate) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::SwapList list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);

    // Finite band edges, in this rank's own coordinate frame; the open side
    // carries a sentinel far outside any physical box.
    EXPECT_DOUBLE_EQ(list[0].slabhi, 3.0); // sublo[0] + cutoff
    EXPECT_LT(list[0].slablo, -1.0e6);     // -BIG
    EXPECT_DOUBLE_EQ(list[1].slablo, 7.0); // subhi[0] - cutoff
    EXPECT_GT(list[1].slabhi, 1.0e6);      // +BIG

    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/10);
    list.rebuild_sendlist(/*iswap=*/1, frame, /*sources_end=*/10);

    // q - 0.0 < 3.0  -> x = 0, 1, 2   (x = 3.0 excluded)
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2}));
    // 10.0 - q < 3.0 -> x = 8, 9      (x = 7.0 excluded)
    EXPECT_EQ(list[1].sendlist, (std::vector<int>{8, 9}));
}

/// Cascade semantics: the scan bound is the caller's, so a later axis stage
/// (owners + earlier-axis ghosts) picks up the appended ghost rows without
/// the swap table knowing anything about staging.
TEST(TestSwapList, SlabBandCascadeHonorsSourcesEnd) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::SwapList list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);

    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/12);
    list.rebuild_sendlist(/*iswap=*/1, frame, /*sources_end=*/12);

    // row 10 (x = -0.5) enters the lo band; row 11 (x = 10.5) the hi band.
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2, 10}));
    EXPECT_EQ(list[1].sendlist, (std::vector<int>{8, 9, 11}));

    // Shrinking the scan back drops them again (refill, not append).
    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/10);
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2}));
}

// ── 5. ghost slot prefix sum ──────────────────────────────────────────

/// `firstrecv[i] = n_local + sum_{j<i} nrecv[j]` becomes a type invariant,
/// replacing the `slot_start[6]` prefix sum hand-computed in `HaloRecv`.
TEST(TestSwapList, CommitRecvPrefixSum) {
    cpu::comm::SwapList list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);
    ASSERT_EQ(static_cast<int>(list.size()), 6);

    const int nrecv[6] = {3, 2, 0, 0, 0, 0};
    constexpr int n_local = 10;
    int next_slot = n_local;
    for (int i = 0; i < 6; ++i)
        next_slot = list.commit_recv(i, nrecv[i], next_slot);

    const int expect_first[6] = {10, 13, 15, 15, 15, 15};
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_EQ(list[i].nrecv, nrecv[i]);
        EXPECT_EQ(list[i].firstrecv, expect_first[i]);
    }
    EXPECT_EQ(next_slot, 15);
    EXPECT_EQ(list.total_recv(), 5);
}

// ── 6. multi-hop derivation instead of silent truncation ──────────────

/// **Superseded by `.claude/specs/comm-topology-02-multihop.md`.**
///
/// `-01` refused a decomposed axis whose ghost shell was deeper than the
/// rank's own sub-slab ("use a larger box, fewer ranks along that axis, or
/// a smaller cutoff"), because the halo exchanged with immediate
/// neighbours only. `-02` derives the extra hops instead — the shell is
/// filled by forwarding each round's ghosts one rank further — so
/// `need > sub` is no longer a refusal.
///
/// The remaining refusal is the **minimum-image** bound
/// `max_hops(d) <= dims[d] - 1` (`need <= L - w_self`); the second half of
/// this test keeps the `-01` geometry, which happens to sit exactly beyond
/// it (10.5 A shell on a 20 A box whose 2-rank slab is 10 A wide), and
/// pins that it still throws — for the new reason. The hop counts
/// themselves live in `tests/cpu/communication/test_swap_list_maxneed.cpp`.
TEST(TestSwapList, DerivesMultiHopWhenNeedExceedsSub) {
    // 20 A box split 4 ways: sub = 5 A, shell 6 A => ceil(1.2) = 2 hops,
    // and 2 <= dims - 1 = 3, so the minimum-image bound is satisfied.
    DomainDecomposition four_way;
    four_way.dims[0] = 4;
    four_way.dims[1] = 1;
    four_way.dims[2] = 1;
    four_way.coords[0] = 1;
    four_way.sublo[0] = 5.0;
    four_way.subhi[0] = 10.0;
    four_way.sublo[1] = 0.0;
    four_way.subhi[1] = 10.0;
    four_way.sublo[2] = 0.0;
    four_way.subhi[2] = 16.0;
    four_way.face_neighbors = std::array<int, 6>{0, 2, 1, 1, 1, 1};

    cpu::comm::SwapList list;
    ASSERT_NO_THROW(list.setup(four_way, test_box(), 6.0, /*my_rank=*/1));
    EXPECT_EQ(list.max_hops(0), 2);

    // Minimum-image bound: 10.5 A > L - w_self = 20 - 10 A. Still refused,
    // and no longer with the "fewer ranks" how-to.
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);
    cpu::comm::SwapList over;
    EXPECT_THROW(over.setup(dd, test_box(), 10.5, /*my_rank=*/0),
                 std::runtime_error);
}

/// The ceil convention keeps `r_halo == sub_width` a **single** hop, which
/// is what the outgoing guard did (it rejected only `need > sub`, letting
/// equality through). Both the exact edge and one epsilon below it pass.
TEST(TestSwapList, AcceptsCutoffUpToSubdomainWidth) {
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);

    cpu::comm::SwapList just_under;
    EXPECT_NO_THROW(
        just_under.setup(dd, test_box(), 10.0 - 1e-9, /*my_rank=*/0));
    EXPECT_EQ(just_under.maxneed(0), 1);
    EXPECT_EQ(static_cast<int>(just_under.size()), 6);

    cpu::comm::SwapList exact;
    EXPECT_NO_THROW(exact.setup(dd, test_box(), 10.0, /*my_rank=*/0));
    EXPECT_EQ(exact.maxneed(0), 1);
    EXPECT_EQ(static_cast<int>(exact.size()), 6);
}

/// **Undecomposed axes are exempt from the guard.** Today's guard short-
/// circuits `dims[d] <= 1` *before* comparing (`halo_exchange.cpp:498-499`),
/// so a shell wider than the entire axis runs fine there: that axis grows no
/// ghosts at all and its periodicity is the cell list's min-image on
/// non-extended axes. `setup` must replicate the short-circuit **ahead of**
/// the ceil, or an EAM-style `2*cutoff` shell on a thin axis turns a run
/// that is legal today into a hard throw.
TEST(TestSwapList, UndecomposedAxisExemptFromGuard) {
    const DomainDecomposition dd = thin_y_axis_dd(/*rank=*/0);
    cpu::comm::SwapList list;
    // 12 A shell: fits the 15 A x sub-slab, but L_y = 8 A would give a naive
    // ceil(12 * 1 / 8) = 2 on the undecomposed y axis.
    ASSERT_NO_THROW(list.setup(dd, thin_y_axis_box(), 12.0, /*my_rank=*/0));

    EXPECT_EQ(list.maxneed(0), 1); // ceil(12 * 2 / 30) = 1, decomposed
    EXPECT_EQ(list.maxneed(1), 1); // forced by dims[1] == 1, not derived
    EXPECT_EQ(list.maxneed(2), 1);
    ASSERT_EQ(static_cast<int>(list.size()), 6);

    for (int i = 2; i < 4; ++i) { // the thin y axis
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_EQ(list[i].dim, 1);
        EXPECT_TRUE(list[i].sendself);
        EXPECT_EQ(list[i].nrecv, 0);
    }
    EXPECT_FALSE(list[0].sendself); // x is still a real exchange
    EXPECT_FALSE(list[1].sendself);
    EXPECT_EQ(list.total_recv(), 0);
}

// ── 7. single rank ────────────────────────────────────────────────────

/// One rank owns everything: every swap is `sendself`, so the table exists
/// but moves nothing and no ghost slot is ever claimed.
TEST(TestSwapList, AllSelfWhenSingleRank) {
    cpu::comm::SwapList list;
    list.setup(single_rank_dd(), test_box(), kHaloCutoff, /*my_rank=*/0);

    ASSERT_EQ(static_cast<int>(list.size()), 6);
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_TRUE(list[i].sendself);
        EXPECT_EQ(list[i].sendproc, 0);
        EXPECT_EQ(list[i].recvproc, 0);
        EXPECT_EQ(list[i].nrecv, 0);
        expect_pbc(list[i], 0, 0, 0);
    }
    EXPECT_EQ(list.total_recv(), 0);
}

// ── 8. invalidation ───────────────────────────────────────────────────

/// `invalidate()` is the post-sort / post-migration hook: every stored row
/// index refers to frame slots that just moved, so the index side is dropped
/// — but the *geometry* (which rank, which band, which image flag) is
/// unaffected by a reshuffle and must survive, so the next `borders()` pass
/// only has to re-scan, not re-derive the table.
TEST(TestSwapList, InvalidateClearsIndicesKeepsGeometry) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::SwapList list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);
    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/10);
    list.rebuild_sendlist(/*iswap=*/1, frame, /*sources_end=*/10);
    int next_slot = 10;
    next_slot = list.commit_recv(0, 3, next_slot);
    next_slot = list.commit_recv(1, 2, next_slot);
    ASSERT_FALSE(list[0].sendlist.empty());
    ASSERT_EQ(list.total_recv(), 5);

    // Snapshot the geometry as plain scalars: `operator[]` is the only
    // element accessor the spec declares.
    int was_dim[6], was_sign[6], was_sendproc[6], was_recvproc[6];
    bool was_sendself[6];
    int was_pbc[6][3];
    double was_slablo[6], was_slabhi[6];
    for (int i = 0; i < 6; ++i) {
        was_dim[i] = list[i].dim;
        was_sign[i] = list[i].sign;
        was_sendproc[i] = list[i].sendproc;
        was_recvproc[i] = list[i].recvproc;
        was_sendself[i] = list[i].sendself;
        for (int d = 0; d < 3; ++d)
            was_pbc[i][d] = list[i].pbc[d];
        was_slablo[i] = list[i].slablo;
        was_slabhi[i] = list[i].slabhi;
    }

    list.invalidate();

    ASSERT_EQ(static_cast<int>(list.size()), 6);
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        // indices dropped
        EXPECT_TRUE(list[i].sendlist.empty());
        EXPECT_EQ(list[i].nrecv, 0);
        // geometry kept
        EXPECT_EQ(list[i].dim, was_dim[i]);
        EXPECT_EQ(list[i].sign, was_sign[i]);
        EXPECT_EQ(list[i].sendproc, was_sendproc[i]);
        EXPECT_EQ(list[i].recvproc, was_recvproc[i]);
        EXPECT_EQ(list[i].sendself, was_sendself[i]);
        expect_pbc(list[i], was_pbc[i][0], was_pbc[i][1], was_pbc[i][2]);
        EXPECT_DOUBLE_EQ(list[i].slablo, was_slablo[i]);
        EXPECT_DOUBLE_EQ(list[i].slabhi, was_slabhi[i]);
    }
    EXPECT_EQ(list.total_recv(), 0);
}
