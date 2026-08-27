/// @file tests/cpu/communication/test_halo_queries.cpp
/// @brief Unit tests for the `Swap` / `Halo` queries that `comm-maint-02-dedup`
///        moved out of `comm_brick.cpp`'s anonymous namespace.
///
/// These six were pure functions of the swap table all along, but file scope
/// put them out of reach of any test: the only thing that could exercise them
/// was a multi-rank end-to-end run. Now they are on their owning types, so
/// they get pinned directly.
///
/// Surface pinned:
///   Swap::is_silent, Swap::kNoPartner, Swap::pbc_shift
///   Halo::stage_end, Halo::round_end, Halo::round_begin, Halo::mirror_swap
///   Halo::setup_uniform
///
/// Categories: basics / edge / domain. No MPI — `Halo` is MPI-free by
/// contract, which is exactly why these are testable at all.

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/box.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

/// @brief Production bench geometry: 63.0016 A cube on an 8x8x4 grid with an
/// 11 A shell — the case that gives hops 2,2,1.
constexpr double kBoxEdge = 63.0016;
constexpr double kShell = 11.0;

cpu::Box bench_box() {
    return cpu::Box::orthorhombic(kBoxEdge, kBoxEdge, kBoxEdge);
}

/// @brief Uniform brick DD at `coords` on a `dims` grid over the bench box.
DomainDecomposition brick_dd(int dx, int dy, int dz, int cx, int cy, int cz) {
    DomainDecomposition dd;
    dd.dims[0] = dx;
    dd.dims[1] = dy;
    dd.dims[2] = dz;
    dd.coords[0] = cx;
    dd.coords[1] = cy;
    dd.coords[2] = cz;
    dd.periods[0] = dd.periods[1] = dd.periods[2] = 1;
    const int d[3] = {dx, dy, dz};
    const int c[3] = {cx, cy, cz};
    for (int a = 0; a < 3; ++a) {
        const double w = kBoxEdge / d[a];
        dd.sublo[a] = w * c[a];
        dd.subhi[a] = dd.sublo[a] + w;
    }
    // Distinct non-self partners on every axis; the exact ids do not matter
    // for the queries under test, only that they are not this rank.
    dd.face_neighbors = std::array<int, 6>{1, 2, 3, 4, 5, 6};
    return dd;
}

} // namespace

// ── Swap::is_silent and the no-partner sentinel ───────────────────────

/// A swap with a real partner moves data.
TEST(TestHaloQueries, SwapWithPartnerIsNotSilent) {
    cpu::comm::Swap s;
    s.sendproc = 7;
    s.sendself = false;
    EXPECT_FALSE(s.is_silent());
}

/// Both silencing reasons are honoured, and they are the only two.
TEST(TestHaloQueries, SwapIsSilentOnSelfOrNoPartner) {
    cpu::comm::Swap self;
    self.sendproc = 3;
    self.sendself = true;
    EXPECT_TRUE(self.is_silent());

    cpu::comm::Swap orphan;
    orphan.sendproc = cpu::comm::Swap::kNoPartner;
    orphan.sendself = false;
    EXPECT_TRUE(orphan.is_silent());
}

/// The sentinel is the value the transport actually writes. `Halo` cannot
/// name `MPI_PROC_NULL` (it is MPI-free by contract), so `comm_brick.cpp`
/// carries a static_assert that the two agree; this pins the value the
/// header promises so a silent change shows up here too.
TEST(TestHaloQueries, NoPartnerSentinelIsMinusOne) {
    EXPECT_EQ(cpu::comm::Swap::kNoPartner, -1);
}

// ── Swap::pbc_shift ───────────────────────────────────────────────────

/// An interior swap shifts nothing.
TEST(TestHaloQueries, PbcShiftIsZeroWithoutImageFlags) {
    cpu::comm::Swap s;
    double out[3] = {9.0, 9.0, 9.0};
    s.pbc_shift(bench_box(), out);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);
    EXPECT_DOUBLE_EQ(out[2], 0.0);
}

/// The displacement is formed against the CURRENT box, so a rescaled cell
/// cannot ship a stale shift. Same flags, two boxes, two answers.
TEST(TestHaloQueries, PbcShiftTracksTheCurrentBox) {
    cpu::comm::Swap s;
    s.pbc[0] = +1;
    s.pbc[2] = -1;

    double out[3] = {};
    s.pbc_shift(bench_box(), out);
    EXPECT_DOUBLE_EQ(out[0], kBoxEdge);
    EXPECT_DOUBLE_EQ(out[1], 0.0);
    EXPECT_DOUBLE_EQ(out[2], -kBoxEdge);

    const cpu::Box grown =
        cpu::Box::orthorhombic(2.0 * kBoxEdge, kBoxEdge, 0.5 * kBoxEdge);
    s.pbc_shift(grown, out);
    EXPECT_DOUBLE_EQ(out[0], 2.0 * kBoxEdge);
    EXPECT_DOUBLE_EQ(out[2], -0.5 * kBoxEdge);
}

// ── stage / round partitioning of the swap table ──────────────────────

/// Single-hop geometry: six swaps, three stages of two, each stage one round.
TEST(TestHaloQueries, SingleHopStagesAreFacePairs) {
    cpu::comm::Halo h;
    // 21 A subdomains against an 11 A shell => one hop per axis.
    h.setup_uniform(brick_dd(3, 3, 3, 1, 1, 1), bench_box(), kShell,
                    /*my_rank=*/0);
    ASSERT_EQ(static_cast<int>(h.size()), 6);

    for (int begin = 0; begin < 6; begin += 2) {
        SCOPED_TRACE("stage at " + std::to_string(begin));
        EXPECT_EQ(h.stage_end(begin), begin + 2);
        EXPECT_EQ(h.round_end(begin), begin + 2);
        EXPECT_EQ(h.round_begin(begin + 2), begin);
    }
}

/// Multi-hop geometry: the bench grid. An axis stage spans all its rounds,
/// while a round is still just the (lo, hi) pair — that distinction is the
/// whole reason both queries exist.
TEST(TestHaloQueries, MultiHopStageSpansRoundsButRoundIsAPair) {
    cpu::comm::Halo h;
    h.setup_uniform(brick_dd(8, 8, 4, 3, 3, 1), bench_box(), kShell,
                    /*my_rank=*/0);
    // hops 2,2,1 => 2*(2+2+1) = 10 swaps.
    ASSERT_EQ(static_cast<int>(h.size()), 10);
    EXPECT_EQ(h.max_hops(0), 2);
    EXPECT_EQ(h.max_hops(1), 2);
    EXPECT_EQ(h.max_hops(2), 1);

    // Axis 0 owns swaps [0, 4): two rounds of two.
    EXPECT_EQ(h.stage_end(0), 4);
    EXPECT_EQ(h.round_end(0), 2);
    EXPECT_EQ(h.round_end(2), 4);
    // Axis 1 owns [4, 8), axis 2 owns [8, 10).
    EXPECT_EQ(h.stage_end(4), 8);
    EXPECT_EQ(h.stage_end(8), 10);
    EXPECT_EQ(h.round_end(8), 10);
}

/// `round_begin` is the inverse of `round_end`, which is what lets the
/// reverse pass walk the table backwards deepest-hop-first.
TEST(TestHaloQueries, RoundBeginInvertsRoundEnd) {
    cpu::comm::Halo h;
    h.setup_uniform(brick_dd(8, 8, 4, 3, 3, 1), bench_box(), kShell,
                    /*my_rank=*/0);
    const int n = static_cast<int>(h.size());
    for (int begin = 0; begin < n;) {
        const int end = h.round_end(begin);
        SCOPED_TRACE("round [" + std::to_string(begin) + ", "
                     + std::to_string(end) + ")");
        EXPECT_EQ(h.round_begin(end), begin);
        begin = end;
    }
}

// ── mirror_swap ───────────────────────────────────────────────────────

/// Within a round the two faces mirror onto each other, and mirroring twice
/// is the identity — the property the receive tag depends on.
TEST(TestHaloQueries, MirrorSwapIsAnInvolutionWithinTheRound) {
    EXPECT_EQ(cpu::comm::Halo::mirror_swap(4, 6, 4), 5);
    EXPECT_EQ(cpu::comm::Halo::mirror_swap(4, 6, 5), 4);
    for (int s = 4; s < 6; ++s) {
        const int m = cpu::comm::Halo::mirror_swap(4, 6, s);
        EXPECT_EQ(cpu::comm::Halo::mirror_swap(4, 6, m), s);
    }
}

/// A one-swap round (an undecomposed axis keeps a lone placeholder pair in
/// some geometries) mirrors onto itself rather than off the end.
TEST(TestHaloQueries, MirrorSwapOfASingletonRoundIsItself) {
    EXPECT_EQ(cpu::comm::Halo::mirror_swap(7, 8, 7), 7);
}

// ── setup_uniform is the rank-uniform entry point ─────────────────────

/// Every rank on an axis must expand the same number of rounds, or a partner
/// posts a receive nobody answers. `setup_uniform` sweeps all coordinates, so
/// a corner rank and an interior rank agree.
TEST(TestHaloQueries, SetupUniformGivesEveryRankTheSameTable) {
    cpu::comm::Halo corner;
    cpu::comm::Halo interior;
    corner.setup_uniform(brick_dd(8, 8, 4, 0, 0, 0), bench_box(), kShell, 0);
    interior.setup_uniform(brick_dd(8, 8, 4, 4, 5, 2), bench_box(), kShell, 0);

    EXPECT_EQ(corner.size(), interior.size());
    for (int d = 0; d < 3; ++d) {
        SCOPED_TRACE("axis " + std::to_string(d));
        EXPECT_EQ(corner.maxneed(d), interior.maxneed(d));
        EXPECT_EQ(corner.max_hops(d), interior.max_hops(d));
    }
}
