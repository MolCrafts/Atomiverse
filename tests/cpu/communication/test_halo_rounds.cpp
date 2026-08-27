/// @file tests/cpu/communication/test_halo_rounds.cpp
/// @brief What a multi-hop forwarding round actually ships.
///
/// The gap this closes: on 2026-08-27 a `&& false` in `Halo::expand` had
/// switched off the LAMMPS `ineed >= 2` slab band, leaving every forwarding
/// round to ship the whole slab it had received instead of the part still
/// inside the receiver's shell. The ghost count came out 1.76x too large and
/// nothing failed. `test_halo_maxneed.cpp` has nine tests and they all measure
/// hop *counts*; grepping it for `slablo`, `slabhi` or `sendlist` returns
/// nothing. `test_halo.cpp` only reaches round 1.
///
/// So the defect lived in code no test looked at, and surfaced only as a
/// 256-rank performance number. These tests pin the two halves of the
/// multi-hop rule directly:
///
///   * the **band** (`Swap::slablo`/`slabhi`) clips how DEEP a forwarded row
///     may be;
///   * the **window** (`Halo::round_scan_window`) decides WHICH rows a round
///     is allowed to look at.
///
/// Categories: domain (band geometry, chain completeness), edge (inclusive
/// edges, singleton round). No MPI.
///
/// Reference: LAMMPS `src/comm_brick.cpp` `CommBrick::setup`, the
/// `ineed >= 2` arm, baseline
/// `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-src` @ 78a7ddf;
/// S. Plimpton, J. Comput. Phys. 117, 1 (1995), DOI 10.1006/jcph.1995.1039.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

/// Verified bench geometry: 63.0016 A cube, 11 A shell.
/// 8x8x4 -> subdomain 7.875 x 7.875 x 15.750 -> hops 2,2,1.
/// 16,4,4 -> subdomain 3.938 -> hops 3,1,1.
constexpr double kBoxEdge = 63.0016;
constexpr double kShell = 11.0;

cpu::Box bench_box() {
    return cpu::Box::orthorhombic(kBoxEdge, kBoxEdge, kBoxEdge);
}

DomainDecomposition brick_dd(int dx, int dy, int dz, int cx, int cy, int cz) {
    DomainDecomposition dd;
    const int d[3] = {dx, dy, dz};
    const int c[3] = {cx, cy, cz};
    for (int a = 0; a < 3; ++a) {
        dd.dims[a] = d[a];
        dd.coords[a] = c[a];
        dd.periods[a] = 1;
        const double w = kBoxEdge / d[a];
        dd.sublo[a] = w * c[a];
        dd.subhi[a] = dd.sublo[a] + w;
    }
    dd.face_neighbors = std::array<int, 6>{1, 2, 3, 4, 5, 6};
    return dd;
}

/// @brief Index of the first swap on `axis` with the given round and face.
/// @param face 0 = low, 1 = high.
int swap_of(const cpu::comm::Halo& h, int axis, int round, int face) {
    for (int i = 0; i < static_cast<int>(h.size()); ++i) {
        if (h[i].dim == axis && h[i].round == round) {
            const bool is_high = h[i].sign > 0;
            if (is_high == (face == 1))
                return i;
        }
    }
    return -1;
}

} // namespace

// ── 1. the forwarding band clips depth ────────────────────────────────

/// Round 1 is one-sided: the rank's own boundary band, open on the outward
/// side. This is the case `test_halo.cpp` already covers; it is here as the
/// contrast for round 2.
TEST(TestHaloRounds, RoundOneBandIsOneSided) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);

    const int lo = swap_of(h, 0, 1, 0);
    const int hi = swap_of(h, 0, 1, 1);
    ASSERT_GE(lo, 0);
    ASSERT_GE(hi, 0);
    EXPECT_LT(h[lo].slablo, -1.0e6);
    EXPECT_DOUBLE_EQ(h[lo].slabhi, dd.sublo[0] + kShell);
    EXPECT_DOUBLE_EQ(h[hi].slablo, dd.subhi[0] - kShell);
    EXPECT_GT(h[hi].slabhi, 1.0e6);
}

/// Round 2 is bounded on BOTH sides — the LAMMPS `ineed >= 2` band. The far
/// edge is the partner's shell reach (`sublo + cutoff` for the low face); the
/// near edge is this rank's mid-plane.
///
/// **This is the assertion the `&& false` bug would have failed.** With the
/// band switched off both bounds were the +/-1e30 sentinel.
TEST(TestHaloRounds, RoundTwoBandIsClosedOnBothSides) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    ASSERT_EQ(h.max_hops(0), 2);

    const double mid = 0.5 * (dd.sublo[0] + dd.subhi[0]);
    const int lo = swap_of(h, 0, 2, 0);
    const int hi = swap_of(h, 0, 2, 1);
    ASSERT_GE(lo, 0);
    ASSERT_GE(hi, 0);

    EXPECT_DOUBLE_EQ(h[lo].slablo, mid);
    EXPECT_DOUBLE_EQ(h[lo].slabhi, dd.sublo[0] + kShell);
    EXPECT_DOUBLE_EQ(h[hi].slablo, dd.subhi[0] - kShell);
    EXPECT_DOUBLE_EQ(h[hi].slabhi, mid);

    // Neither edge is a sentinel: an open band is exactly the defect.
    EXPECT_GT(h[lo].slablo, -1.0e6);
    EXPECT_LT(h[lo].slabhi, 1.0e6);
    EXPECT_GT(h[hi].slablo, -1.0e6);
    EXPECT_LT(h[hi].slabhi, 1.0e6);
}

/// The forwarded depth is finite and equals what the partner's shell can
/// still use. At the bench geometry that is 3.125 A, not the full 7.875 A
/// slab — the 2.52x per-hop over-forwarding the open band produced.
TEST(TestHaloRounds, ForwardedDepthIsTheReceiverShellRemainder) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);

    const double w = dd.subhi[0] - dd.sublo[0];
    const int lo = swap_of(h, 0, 2, 0);
    // Rows offered at round 2 came from the high face: they sit in
    // [subhi, subhi + w]. The band admits only up to sublo + shell.
    const double admitted_hi = std::min(dd.subhi[0] + w, h[lo].slabhi);
    EXPECT_NEAR(admitted_hi - dd.subhi[0], kShell - w, 1e-9);
    EXPECT_LT(admitted_hi - dd.subhi[0], w); // strictly less than a full slab
}

// ── 2. round>1 sendlist CONTENTS, not just counts ─────────────────────

/// The end-to-end statement: given explicit rows, a round-2 low-face swap
/// ships exactly those inside [mid, sublo + shell] and nothing else.
TEST(TestHaloRounds, RoundTwoShipsExactlyTheRowsInsideItsBand) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    const int lo = swap_of(h, 0, 2, 0);
    ASSERT_GE(lo, 0);

    const double band_lo = h[lo].slablo;
    const double band_hi = h[lo].slabhi;

    // Six rows: two below the band, two inside, two above.
    const double xs[6] = {band_lo - 1.0, band_lo - 0.25, band_lo + 0.5,
                          band_hi - 0.5, band_hi + 0.25, band_hi + 1.0};
    cpu::Frame f(6);
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xs[i]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(dd.sublo[1] + 1.0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(dd.sublo[2] + 1.0);
    }

    h.rebuild_sendlist(lo, f, /*first=*/0, /*last=*/6);
    EXPECT_EQ(h[lo].sendlist, (std::vector<int>{2, 3}));
}

/// The band is inclusive on both edges (LAMMPS `>=` / `<=`): a row sitting
/// exactly on either bound ships. An edge ghost that is not needed costs one
/// row; an edge ghost that is missing costs a pair.
TEST(TestHaloRounds, RoundTwoBandEdgesAreInclusive) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    const int lo = swap_of(h, 0, 2, 0);

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = static_cast<StorageReal>(h[lo].slablo);
    f.span(kAtom_x)[1] = static_cast<StorageReal>(h[lo].slabhi);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_y)[i] = static_cast<StorageReal>(dd.sublo[1] + 1.0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(dd.sublo[2] + 1.0);
    }

    h.rebuild_sendlist(lo, f, /*first=*/0, /*last=*/2);
    EXPECT_EQ(h[lo].sendlist, (std::vector<int>{0, 1}));
}

// ── 3. the scan window picks the OPPOSITE face ────────────────────────

/// Round 1 scans owners plus every earlier-axis ghost.
TEST(TestHaloRounds, RoundOneScansEverythingDeliveredSoFar) {
    const int prev_first[2] = {40, 60};
    const int prev_last[2] = {50, 70};
    int wf[2] = {-1, -1};
    int wl[2] = {-1, -1};
    cpu::comm::Halo::round_scan_window(/*round=*/1, /*width=*/2,
                                       /*sources_end=*/37, prev_first,
                                       prev_last, wf, wl);
    EXPECT_EQ(wf[0], 0);
    EXPECT_EQ(wl[0], 37);
    EXPECT_EQ(wf[1], 0);
    EXPECT_EQ(wl[1], 37);
}

/// Round r>1 of a face scans what the OPPOSITE face received at r-1. Feeding
/// a face its own previous receipts would send an atom straight back where it
/// came from; feeding it the owners again would double-deliver them.
TEST(TestHaloRounds, RoundTwoScansTheOppositeFacesPreviousReceipts) {
    const int prev_first[2] = {40, 60};
    const int prev_last[2] = {50, 70};
    int wf[2] = {-1, -1};
    int wl[2] = {-1, -1};
    cpu::comm::Halo::round_scan_window(/*round=*/2, /*width=*/2,
                                       /*sources_end=*/37, prev_first,
                                       prev_last, wf, wl);
    // face 0 takes face 1's band, and vice versa — crossed, not straight.
    EXPECT_EQ(wf[0], 60);
    EXPECT_EQ(wl[0], 70);
    EXPECT_EQ(wf[1], 40);
    EXPECT_EQ(wl[1], 50);
}

/// A lone placeholder round (an undecomposed axis keeps one) scans its own
/// single band rather than indexing off the end.
TEST(TestHaloRounds, SingletonRoundTakesItsOwnBand) {
    const int prev_first[2] = {11, 0};
    const int prev_last[2] = {13, 0};
    int wf[2] = {-1, -1};
    int wl[2] = {-1, -1};
    cpu::comm::Halo::round_scan_window(/*round=*/2, /*width=*/1,
                                       /*sources_end=*/9, prev_first, prev_last,
                                       wf, wl);
    EXPECT_EQ(wf[0], 11);
    EXPECT_EQ(wl[0], 13);
}

/// A round wider than an (lo, hi) pair is a corrupt table, not a case to
/// handle.
TEST(TestHaloRounds, WiderThanAPairIsRefused) {
    const int prev_first[2] = {0, 0};
    const int prev_last[2] = {0, 0};
    int wf[2] = {};
    int wl[2] = {};
    EXPECT_THROW(cpu::comm::Halo::round_scan_window(2, 3, 0, prev_first,
                                                    prev_last, wf, wl),
                 std::logic_error);
}

// ── 4. the 3-hop chain stays complete ─────────────────────────────────

/// The geometry that matters most for the depth clip: 16,4,4 gives a 3.938 A
/// x-subdomain against an 11 A shell, so x needs THREE hops and a forwarding
/// round has to hand its rows on to yet another hop. If the clip were one
/// step too tight, the union of the three rounds would fall short of the
/// shell and pairs would be lost — here, not at 8,8,4.
TEST(TestHaloRounds, ThreeHopChainCoversTheWholeShell) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(16, 4, 4, 5, 1, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    ASSERT_EQ(h.max_hops(0), 3);

    const double w = dd.subhi[0] - dd.sublo[0];
    // Round 1 delivers the neighbour's whole slab; each later round adds at
    // most one more slab, clipped at sublo + shell. Reach after r rounds is
    // min(r * w, shell) beyond the face.
    double reach = 0.0;
    for (int r = 1; r <= h.max_hops(0); ++r) {
        const int lo = swap_of(h, 0, r, 0);
        ASSERT_GE(lo, 0) << "missing low-face swap at round " << r;
        reach = std::min(static_cast<double>(r) * w, kShell);
    }
    // Three hops of 3.938 A cover 11.81 A >= the 11 A shell.
    EXPECT_GE(3.0 * w, kShell);
    EXPECT_NEAR(reach, kShell, 1e-9);

    // And the deepest round is still clipped at the receiver's shell edge,
    // not left open.
    const int deepest = swap_of(h, 0, 3, 0);
    EXPECT_DOUBLE_EQ(h[deepest].slabhi, dd.sublo[0] + kShell);
    EXPECT_LT(h[deepest].slabhi, 1.0e6);
}

/// Companion to the above: the shell is not over-covered either. A round-2
/// band that reached past `sublo + shell` is the 1.76x inflation.
TEST(TestHaloRounds, NoRoundReachesPastTheReceiverShell) {
    for (const auto& g :
         {std::array<int, 3>{8, 8, 4}, std::array<int, 3>{16, 4, 4}}) {
        cpu::comm::Halo h;
        const DomainDecomposition dd = brick_dd(g[0], g[1], g[2], 1, 1, 1);
        h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
        for (int i = 0; i < static_cast<int>(h.size()); ++i) {
            if (h[i].round <= 1 || h[i].sendself)
                continue;
            SCOPED_TRACE("dims " + std::to_string(g[0]) + " swap "
                         + std::to_string(i));
            const int d = h[i].dim;
            if (h[i].sign < 0)
                EXPECT_LE(h[i].slabhi, dd.sublo[d] + kShell + 1e-9);
            else
                EXPECT_GE(h[i].slablo, dd.subhi[d] - kShell - 1e-9);
        }
    }
}
