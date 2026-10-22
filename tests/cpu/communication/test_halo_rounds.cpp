/// @file Unit tests for fractional periodic halo geometry.

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
        const double w = 1.0 / d[a];
        dd.sublo[a] = w * c[a];
        dd.subhi[a] = dd.sublo[a] + w;
    }
    dd.face_neighbors = std::array<int, 6>{1, 2, 3, 4, 5, 6};
    return dd;
}

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

TEST(TestHaloRounds, RoundOneBandIsOneSided) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);

    const int lo = swap_of(h, 0, 1, 0);
    const int hi = swap_of(h, 0, 1, 1);
    ASSERT_GE(lo, 0);
    ASSERT_GE(hi, 0);
    EXPECT_LT(h[lo].slablo, -1.0e6);
    EXPECT_DOUBLE_EQ(h[lo].slabhi, dd.sublo[0] + kShell / kBoxEdge);
    EXPECT_DOUBLE_EQ(h[hi].slablo, dd.subhi[0] - kShell / kBoxEdge);
    EXPECT_GT(h[hi].slabhi, 1.0e6);
}

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
    EXPECT_DOUBLE_EQ(h[lo].slabhi, dd.sublo[0] + kShell / kBoxEdge);
    EXPECT_DOUBLE_EQ(h[hi].slablo, dd.subhi[0] - kShell / kBoxEdge);
    EXPECT_DOUBLE_EQ(h[hi].slabhi, mid);

    // Neither edge is a sentinel: an open band is exactly the defect.
    EXPECT_GT(h[lo].slablo, -1.0e6);
    EXPECT_LT(h[lo].slabhi, 1.0e6);
    EXPECT_GT(h[hi].slablo, -1.0e6);
    EXPECT_LT(h[hi].slabhi, 1.0e6);
}

TEST(TestHaloRounds, ForwardedDepthIsTheReceiverShellRemainder) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);

    const double w = dd.subhi[0] - dd.sublo[0];
    const int lo = swap_of(h, 0, 2, 0);
    // Rows offered at round 2 came from the high face: they sit in
    // [subhi, subhi + w]. The band admits only up to sublo + shell.
    const double admitted_hi = std::min(dd.subhi[0] + w, h[lo].slabhi);
    EXPECT_NEAR(admitted_hi - dd.subhi[0], kShell / kBoxEdge - w, 1e-9);
    EXPECT_LT(admitted_hi - dd.subhi[0], w); // strictly less than a full slab
}

// ── 2. round>1 sendlist CONTENTS, not just counts ─────────────────────

TEST(TestHaloRounds, RoundTwoShipsExactlyTheRowsInsideItsBand) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    const int lo = swap_of(h, 0, 2, 0);
    ASSERT_GE(lo, 0);

    const double band_lo = h[lo].slablo;
    const double band_hi = h[lo].slabhi;

    // Six rows: two below the band, two inside, two above.
    const double xs[6] = {band_lo - 0.01,  band_lo - 0.005, band_lo + 0.001,
                          band_hi - 0.001, band_hi + 0.005, band_hi + 0.01};
    cpu::Frame f(6);
    f.box = cpu::Box::cubic(1.0);
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xs[i]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(dd.sublo[1] + 1.0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(dd.sublo[2] + 1.0);
    }

    h.rebuild_sendlist(lo, f, /*first=*/0, /*last=*/6);
    EXPECT_EQ(h[lo].sendlist, (std::vector<int>{2, 3}));
}

TEST(TestHaloRounds, RoundTwoBandEdgesAreInclusive) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    const int lo = swap_of(h, 0, 2, 0);

    cpu::Frame f(2);
    f.box = cpu::Box::cubic(1.0);
    f.span(kAtom_x)[0] = static_cast<StorageReal>(h[lo].slablo);
    f.span(kAtom_x)[1] = static_cast<StorageReal>(h[lo].slabhi);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_y)[i] = static_cast<StorageReal>(dd.sublo[1] + 1.0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(dd.sublo[2] + 1.0);
    }

    h.rebuild_sendlist(lo, f, /*first=*/0, /*last=*/2);
    EXPECT_EQ(h[lo].sendlist, (std::vector<int>{0, 1}));
}

// ── 3. the mid-plane edge sorts the shared window ─────────────────────

TEST(TestHaloRounds, MidPlaneEdgeSortsTheSharedWindowBetweenFaces) {
    cpu::comm::Halo h;
    const DomainDecomposition dd = brick_dd(8, 8, 4, 3, 3, 1);
    h.setup_uniform(dd, bench_box(), kShell, /*my_rank=*/0);
    const int lo = swap_of(h, 0, 2, 0);
    const int hi = swap_of(h, 0, 2, 1);
    ASSERT_GE(lo, 0);
    ASSERT_GE(hi, 0);

    // Row 0: a ghost delivered across the LOW face at round 1 — it lives
    // just below this subdomain. Row 1: one delivered across the HIGH
    // face, just above. Both are within the forwardable depth
    // (`shell - w` = 3.125 A here), so only the mid-plane edge tells them
    // apart.
    const double depth =
        0.5 * (kShell / kBoxEdge - (dd.subhi[0] - dd.sublo[0]));
    cpu::Frame f(2);
    f.box = cpu::Box::cubic(1.0);
    f.span(kAtom_x)[0] = static_cast<StorageReal>(dd.sublo[0] - depth);
    f.span(kAtom_x)[1] = static_cast<StorageReal>(dd.subhi[0] + depth);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_y)[i] = static_cast<StorageReal>(dd.sublo[1] + 1.0);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(dd.sublo[2] + 1.0);
    }

    h.rebuild_sendlist(lo, f, /*first=*/0, /*last=*/2);
    EXPECT_EQ(h[lo].sendlist, (std::vector<int>{1}));
    h.rebuild_sendlist(hi, f, /*first=*/0, /*last=*/2);
    EXPECT_EQ(h[hi].sendlist, (std::vector<int>{0}));
}

// ── 4. the 3-hop chain stays complete ─────────────────────────────────

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
        reach = std::min(static_cast<double>(r) * w, kShell / kBoxEdge);
    }
    // Three hops of 3.938 A cover 11.81 A >= the 11 A shell.
    EXPECT_GE(3.0 * w, kShell / kBoxEdge);
    EXPECT_NEAR(reach, kShell / kBoxEdge, 1e-9);

    // And the deepest round is still clipped at the receiver's shell edge,
    // not left open.
    const int deepest = swap_of(h, 0, 3, 0);
    EXPECT_DOUBLE_EQ(h[deepest].slabhi, dd.sublo[0] + kShell / kBoxEdge);
    EXPECT_LT(h[deepest].slabhi, 1.0e6);
}

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
                EXPECT_LE(h[i].slabhi, dd.sublo[d] + kShell / kBoxEdge + 1e-9);
            else
                EXPECT_GE(h[i].slablo, dd.subhi[d] - kShell / kBoxEdge - 1e-9);
        }
    }
}
