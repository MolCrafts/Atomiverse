/// @file Unit tests for fractional periodic halo geometry.

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>
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

cpu::Box test_box() { return cpu::Box::orthorhombic(20.0, 10.0, 16.0); }

DomainDecomposition split_x_dd(int rank) {
    DomainDecomposition dd;
    dd.dims[0] = 2;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = rank;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
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
    dd.compute_sublo_subhi(test_box());
    return dd;
}

DomainDecomposition thin_y_axis_dd(int rank) {
    DomainDecomposition dd;
    dd.dims[0] = 2;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = rank;
    dd.coords[1] = 0;
    dd.coords[2] = 0;
    dd.sublo[0] = 15.0 * rank;
    dd.subhi[0] = 15.0 * rank + 15.0;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 8.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 16.0;
    const int other = 1 - rank;
    dd.face_neighbors =
        std::array<int, 6>{other, other, rank, rank, rank, rank};
    dd.compute_sublo_subhi(test_box());
    return dd;
}

cpu::Box thin_y_axis_box() { return cpu::Box::orthorhombic(30.0, 8.0, 16.0); }

DomainDecomposition single_rank_dd() {
    DomainDecomposition dd; // dims {1,1,1}, coords {0,0,0}
    dd.sublo[0] = 0.0;
    dd.subhi[0] = 20.0;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = 10.0;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = 16.0;
    dd.face_neighbors = std::array<int, 6>{0, 0, 0, 0, 0, 0};
    dd.compute_sublo_subhi(test_box());
    return dd;
}

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

TEST(TestHalo, HappyPathExpansion) {
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);
    cpu::comm::Halo list;
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

TEST(TestHalo, PbcFlags) {
    cpu::comm::Halo lo_edge;
    lo_edge.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff,
                  /*my_rank=*/0);
    // coords[0] == 0: the -x partner sits at the far side of the global box.
    expect_pbc(lo_edge[0], +1, 0, 0);
    // coords[0] == 0 != dims[0]-1: the +x partner is an interior neighbour.
    expect_pbc(lo_edge[1], 0, 0, 0);

    cpu::comm::Halo hi_edge;
    hi_edge.setup(split_x_dd(/*rank=*/1), test_box(), kHaloCutoff,
                  /*my_rank=*/1);
    expect_pbc(hi_edge[0], 0, 0, 0);
    // coords[0] == dims[0]-1: the +x partner wraps to the low side.
    expect_pbc(hi_edge[1], -1, 0, 0);

    // Undecomposed axes transmit nothing (`sendself`), so they record no
    // image shift at all — periodicity there is the pair kernel's min-image.
    for (int i = 2; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_EQ(lo_edge[i].pbc[i / 2], i % 2 == 0 ? 1 : -1);
        EXPECT_EQ(hi_edge[i].pbc[i / 2], i % 2 == 0 ? 1 : -1);
    }
}

// ── 3. sendself is a skip ─────────────────────────────────────────────

TEST(TestHalo, SendselfSkip) {
    cpu::comm::Halo list;
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

// ── 4. the inclusive eligibility band ─────────────────────────────────

TEST(TestHalo, SlabBandIsInclusiveOnBothEdges) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::Halo list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);

    // Finite band edges, in this rank's own coordinate frame; the open side
    // carries a sentinel far outside any physical box.
    EXPECT_DOUBLE_EQ(list[0].slabhi, 3.0 / 20.0); // sublo[0] + cutoff
    EXPECT_LT(list[0].slablo, -1.0e6);            // -BIG
    EXPECT_DOUBLE_EQ(list[1].slablo, 7.0 / 20.0); // subhi[0] - cutoff
    EXPECT_GT(list[1].slabhi, 1.0e6);             // +BIG

    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/10);
    list.rebuild_sendlist(/*iswap=*/1, frame, /*sources_end=*/10);

    // q <= 3.0  -> x = 0, 1, 2, 3   (row 3 sits ON the edge and ships)
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2, 3}));
    // q >= 7.0  -> x = 7, 8, 9      (row 7 sits ON the edge and ships)
    EXPECT_EQ(list[1].sendlist, (std::vector<int>{7, 8, 9}));
}

TEST(TestHalo, SlabBandCascadeHonorsSourcesEnd) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::Halo list;
    list.setup(split_x_dd(/*rank=*/0), test_box(), kHaloCutoff, /*my_rank=*/0);

    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/12);
    list.rebuild_sendlist(/*iswap=*/1, frame, /*sources_end=*/12);

    // row 10 (x = -0.5) enters the lo band; row 11 (x = 10.5) the hi band.
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2, 3, 10}));
    EXPECT_EQ(list[1].sendlist, (std::vector<int>{7, 8, 9, 11}));

    // Shrinking the scan back drops them again (refill, not append).
    list.rebuild_sendlist(/*iswap=*/0, frame, /*sources_end=*/10);
    EXPECT_EQ(list[0].sendlist, (std::vector<int>{0, 1, 2, 3}));
}

// ── 5. ghost slot prefix sum ──────────────────────────────────────────

TEST(TestHalo, CommitRecvPrefixSum) {
    cpu::comm::Halo list;
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

TEST(TestHalo, DerivesMultiHopWhenNeedExceedsSub) {
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

    four_way.compute_sublo_subhi(test_box());
    cpu::comm::Halo list;
    ASSERT_NO_THROW(list.setup(four_way, test_box(), 6.0, /*my_rank=*/1));
    EXPECT_EQ(list.max_hops(0), 2);

    // Minimum-image bound: 10.5 A > L - w_self = 20 - 10 A. Still refused,
    // and no longer with the "fewer ranks" how-to.
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);
    cpu::comm::Halo over;
    ASSERT_NO_THROW(over.setup(dd, test_box(), 10.5, /*my_rank=*/0));
    EXPECT_EQ(over.max_hops(0), 2);
}

TEST(TestHalo, AcceptsCutoffUpToSubdomainWidth) {
    const DomainDecomposition dd = split_x_dd(/*rank=*/0);

    cpu::comm::Halo just_under;
    EXPECT_NO_THROW(
        just_under.setup(dd, test_box(), 10.0 - 1e-9, /*my_rank=*/0));
    EXPECT_EQ(just_under.maxneed(0), 1);
    EXPECT_EQ(static_cast<int>(just_under.size()), 6);

    cpu::comm::Halo exact;
    EXPECT_NO_THROW(exact.setup(dd, test_box(), 10.0, /*my_rank=*/0));
    EXPECT_EQ(exact.maxneed(0), 2);
    EXPECT_EQ(static_cast<int>(exact.size()), 10);
}

TEST(TestHalo, UndecomposedAxisExemptFromGuard) {
    const DomainDecomposition dd = thin_y_axis_dd(/*rank=*/0);
    cpu::comm::Halo list;
    // 12 A shell: fits the 15 A x sub-slab, but L_y = 8 A would give a naive
    // ceil(12 * 1 / 8) = 2 on the undecomposed y axis.
    ASSERT_NO_THROW(list.setup(dd, thin_y_axis_box(), 12.0, /*my_rank=*/0));

    EXPECT_EQ(list.maxneed(0), 1); // ceil(12 * 2 / 30) = 1, decomposed
    EXPECT_EQ(list.maxneed(1), 2); // forced by dims[1] == 1, not derived
    EXPECT_EQ(list.maxneed(2), 1);
    ASSERT_EQ(static_cast<int>(list.size()), 8);

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

TEST(TestHalo, AllSelfWhenSingleRank) {
    cpu::comm::Halo list;
    list.setup(single_rank_dd(), test_box(), kHaloCutoff, /*my_rank=*/0);

    ASSERT_EQ(static_cast<int>(list.size()), 6);
    for (int i = 0; i < 6; ++i) {
        SCOPED_TRACE("swap " + std::to_string(i));
        EXPECT_TRUE(list[i].sendself);
        EXPECT_EQ(list[i].sendproc, 0);
        EXPECT_EQ(list[i].recvproc, 0);
        EXPECT_EQ(list[i].nrecv, 0);
        EXPECT_EQ(list[i].pbc[i / 2], i % 2 == 0 ? 1 : -1);
    }
    EXPECT_EQ(list.total_recv(), 0);
}

// ── 8. invalidation ───────────────────────────────────────────────────

TEST(TestHalo, InvalidateClearsIndicesKeepsGeometry) {
    const cpu::Frame frame = x_chain_frame();
    cpu::comm::Halo list;
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

TEST(Halo, TriclinicSlabsUseReciprocalWidthsAndFullImageVectors) {
    Mat3 cell = Mat3::diagonal(20.0, 10.0, 16.0);
    cell(0, 1) = 4.0;
    cell(1, 0) = 3.0;
    cpu::Box box(cell);
    auto dd = split_x_dd(0);
    dd.cuts[0] = {0.0, 0.3, 1.0};
    dd.compute_sublo_subhi(box);
    EXPECT_DOUBLE_EQ(dd.sublo[0], 0.0);
    EXPECT_DOUBLE_EQ(dd.subhi[0], 0.3);
    cpu::comm::Halo halo;
    halo.setup_uniform(dd, box, 2.0, 0);
    EXPECT_NEAR(halo[0].slabhi, 2.0 / box.face_height(0), 1e-15);
    double shift[3];
    halo[0].pbc_shift(box, shift);
    EXPECT_DOUBLE_EQ(shift[0], 20.0);
    EXPECT_DOUBLE_EQ(shift[1], 3.0);
    EXPECT_DOUBLE_EQ(shift[2], 0.0);
    EXPECT_TRUE(halo[2].sendself);
    EXPECT_FALSE(halo[2].is_silent());
    halo[2].pbc_shift(box, shift);
    EXPECT_DOUBLE_EQ(shift[0], 4.0);
    EXPECT_DOUBLE_EQ(shift[1], 10.0);

    cpu::Frame frame(2);
    frame.box = box;
    for (int i = 0; i < 2; ++i) {
        const Vec3 point = box.to_cartesian({i == 0 ? 0.05 : 0.2, 0.95, 0.5});
        frame.span(kAtom_x)[i] = point[0];
        frame.span(kAtom_y)[i] = point[1];
        frame.span(kAtom_z)[i] = point[2];
    }
    halo.rebuild_sendlist(0, frame, 2);
    ASSERT_EQ(halo[0].sendlist.size(), 1u);
    EXPECT_EQ(halo[0].sendlist[0], 0);
}

TEST(Halo, OversizedTriclinicImposedRoundsPreserveExistingTable) {
    Mat3 cell = Mat3::diagonal(20.0, 10.0, 16.0);
    cell(0, 1) = 3.0;
    cpu::Box box(cell);
    auto dd = split_x_dd(0);
    dd.compute_sublo_subhi(box);
    cpu::comm::Halo halo;
    halo.setup_uniform(dd, box, 2.0, 0);
    const auto previous_size = halo.size();
    const int previous_hops = halo.max_hops(0);
    const int excessive[3] = {std::numeric_limits<int>::max(), 1, 1};
    EXPECT_THROW(halo.setup(dd, box, 2.0, 0, excessive), std::length_error);
    EXPECT_EQ(halo.size(), previous_size);
    EXPECT_EQ(halo.max_hops(0), previous_hops);
}

TEST(Halo, GhostSlotArithmeticRejectsOverflowBeforeMutation) {
    auto dd = split_x_dd(0);
    cpu::comm::Halo halo;
    halo.setup(dd, test_box(), 2.0, 0);
    const int maximum = std::numeric_limits<int>::max();
    EXPECT_THROW(halo.commit_recv(0, 1, maximum), std::overflow_error);
    EXPECT_EQ(halo[0].nrecv, 0);
    EXPECT_EQ(halo[0].firstrecv, 0);
    EXPECT_EQ(halo.commit_recv(0, maximum, 0), maximum);
    EXPECT_EQ(halo.total_recv(), maximum);
    // Independent records still cannot overflow the total count query.
    EXPECT_EQ(halo.commit_recv(1, 1, 0), 1);
    EXPECT_THROW((void)halo.total_recv(), std::overflow_error);
}
