/// @file tests/cuda/test_domain_decomposition.cu
/// @brief cuda::DomainDecomposition — 3D factorization, slab bounds, periodic
///        face neighbors, position->owner. Pure host logic (no GPU used).

#include <gtest/gtest.h>
#include <set>

#include "atomiverse/cuda/comm/domain_decomposition.cuh"

using namespace ATV;
using DD = cuda::DomainDecomposition;

TEST(DomainDecomposition, SingleGpuTrivial) {
    DD dd(1, 10.0, 10.0, 10.0);
    EXPECT_EQ(dd.count(), 1);
    EXPECT_EQ(dd.dims(), (std::array<int, 3>{1, 1, 1}));
    EXPECT_EQ(dd.owner_of(3, 7, 1), 0);
    for (int f = 0; f < 6; ++f)
        EXPECT_EQ(dd.face_neighbor(0, f), 0); // own periodic neighbor
}

TEST(DomainDecomposition, BalancedFactorization) {
    EXPECT_EQ(DD(2, 10, 10, 10).dims(), (std::array<int, 3>{2, 1, 1}));
    EXPECT_EQ(DD(4, 10, 10, 10).dims(), (std::array<int, 3>{2, 2, 1}));
    EXPECT_EQ(DD(8, 10, 10, 10).dims(), (std::array<int, 3>{2, 2, 2}));
    // Cubic box, n=2 splits the (tied) shortest surface — any single axis is
    // valid; just assert the product is correct.
    auto d3 = DD(3, 10, 10, 10).dims();
    EXPECT_EQ(d3[0] * d3[1] * d3[2], 3);
}

TEST(DomainDecomposition, FactorizationFollowsBoxShape) {
    // A long-x box: n=2 should split along x (smallest resulting surface).
    auto d = DD(2, 40.0, 10.0, 10.0).dims();
    EXPECT_EQ(d, (std::array<int, 3>{2, 1, 1}));
}

TEST(DomainDecomposition, SlabsTileTheBox) {
    DD dd(8, 12.0, 12.0, 12.0); // 2x2x2, each slab 6x6x6
    double vol = 0.0;
    std::set<int> seen;
    for (int s = 0; s < dd.count(); ++s) {
        auto lo = dd.sublo(s), hi = dd.subhi(s);
        vol += (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
        seen.insert(s);
        for (int d = 0; d < 3; ++d) {
            EXPECT_GE(lo[d], 0.0);
            EXPECT_LE(hi[d], 12.0 + 1e-9);
            EXPECT_GT(hi[d], lo[d]);
        }
    }
    EXPECT_EQ(seen.size(), 8u);
    EXPECT_NEAR(vol, 12.0 * 12.0 * 12.0, 1e-6); // slabs exactly tile the box
}

TEST(DomainDecomposition, OwnerMatchesSlab) {
    DD dd(8, 12.0, 12.0, 12.0);
    // Every atom's owner slab must actually contain it.
    for (double x = 0.5; x < 12; x += 2.0)
        for (double y = 0.5; y < 12; y += 2.0)
            for (double z = 0.5; z < 12; z += 2.0) {
                int s = dd.owner_of(x, y, z);
                auto lo = dd.sublo(s), hi = dd.subhi(s);
                EXPECT_GE(x, lo[0]);
                EXPECT_LT(x, hi[0]);
                EXPECT_GE(y, lo[1]);
                EXPECT_LT(y, hi[1]);
                EXPECT_GE(z, lo[2]);
                EXPECT_LT(z, hi[2]);
            }
}

TEST(DomainDecomposition, OwnerWrapsPbc) {
    DD dd(2, 10.0, 10.0, 10.0);                                 // 2x1x1
    EXPECT_EQ(dd.owner_of(-1.0, 5, 5), dd.owner_of(9.0, 5, 5)); // -1 wraps to 9
    EXPECT_EQ(dd.owner_of(11.0, 5, 5), dd.owner_of(1.0, 5, 5)); // 11 wraps to 1
}

TEST(DomainDecomposition, FaceNeighborsPeriodicInvolution) {
    DD dd(8, 12.0, 12.0, 12.0); // 2x2x2: dims==2 each axis
    for (int s = 0; s < dd.count(); ++s)
        for (int axis = 0; axis < 3; ++axis) {
            int plus = dd.face_neighbor(s, 2 * axis + 1);
            int minus = dd.face_neighbor(s, 2 * axis);
            // dims==2 → +neighbor and -neighbor are the same other slot, and
            // applying twice returns to self.
            EXPECT_EQ(plus, minus);
            EXPECT_EQ(dd.face_neighbor(plus, 2 * axis), s);
            EXPECT_NE(plus, s);
        }
}
