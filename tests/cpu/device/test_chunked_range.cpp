/// @file tests/cpu/device/test_chunked_range.cpp
/// @brief `cpu::device::ChunkedRange` partitions an index range into pieces
///        that are contiguous, disjoint, covering — and, above all, chosen
///        without reference to the OpenMP thread count.
///
/// Every reproducible reduction in the GPW grid kernels rests on this: a chunk's
/// partial sum is only meaningful if the chunk's extent is the same on one
/// thread and on sixteen.

#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/device/chunked_range.h"

using ATV::cpu::device::block_reduce_chunks;
using ATV::cpu::device::ChunkedRange;

TEST(ChunkedRange, ChunksTileTheRangeExactlyOnce) {
    for (long n : {1L, 2L, 7L, 64L, 65L, 1000L, 1'000'001L})
        for (long k : {1L, 2L, 3L, 8L, 64L, 256L}) {
            const ChunkedRange r(n, k);
            ASSERT_GT(r.chunks(), 0) << "n=" << n << " k=" << k;
            ASSERT_LE(r.chunks(), k) << "n=" << n << " k=" << k;
            EXPECT_EQ(r.begin(0), 0) << "n=" << n << " k=" << k;
            EXPECT_EQ(r.end(r.chunks() - 1), n) << "n=" << n << " k=" << k;
            for (long c = 0; c + 1 < r.chunks(); ++c)
                EXPECT_EQ(r.end(c), r.begin(c + 1))
                    << "n=" << n << " k=" << k << " c=" << c;
        }
}

TEST(ChunkedRange, ChunkSizesDifferByAtMostOne) {
    const ChunkedRange r(1000, 7);
    long lo = 1000, hi = 0;
    for (long c = 0; c < r.chunks(); ++c) {
        const long len = r.end(c) - r.begin(c);
        lo = std::min(lo, len);
        hi = std::max(hi, len);
    }
    EXPECT_LE(hi - lo, 1);
}

TEST(ChunkedRange, EmptyRangeHasNoChunks) {
    EXPECT_EQ(ChunkedRange(0, 8).chunks(), 0);
    EXPECT_EQ(ChunkedRange(-5, 8).chunks(), 0);
}

// A range shorter than the requested chunk count degenerates to one index per
// chunk rather than producing empty chunks.
TEST(ChunkedRange, ChunkCountNeverExceedsRangeLength) {
    const ChunkedRange r(3, 64);
    EXPECT_EQ(r.chunks(), 3);
    for (long c = 0; c < r.chunks(); ++c)
        EXPECT_EQ(r.end(c) - r.begin(c), 1);
}

// The block-accumulator budget must depend on the block size alone: a count
// that consulted the thread count would reintroduce the very defect the type
// exists to remove.
TEST(ChunkedRange, BlockReduceChunksFallsWithBlockSize) {
    EXPECT_EQ(block_reduce_chunks(1024), 64);       // tiny block: full width
    EXPECT_EQ(block_reduce_chunks(8u << 20), 8);    // 8 MiB block: 64 MiB / 8
    EXPECT_EQ(block_reduce_chunks(128u << 20), 1);  // over budget: never 0
    EXPECT_EQ(block_reduce_chunks(0), 64);          // empty block: full width
}
