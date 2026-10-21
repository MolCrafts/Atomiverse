/// @file tests/cuda/potentials/test_packed_tiles.cu
/// @brief Direct baked-mask packing checks; no force evaluation or MD.
#include "../../../src/cuda/potentials/internal/packed_tiles.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include <gtest/gtest.h>
#include <vector>

namespace {
template <class T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(std::size_t size) {
        if (size)
            ATV_CUDA_CHECK(cudaMalloc(&data, size * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(data); }
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (!values.empty())
            ATV_CUDA_CHECK(cudaMemcpyAsync(data, values.data(),
                                           values.size() * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
    }
    std::vector<T> download(std::size_t size, cudaStream_t stream) const {
        std::vector<T> result(size);
        if (size)
            ATV_CUDA_CHECK(cudaMemcpyAsync(result.data(), data,
                                           size * sizeof(T),
                                           cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
};

// Component-private primitive fixture. Every output and CUB buffer is
// caller-preallocated, exactly as in PairLJCutCoulLong::build.
struct Input {
    cudaStream_t stream = nullptr;
    DeviceArray<int> rows, columns, counts, offsets, ci, j;
    DeviceArray<std::uint32_t> masks, packed_masks;
    DeviceArray<char> temp;
    ATV::cuda::PairList source;
    ATV::cuda::internal::PackedTileWorkspace work;
    Input(const std::vector<int>& host_rows,
          const std::vector<int>& host_columns)
        : rows(host_rows.size()), columns(host_columns.size()),
          counts(host_rows.size()), offsets(host_rows.size()),
          ci(host_columns.size()), j(host_columns.size() * 32),
          masks(host_columns.size() * 32),
          packed_masks(host_columns.size() * 32),
          temp(ATV::cuda::internal::packed_tiles_scan_bytes(
              int(host_rows.size()) - 1)) {
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        rows.upload(host_rows, stream);
        columns.upload(host_columns, stream);
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        source.num_clusters = int(host_rows.size()) - 1;
        source.cl_row_ptr = rows.data;
        source.cl_col_idx = columns.data;
        source.cl_mask = reinterpret_cast<const std::uint64_t*>(masks.data);
        work.counts = counts.data;
        work.offsets = offsets.data;
        work.ci = ci.data;
        work.j = j.data;
        work.masks = packed_masks.data;
        work.capacity = int(host_columns.size());
        work.scan_temp = temp.data;
        work.scan_bytes =
            ATV::cuda::internal::packed_tiles_scan_bytes(source.num_clusters);
        ATV_CUDA_CHECK(cudaMallocHost(&work.host_count, sizeof(int)));
    }
    ~Input() {
        cudaStreamSynchronize(stream);
        cudaFreeHost(work.host_count);
        cudaStreamDestroy(stream);
    }
    int pack(const std::vector<std::uint32_t>& bits) {
        masks.upload(bits, stream);
        return ATV::cuda::internal::pack_tiles(source, work, stream);
    }
};
} // namespace

TEST(CudaPackedTiles, EmptyRowsBakedExclusionsAndDiagonalLowerBitsStayAbsent) {
    Input input({0, 0, 2, 3}, {0, 1, 2});
    std::vector<std::uint32_t> bits(3 * 32, 0);
    for (int lane = 0; lane < 32; ++lane) {
        bits[lane] = 0xffffffffu; // Row 1, lower cluster 0: all discarded.
        bits[32 + lane] = lane == 31 ? 0xffffffffu : (1u << (lane + 1)) - 1;
        // Row 2's diagonal is fully baked out (all zeros).
    }
    EXPECT_EQ(input.pack(bits), 0);
    EXPECT_EQ(input.offsets.download(4, input.stream),
              (std::vector<int>{0, 0, 0, 0}));
}

TEST(CudaPackedTiles,
     PacksAcrossClustersAndPadsPartialTileWithoutRevivingBits) {
    Input input({0, 3, 3, 3}, {0, 1, 2});
    std::vector<std::uint32_t> bits(3 * 32, 0);
    bits[0] = 3;            // Self bit removed; upper-diagonal J=1 is retained.
    bits[1] = 1;            // Lower-diagonal interaction must not reappear.
    bits[32] = 0xffffffffu; // All 32 columns of cluster 1 for I lane 0.
    bits[65] = 1u << 3;     // Cluster 2, I lane 1, J slot 67.
    ASSERT_EQ(input.pack(bits), 2);
    EXPECT_EQ(input.ci.download(2, input.stream), (std::vector<int>{0, 0}));
    std::vector<int> expected_j(64, -1);
    expected_j[0] = 1;
    for (int k = 0; k < 32; ++k)
        expected_j[k + 1] = 32 + k;
    expected_j[33] = 67;
    EXPECT_EQ(input.j.download(64, input.stream), expected_j);
    std::vector<std::uint32_t> expected_mask(64, 0);
    expected_mask[0] = 0xffffffffu;
    expected_mask[32] = 1;
    expected_mask[33] = 2;
    EXPECT_EQ(input.packed_masks.download(64, input.stream), expected_mask);

    // Reuse the identical output buffers with fewer tiles. Only the new
    // active prefix is consumed; no previous column or mask may survive.
    bits.assign(3 * 32, 0);
    EXPECT_EQ(input.pack(bits), 0);
    bits[95] = 1u << 31;
    ASSERT_EQ(input.pack(bits), 1);
    expected_j.assign(32, -1);
    expected_j[0] = 95;
    EXPECT_EQ(input.j.download(32, input.stream), expected_j);
    expected_mask.assign(32, 0);
    expected_mask[31] = 1;
    EXPECT_EQ(input.packed_masks.download(32, input.stream), expected_mask);
}

TEST(CudaPackedTiles, ExactFullTileDoesNotEmitExtraPaddingTile) {
    Input input({0, 1, 1}, {1});
    std::vector<std::uint32_t> bits(32, 0);
    bits[7] = 0xffffffffu;
    ASSERT_EQ(input.pack(bits), 1);
    EXPECT_EQ(input.offsets.download(3, input.stream),
              (std::vector<int>{0, 1, 1}));
    std::vector<int> expected_j(32);
    for (int k = 0; k < 32; ++k)
        expected_j[k] = 32 + k;
    EXPECT_EQ(input.j.download(32, input.stream), expected_j);
    EXPECT_EQ(input.packed_masks.download(32, input.stream), bits);
}
