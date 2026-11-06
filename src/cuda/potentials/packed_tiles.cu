/// @file cuda/potentials/packed_tiles.cu
/// @brief Packed-J tiles from the baked cluster CSR masks (the rebuild-time
///        half of the cluster-tile engine, `internal/cluster_tile.cuh`).
/// Algorithm reference: OpenMM findInteractingBlocks.cu, commit
/// 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e; implementation ATV-native.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "internal/packed_tiles.cuh"

namespace ATV::cuda {
namespace {
constexpr int kPackBlock = 128;

// OpenMM packs only J atoms that interact with an I cluster, crossing the
// original J-cluster boundaries. Here the exact baked masks are the input;
// no geometry or exclusion decision is repeated by the packer.
__device__ unsigned upper_mask(const PairList& source, int p, int ci,
                               int lane) {
    const int cj = source.cl_col_idx[p];
    unsigned bits = openmm_mask_jbits(source.cl_mask, p, lane);
    if (cj < ci)
        return 0;
    if (cj == ci)
        bits &= lane == 31 ? 0u : (0xffffffffu << (lane + 1));
    return bits;
}

__global__ void count_packed_tiles(PairList source, int* counts, int* valid) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci > source.num_clusters)
        return;
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int columns = 0, entries = 0;
    if (ci < source.num_clusters)
        for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1];
             ++p) {
            const unsigned used = cg::reduce(
                warp, upper_mask(source, p, ci, lane), cg::bit_or<unsigned>{});
            columns += __popc(used);
            entries += used != 0;
        }
    if (lane == 0) {
        counts[ci] = (columns + 31) / 32;
        if (valid != nullptr && entries)
            atomicAdd(valid, entries);
    }
}

__global__ void fill_packed_tiles(PairList source,
                                  internal::PackedTileWorkspace out,
                                  ForceFault* fault) {
    const int ci = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (ci >= source.num_clusters)
        return;
    // A packed tile count never exceeds the CSR entry count, which the
    // neighbor list already bounds by the same reserve; guard regardless.
    const int total = out.offsets[source.num_clusters];
    if (total > out.capacity) {
        if (ci == 0 && lane == 0)
            raise_capacity_fault(fault, total);
        return;
    }
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tile = out.offsets[ci], column = 0;
    unsigned packed_bits = 0;
    for (int p = source.cl_row_ptr[ci]; p < source.cl_row_ptr[ci + 1]; ++p) {
        const unsigned row = upper_mask(source, p, ci, lane);
        unsigned columns = cg::reduce(warp, row, cg::bit_or<unsigned>{});
        while (columns) {
            const int j = __ffs(columns) - 1;
            columns &= columns - 1;
            if (row & (1u << j))
                packed_bits |= 1u << column;
            if (lane == column)
                out.j[static_cast<std::size_t>(tile) * 32 + lane] =
                    source.cl_col_idx[p] * 32 + j;
            if (++column == 32) {
                out.masks[static_cast<std::size_t>(tile) * 32 + lane] =
                    packed_bits;
                if (lane == 0)
                    out.ci[tile] = ci;
                ++tile;
                column = 0;
                packed_bits = 0;
            }
        }
    }
    if (column) {
        const std::size_t slot = static_cast<std::size_t>(tile) * 32 + lane;
        if (lane >= column)
            out.j[slot] = -1;
        out.masks[slot] = packed_bits;
        if (lane == 0)
            out.ci[tile] = ci;
    }
}

} // namespace

std::size_t internal::packed_tiles_scan_bytes(int num_clusters) {
    std::size_t bytes = 0;
    ATV_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr), num_clusters + 1));
    return bytes;
}

void internal::pack_tiles(const PairList& source,
                          const PackedTileWorkspace& work, ForceFault* fault,
                          cudaStream_t stream) {
    if (work.valid != nullptr)
        ATV_CUDA_CHECK(cudaMemsetAsync(work.valid, 0, sizeof(int), stream));
    count_packed_tiles<<<(source.num_clusters + 4) / 4, kPackBlock, 0,
                         stream>>>(source, work.counts, work.valid);
    ATV_CUDA_CHECK(cudaGetLastError());
    std::size_t bytes = work.scan_bytes;
    ATV_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        work.scan_temp, bytes, work.counts, work.offsets,
        source.num_clusters + 1, stream));
    if (source.num_clusters > 0) {
        fill_packed_tiles<<<(source.num_clusters + 3) / 4, kPackBlock, 0,
                            stream>>>(source, work, fault);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ATV::cuda
