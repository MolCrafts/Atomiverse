/// @file cuda/potentials/internal/packed_tiles.cuh
/// @brief Private, caller-preallocated packing of baked cluster masks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace ATV::cuda {
struct PairList;
namespace internal {

// Each output tile keeps one 32-atom I cluster and 32 arbitrary sorted J
// slots. Masks remain row-oriented: bit j of mask[32*tile+i] selects J lane j.
// Partial tiles have J=-1 and zero mask bits in their padding columns.
struct PackedTileWorkspace {
    int* counts = nullptr;          // num_clusters+1, last count is zero.
    int* offsets = nullptr;         // exclusive scan of counts.
    int* host_count = nullptr;      // caller-owned pinned scalar readback.
    int* ci = nullptr;              // capacity tiles.
    int* j = nullptr;               // capacity*32 sorted atom slots.
    std::uint32_t* masks = nullptr; // capacity*32 rows.
    void* scan_temp = nullptr;
    std::size_t scan_bytes = 0;
    int capacity = 0;
};

std::size_t packed_tiles_scan_bytes(int num_clusters);

// Consumes already-baked CSR masks, removes lower/diagonal duplicates, and
// packs nonempty J columns across CSR entries in each I row. Returns the
// active tile count after one checked scalar readback on this stream.
// No allocation, geometric approximations, or per-atom fallback.
int pack_tiles(const PairList& source, const PackedTileWorkspace& work,
               cudaStream_t stream);

} // namespace internal
} // namespace ATV::cuda
