#pragma once

/// @file cuda/neighbor/nlist_data.cuh
/// @brief GPU data structures for the OpenMM-style cluster neighbor list.
///
/// Canonical CUDA MD neighbor representation: kClusterSize=32 tiles, FULL CSR
/// of neighboring clusters, multi-word interaction masks (32×32 bit flags per
/// CSR entry). See cuda-openmm-md chain / docs/developer/project-contract.md OpenMM engine rule.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Atoms per cluster (OpenMM Eastman–Pande tile width).
/// Locked equal to public `cuda::kOpenmmClusterSize`.
inline constexpr int kClusterSize = 32;
static_assert(kClusterSize == 32, "OpenMM cluster size is 32");

/// @brief uint64 words of interaction mask per CSR cluster-pair.
///
/// Layout (locked for force kernels, phase 02–03):
///   For CSR entry `p` (cluster pair ci→cj), bits live in
///   `mask + p * kMaskWords` … `+ kMaskWords - 1`.
///   Bit for (i_slot, j_slot) is at linear index `i_slot * kClusterSize + j_slot`
///   (row-major over i then j). Equivalently, as `uint32_t[kClusterSize]`:
///   word `i_slot` holds the 32-bit j-mask for that i atom.
///   Pad slots (orig_index < 0) should have zero rows / fail distance tests.
inline constexpr int kMaskWords = (kClusterSize * kClusterSize + 63) / 64; // 16
static_assert(kMaskWords == 16, "CS=32 ⇒ 16×uint64 mask words");

/// @brief Read the 32-bit j-interaction flags for i_slot of CSR pair p.
__host__ __device__ inline std::uint32_t mask_jbits(const std::uint64_t* mask,
                                                    int pair_idx, int i_slot) {
    const auto* rows = reinterpret_cast<const std::uint32_t*>(
        mask + static_cast<std::size_t>(pair_idx) * kMaskWords);
    return rows[i_slot];
}

/// @brief Write the 32-bit j-interaction flags for i_slot of CSR pair p.
__host__ __device__ inline void mask_set_jbits(std::uint64_t* mask,
                                               int pair_idx, int i_slot,
                                               std::uint32_t bits) {
    auto* rows = reinterpret_cast<std::uint32_t*>(
        mask + static_cast<std::size_t>(pair_idx) * kMaskWords);
    rows[i_slot] = bits;
}

} // namespace ATV::cuda::md::nlist
