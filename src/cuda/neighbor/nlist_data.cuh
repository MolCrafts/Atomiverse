#pragma once

/// @file cuda/neighbor/nlist_data.cuh
/// @brief GPU data structures for the OpenMM-style cluster neighbor list.
///
/// Canonical CUDA MD neighbor representation: kClusterSize=32 tiles, FULL CSR
/// of neighboring clusters, multi-word interaction masks (32×32 bit flags per
/// CSR entry). See cuda-openmm-md chain / CLAUDE.md OpenMM engine rule.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "flat_pair_list.cuh"

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

/// @brief uint32 rows per mask (= kClusterSize). Helper for host/device.
inline constexpr int kMaskU32PerPair = kClusterSize;

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

/// @brief Legacy per-atom Verlet column capacity (multi-GPU transitional only).
///
/// Single-GPU force does not use vl_*. multi_gpu may still allocate until
/// cuda-openmm-md-04 retires it. Auto-grows on overflow in rebuild.
inline constexpr int kMaxVerletNeigh = 116;

/// @brief Per-cluster atom data in SoA layout.
/// @note Non-owning view; lifetime managed by NListBuffers.
template <typename work_t>
struct ClusterData {
    work_t* x;           ///< [num_clusters * kClusterSize]
    work_t* y;           ///< [num_clusters * kClusterSize]
    work_t* z;           ///< [num_clusters * kClusterSize]
    int* original_index; ///< [num_clusters * kClusterSize] map to input order
    int* cluster_count;  ///< [num_clusters] actual atom count (<=kClusterSize)
    work_t* aabb_min_x;  ///< [num_clusters]
    work_t* aabb_min_y;  ///< [num_clusters]
    work_t* aabb_min_z;  ///< [num_clusters]
    work_t* aabb_max_x;  ///< [num_clusters]
    work_t* aabb_max_y;  ///< [num_clusters]
    work_t* aabb_max_z;  ///< [num_clusters]
    int num_clusters;
    int num_atoms;
};

/// @brief Cluster pair list in CSR format (OpenMM-style tile neighbors).
/// @note Non-owning view; lifetime managed by NListBuffers.
/// @note FULL neighbor list: every neighboring cluster appears for both
///       endpoints (cj < ci, cj == ci, and cj > ci). Atom-centric / tile force
///       kernels own each atom's force and walk the full set; energy must be
///       de-duplicated (e.g. ×0.5) if both endpoints contribute.
template <typename work_t>
struct ClusterPairList {
    int* row_ptr; ///< [num_clusters + 1] CSR row pointer
    int* col_idx; ///< [num_pairs] j-cluster index per pair
    /// [num_pairs * kMaskWords] multi-word interaction masks (see kMaskWords).
    uint64_t* interaction_mask;
    int num_pairs;
    int num_clusters;
};

/// @brief Spatial grid of clusters for stencil-based pair search.
/// @note Non-owning view; lifetime managed by NListBuffers.
template <typename work_t>
struct ClusterGrid {
    int* cell_start;    ///< [ncells + 1] CSR: cluster start per cell
    int* cell_clusters; ///< [num_clusters] cluster indices sorted by cell
    int dims_x, dims_y, dims_z;  ///< Grid dimensions
    work_t cell_matrix[9];       ///< Cell matrix (row-major)
    work_t inv_cell[9];          ///< Inverse cell matrix
    const real_t* d_cell_matrix; ///< Device pointer to row-major cell matrix
    const real_t* d_inv_cell;    ///< Device pointer to inverse cell matrix
};

} // namespace ATV::cuda::md::nlist
