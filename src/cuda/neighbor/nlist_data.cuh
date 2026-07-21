#pragma once

/// @file nlist_data.cuh
/// @brief GPU data structures for the cluster-based neighbor list.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "flat_pair_list.cuh"

#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Number of atoms per cluster (fixed for warp-friendly sizing).
inline constexpr int kClusterSize = 8;

/// @brief Per-atom Verlet-list column capacity (max neighbors within rc+skin).
///
/// The force path builds a tight per-atom neighbor list once per rebuild and a
/// one-thread-per-atom kernel iterates it — far less warp divergence and no
/// 8×-per-cluster redundancy vs walking the cluster CSR. At LJ-liquid density
/// (rho* ~0.71, rc+skin ~2.6 sigma) the measured max per-atom count within the
/// search shell is ~101 (cutoff-shell neighbors that carry no force but must be
/// listed for the skin interval); 116 covers that with ~15% headroom while
/// keeping the list footprint tight (each column is 4 B/atom of VRAM, so the
/// old loose 128 wasted ~48 B/atom on LJ). This is only the STARTING capacity:
/// a device counter (atomicMax of each atom's count) is read back at every
/// rebuild and launch() grows the buffer and rebuilds if any atom exceeds it,
/// so the much denser PE-melt shell (~400) is reached automatically by
/// self-heal and 116 never silently drops neighbors.
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

/// @brief Cluster pair list in CSR format.
/// @note Non-owning view; lifetime managed by NListBuffers.
/// @note Half-Newton: pairs (ci, cj) with cj >= ci are stored.
///       For diagonal pairs (ci == cj), only upper-triangle interactions
///       (j > i) are set in the mask to avoid double-counting.
///       Force kernels must accumulate both i->j and j->i from each pair.
template <typename work_t>
struct ClusterPairList {
    int* row_ptr; ///< [num_clusters + 1] CSR row pointer
    int* col_idx; ///< [num_pairs] j-cluster index per pair
    uint64_t*
        interaction_mask; ///< [num_pairs] 64-bit: bit[i*8+j]=1 for interaction
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
