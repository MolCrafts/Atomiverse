#pragma once

/// @file flat_pair_list.cuh
/// @brief Flat SoA cluster pair list for the fused search+mask pipeline.
///
/// Unlike CSR ClusterPairList, pairs are stored in a flat array without
/// per-cluster structure. Force kernels iterate one warp per pair.
/// Pairs with zero interaction mask are never stored (pruned at search time).

#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Flat SoA pair list — output of the fused search+mask kernel.
/// @note Non-owning view; lifetime managed by NListBuffers.
template <typename work_t>
struct FlatPairList {
    int* ci; ///< [num_pairs] i-cluster index
    int* cj; ///< [num_pairs] j-cluster index
    uint64_t*
        interaction_mask; ///< [num_pairs] 64-bit mask (only non-zero masks stored)
    const int* num_pairs_device; ///< Device pointer to the current pair count
    int pair_capacity;           ///< Allocated capacity for the flat arrays
};

} // namespace ATV::cuda::md::nlist
