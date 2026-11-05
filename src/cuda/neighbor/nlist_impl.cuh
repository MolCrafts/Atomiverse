#pragma once

/// @file cuda/neighbor/nlist_impl.cuh
/// @brief Internal implementation class bridging public API and CUDA kernels.

#include "nlist_data.cuh"
#include "nlist_memory.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Manages all device memory and kernel orchestration.
///
/// Coordinate copies (cluster SoA, skin-check reference) keep the stored
/// coordinate type `pos_t`; bounds and other search scratch use `T`, the
/// force-path scratch type (`force_t`).
template <typename T>
struct NListBuffers {
    // Hilbert sort temporaries
    DeviceVector<uint32_t> spatial_keys;
    DeviceVector<int> atom_indices;
    DeviceVector<uint32_t> spatial_sorted;
    DeviceVector<int> indices_sorted;
    DeviceVector<char> sort_temp;

    // Cluster SoA
    DeviceVector<pos_t> cluster_x, cluster_y, cluster_z;
    DeviceVector<int> original_index;
    DeviceVector<int> cluster_count;

    // AABB
    DeviceVector<T> aabb_min_x, aabb_min_y, aabb_min_z;
    DeviceVector<T> aabb_max_x, aabb_max_y, aabb_max_z;
    // Six SoA bounds for groups of 32 consecutive Hilbert clusters. Physical
    // AABBs make candidate rejection conservative for arbitrarily wide tiles.
    DeviceVector<T> large_bounds;

    // Pair list (CSR)
    DeviceVector<int> pair_counts; // [num_clusters] per-cluster pair count
    DeviceVector<int> row_ptr;     // [num_clusters + 1]
    DeviceVector<int> col_idx;     // [num_pairs]
    /// [num_pairs * kMaskWords] OpenMM multi-word interaction masks.
    DeviceVector<uint64_t> interaction_mask;
    DeviceVector<char> prefix_temp;

    // Reference positions for the Verlet-skin rebuild check
    // (snapshot of frame.atoms.x/y/z taken at the end of every full
    // rebuild; consumed by the skin gate each step).
    DeviceVector<pos_t> ref_x, ref_y, ref_z;

    void resize_atoms(int n) {
        ref_x.resize(n);
        ref_y.resize(n);
        ref_z.resize(n);
    }

    // The Hilbert/cluster/AABB/CSR pipeline is the CUDA MD force source of
    // truth. Allocate at build-time capacity; set_active_count can reuse
    // these buffers for a smaller owned+ghost atom set.
    void resize_cluster_csr(int n, int n_clusters) {
        spatial_keys.resize(n);
        atom_indices.resize(n);
        spatial_sorted.resize(n);
        indices_sorted.resize(n);

        int padded = n_clusters * kClusterSize;
        cluster_x.resize(padded);
        cluster_y.resize(padded);
        cluster_z.resize(padded);
        original_index.resize(padded);
        cluster_count.resize(n_clusters);

        aabb_min_x.resize(n_clusters);
        aabb_min_y.resize(n_clusters);
        aabb_min_z.resize(n_clusters);
        aabb_max_x.resize(n_clusters);
        aabb_max_y.resize(n_clusters);
        aabb_max_z.resize(n_clusters);
        large_bounds.resize(6 * ((n_clusters + 31) / 32));

        pair_counts.resize(n_clusters);
        row_ptr.resize(n_clusters + 1);
    }

    void resize_pairs(int num_pairs) {
        col_idx.resize(num_pairs);
        interaction_mask.resize(static_cast<std::size_t>(num_pairs)
                                * kMaskWords);
    }
};

/// @brief Device control words of one neighbor list, written only by kernels.
///
/// The skin gate accumulates `motion`, consumes `force` (set by the host
/// stream before a forced rebuild), and publishes `rebuilt` for consumers
/// that keep rebuild-time state. `overflow` is the CSR capacity verdict of
/// the current rebuild; `done` is the last-block counter; `rebuilds` counts
/// rebuilds for introspection after the run.
struct Control {
    int motion = 0;
    int force = 0;
    int rebuilt = 0;
    int overflow = 0;
    unsigned done = 0;
    int rebuilds = 0;
    unsigned long long generation = 0; ///< Unique id of the published masks.
};

void launch_update_inverse_cell(const geom_t* box, force_t* inverse,
                                cudaStream_t stream = 0);

template <typename T>
void launch_compute_spatial_keys(const pos_t* pos_x, const pos_t* pos_y,
                                 const pos_t* pos_z, int num_atoms,
                                 const force_t* inv_cell, bool periodic,
                                 uint32_t* spatial_keys, int* atom_indices,
                                 cudaStream_t stream = 0);

struct ClusterBounds {
    const force_t* min_x;
    const force_t* min_y;
    const force_t* min_z;
    const force_t* max_x;
    const force_t* max_y;
    const force_t* max_z;
};

bool launch_bake_csr_exclusions(const int* row_ptr, const int* col_idx,
                                const int* original_index, int num_clusters,
                                const int* excl_offsets,
                                const int* excl_partners, double special_scale,
                                uint64_t* masks, cudaStream_t stream = 0);

void launch_large_cluster_bounds(ClusterBounds bounds, int num_clusters,
                                 force_t* large_bounds,
                                 cudaStream_t stream = 0);

/// Fills each CSR row. When `overflow` is set the CSR is emptied instead
/// (every row offset zero), so no consumer reads past the reserve.
template <typename T>
void launch_fill_cluster_pairs(ClusterBounds bounds,
                               const force_t* large_bounds, int num_clusters,
                               bool periodic, T cutoff_plus_skin_sq,
                               ATV::cuda::Box box, int* row_ptr, int* col_idx,
                               const int* overflow, cudaStream_t stream = 0);

template <typename T>
void launch_count_cluster_pairs(ClusterBounds bounds,
                                const force_t* large_bounds, int num_clusters,
                                bool periodic, T cutoff_plus_skin_sq,
                                ATV::cuda::Box box, int* pair_counts,
                                cudaStream_t stream = 0);

/// Writes the CSR sentinel row_ptr[num_clusters] and the capacity verdict:
/// `*overflow = total > capacity`, raising `ForceFault::kCapacity` with the
/// needed entry count. `fault` may be null (build-time measurement).
void launch_close_csr(int* row_ptr, const int* pair_counts, int num_clusters,
                      int capacity, int* overflow, ForceFault* fault,
                      cudaStream_t stream = 0);

void launch_radix_sort(uint32_t* keys_in, uint32_t* keys_out, int* values_in,
                       int* values_out, int num_items, DeviceVector<char>& temp,
                       cudaStream_t stream = 0);

void launch_exclusive_sum(const int* input, int* output, int num_items,
                          DeviceVector<char>& temp, cudaStream_t stream = 0);

// Query the maximum temp-storage bytes that CUB radix sort and exclusive
// sum will require for `n_atoms` items / `n_clusters` items, so the
// caller can pre-size DeviceVector<char> temps in build() and avoid any
// cudaMalloc in the per-step launch() path.
//
// Returns {sort_bytes, scan_bytes_atoms, scan_bytes_clusters}. The
// extra `scan_bytes_clusters` covers `prefix_temp` used over
// `pair_counts[num_clusters]` in `launch_exclusive_sum`.
struct CubTempBytes {
    std::size_t sort_bytes;
    std::size_t scan_bytes_atoms;
    std::size_t scan_bytes_clusters;
};
CubTempBytes query_cub_temp_bytes(int num_atoms, int num_clusters);

// Fused scatter-to-cluster + AABB computation
template <typename T>
void launch_scatter_and_aabb(const int* sorted_indices, const pos_t* pos_x,
                             const pos_t* pos_y, const pos_t* pos_z,
                             int num_atoms, int num_clusters, pos_t* cluster_x,
                             pos_t* cluster_y, pos_t* cluster_z,
                             int* original_index, int* cluster_count,
                             force_t* aabb_min_x, force_t* aabb_min_y,
                             force_t* aabb_min_z, force_t* aabb_max_x,
                             force_t* aabb_max_y, force_t* aabb_max_z,
                             cudaStream_t stream = 0);

/// Current-step skin gate. Every block reduces "an atom moved more than the
/// LAMMPS check_distance trigger (half the skin less the reference-cell
/// corner motion), or the cell motion alone consumed the skin"; the last block
/// folds in the forced flag, publishes
/// `control->rebuilt`, advances the rebuild count and mask generation, and
/// sets the conditional rebuild handle. Must run inside the graph that owns
/// `handle`.
template <typename T>
void launch_skin_gate(const pos_t* pos_x, const pos_t* pos_y,
                      const pos_t* pos_z, const pos_t* ref_x,
                      const pos_t* ref_y, const pos_t* ref_z, int num_atoms,
                      double skin, Box box, const Box* reference_box,
                      Control* control, cudaGraphConditionalHandle handle,
                      cudaStream_t stream = 0);

/// Advances the mask generation without a skin gate (multi-GPU rebuild).
void launch_publish_generation(Control* control, cudaStream_t stream = 0);

void launch_snapshot_ref_atoms(const pos_t* pos_x, const pos_t* pos_y,
                               const pos_t* pos_z, pos_t* ref_x, pos_t* ref_y,
                               pos_t* ref_z, int num_atoms,
                               cudaStream_t stream = 0);

/// Interaction masks at the search radius for every CSR entry. The entry
/// count is read on the device (`row_ptr[num_clusters]`); the grid is sized
/// from the build-time `pair_capacity` and strides over the live entries.
template <typename T>
void launch_compute_csr_masks(const pos_t* cluster_x, const pos_t* cluster_y,
                              const pos_t* cluster_z, const int* row_ptr,
                              const int* col_idx, int num_clusters,
                              int pair_capacity, T search_sq,
                              ATV::cuda::Box box, uint64_t* masks,
                              cudaStream_t stream = 0);

} // namespace ATV::cuda::md::nlist
