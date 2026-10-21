#pragma once

/// @file cuda/neighbor/nlist_impl.cuh
/// @brief Internal implementation class bridging public API and CUDA kernels.

#include "nlist_data.cuh"
#include "nlist_memory.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Manages all device memory and kernel orchestration.
template <typename work_t>
struct NListBuffers {
    // Hilbert sort temporaries
    DeviceVector<uint32_t> spatial_keys;
    DeviceVector<int> atom_indices;
    DeviceVector<uint32_t> spatial_sorted;
    DeviceVector<int> indices_sorted;
    DeviceVector<char> sort_temp;

    // Cluster SoA
    DeviceVector<work_t> cluster_x, cluster_y, cluster_z;
    DeviceVector<int> original_index;
    DeviceVector<int> cluster_count;

    // AABB
    DeviceVector<work_t> aabb_min_x, aabb_min_y, aabb_min_z;
    DeviceVector<work_t> aabb_max_x, aabb_max_y, aabb_max_z;
    // Six SoA bounds for groups of 32 consecutive Hilbert clusters. Physical
    // AABBs make candidate rejection conservative for arbitrarily wide tiles.
    DeviceVector<real_t> large_bounds;

    // Grid
    DeviceVector<int> cluster_cell_idx; // [num_clusters] cell assignment
    DeviceVector<int> cell_start;       // [ncells + 1]
    DeviceVector<int> cell_clusters;    // [num_clusters]
    DeviceVector<int> histogram;        // temp for building CSR
    DeviceVector<char> scan_temp;

    // Pair list (CSR)
    DeviceVector<int> pair_counts; // [num_clusters] per-cluster pair count
    DeviceVector<int> row_ptr;     // [num_clusters + 1]
    DeviceVector<int> col_idx;     // [num_pairs]
    /// [num_pairs * kMaskWords] OpenMM multi-word interaction masks.
    DeviceVector<uint64_t> interaction_mask;
    DeviceVector<char> prefix_temp;

    // Reference positions for the Verlet-skin rebuild check
    // (snapshot of frame.atoms.x/y/z taken at the end of every full
    // rebuild; consumed by launch_atom_displacement_max each step).
    DeviceVector<work_t> ref_x, ref_y, ref_z;

    // Stencil (device copy)
    DeviceVector<int> stencil_dx, stencil_dy, stencil_dz;

    // Flat pair list (fused pipeline)
    DeviceVector<int> flat_ci, flat_cj;
    DeviceVector<uint64_t> flat_mask;
    DeviceVector<int> flat_pair_counter; // single int, atomic counter

    // Dirty flags written by launch_cluster_displacement; currently
    // accepted-but-unread by smart_rolling_prune (reserved for a
    // future periodic-skip optimization, see cuda-md-review-cleanup-05*).
    DeviceVector<int> dirty_flags;

    // Per-atom Verlet neighbor list (the force path). Column-major:
    // verlet_nbr[k * verlet_stride + i] is atom i's k-th neighbor (global atom
    // index), for k < verlet_cnt[i]. Built once per rebuild from the cluster
    // CSR at rc+skin. verlet_stride is the build-time atom count (stable
    // addressing across set_active_count). verlet_overflow[0] holds the max
    // per-atom count (atomicMax), read back to detect kMaxVerletNeigh overflow.
    DeviceVector<int> verlet_nbr;
    DeviceVector<int> verlet_cnt;
    DeviceVector<int> verlet_overflow;
    int verlet_stride = 0;
    int verlet_max = kMaxVerletNeigh; ///< per-atom column capacity (auto-sized)

    /// Size the per-atom Verlet list: stride = n (build-time atom count),
    /// column capacity = max_neigh (density-derived in NeighborList::build so it
    /// adapts to LJ vs the much denser PE-melt neighbor shell).
    void resize_verlet(int n, int max_neigh) {
        verlet_stride = n;
        verlet_max = max_neigh;
        verlet_cnt.resize(n);
        verlet_nbr.resize(static_cast<std::size_t>(n) * max_neigh);
        verlet_overflow.resize(1);
    }

    // Always-needed per-atom buffers: the reference positions for the
    // Verlet-skin rebuild check. (The cluster-CSR pipeline below is opt-in.)
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

        cluster_cell_idx.resize(n_clusters);

        pair_counts.resize(n_clusters);
        row_ptr.resize(n_clusters + 1);

        cluster_csr_atoms = n;
    }

    // Active atom count the cluster pipeline was last sized for.
    int cluster_csr_atoms = 0;
    bool cluster_ready(int n) const { return cluster_csr_atoms == n; }

    void resize_grid(int ncells) {
        cell_start.resize(ncells + 1);
        cell_clusters.resize(cluster_x.size() / kClusterSize);
        histogram.resize(ncells);
    }

    // ── Direct per-atom cell binning (LAMMPS-style nbin) ──────────────────
    // atom_bincount[cell] = #atoms in cell. Per-slot packed positions
    // atom_bin_pos[cell*bin_cap + slot] = (x, y, z, __int_as_float(atom_index)):
    // xyz feed the Verlet build's distance test as ONE coalesced 16B load per
    // candidate (vs three scattered SoA gathers x[j]/y[j]/z[j] — the
    // L1/TEX-pipe-saturating access that dominated build_verlet_direct_k, ncu),
    // and the .w lane carries the atom index, so the build reads BOTH the
    // candidate's coords and its global index from a single load. This folds the
    // former separate `atom_bins` int array (ncells*bin_cap*4B) into the unused
    // 4th float lane — frees that storage AND removes one bins[] load from the
    // build's inner loop. bin_cap auto-grows on overflow (LAMMPS atoms_per_bin).
    DeviceVector<int> atom_bincount;   // [ncells]
    DeviceVector<float4> atom_bin_pos; // [ncells * bin_cap] (xyz, idx in .w)
    int bin_cap = 0;
    int bin_ncells = 0;
    void resize_bins(int ncells, int cap) {
        bin_cap = cap;
        bin_ncells = ncells;
        atom_bincount.resize(ncells);
        atom_bin_pos.resize(static_cast<std::size_t>(ncells) * cap);
    }

    void resize_pairs(int num_pairs) {
        col_idx.resize(num_pairs);
        interaction_mask.resize(static_cast<std::size_t>(num_pairs)
                                * kMaskWords);
    }

    void resize_flat_pairs(int max_pairs) {
        flat_ci.resize(max_pairs);
        flat_cj.resize(max_pairs);
        // Multi-word masks match CSR layout when csr_to_flat copies masks.
        flat_mask.resize(static_cast<std::size_t>(max_pairs) * kMaskWords);
        flat_pair_counter.resize(1);
    }

    void resize_displacement(int n_clusters) { dirty_flags.resize(n_clusters); }

    ClusterData<work_t> cluster_data(int num_clusters, int num_atoms) {
        return {cluster_x.data(),
                cluster_y.data(),
                cluster_z.data(),
                original_index.data(),
                cluster_count.data(),
                aabb_min_x.data(),
                aabb_min_y.data(),
                aabb_min_z.data(),
                aabb_max_x.data(),
                aabb_max_y.data(),
                aabb_max_z.data(),
                num_clusters,
                num_atoms};
    }

    ClusterPairList<work_t> pair_list(int num_clusters, int num_pairs) {
        return {row_ptr.data(), col_idx.data(), interaction_mask.data(),
                num_pairs, num_clusters};
    }

    FlatPairList<work_t> flat_pair_list() {
        return {flat_ci.data(), flat_cj.data(), flat_mask.data(),
                flat_pair_counter.data(), static_cast<int>(flat_ci.size())};
    }
};

// Kernel launch declarations (defined in .cu files)
// Inverse cell only: the hierarchy does not require a cell grid or stencil.
void launch_update_inverse_cell(const prec_t* box, real_t* inverse,
                                cudaStream_t stream = 0);

template <typename work_t>
void launch_compute_spatial_keys(const real_t* pos_x, const real_t* pos_y,
                                 const real_t* pos_z, int num_atoms,
                                 const real_t* inv_cell, bool periodic,
                                 uint32_t* spatial_keys, int* atom_indices,
                                 cudaStream_t stream = 0);

template <typename work_t>
void launch_assign_clusters_to_cells(
    const real_t* aabb_min_x, const real_t* aabb_min_y,
    const real_t* aabb_min_z, const real_t* aabb_max_x,
    const real_t* aabb_max_y, const real_t* aabb_max_z, int num_clusters,
    const real_t* inv_cell, int dims_x, int dims_y, int dims_z, bool periodic,
    int* cluster_cell_idx, cudaStream_t stream = 0);

// Non-owning physical AABB arrays for the cluster candidate search.
struct ClusterBounds {
    const real_t* min_x;
    const real_t* min_y;
    const real_t* min_z;
    const real_t* max_x;
    const real_t* max_y;
    const real_t* max_z;
};

// Remove fully excluded pairs from existing geometric masks in place. Partial
// special scaling remains a force-time concern; return whether baking applies.
bool launch_bake_csr_exclusions(const int* row_ptr, const int* col_idx,
                                const int* original_index, int num_clusters,
                                int num_pairs, const int* excl_offsets,
                                const int* excl_partners, double special_scale,
                                uint64_t* masks, cudaStream_t stream = 0);

// Build six SoA bounds for groups of 32 consecutive clusters, in build-time
// scratch of length 6*ceil(num_clusters/32). No allocation in these launchers.
void launch_large_cluster_bounds(ClusterBounds bounds, int num_clusters,
                                 real_t* large_bounds, cudaStream_t stream = 0);

// Exact count and fill use the same hierarchical AABB predicate. The one
// prefix scan of counts is already the final CSR: no compaction is required.
template <typename work_t>
void launch_fill_cluster_pairs(ClusterBounds bounds, const real_t* large_bounds,
                               int num_clusters, bool periodic,
                               work_t cutoff_plus_skin_sq, ATV::cuda::Box box,
                               const int* row_ptr, int* col_idx,
                               cudaStream_t stream = 0);

template <typename work_t>
void launch_count_cluster_pairs(ClusterBounds bounds,
                                const real_t* large_bounds, int num_clusters,
                                bool periodic, work_t cutoff_plus_skin_sq,
                                ATV::cuda::Box box, int* pair_counts,
                                cudaStream_t stream = 0);

// Write the CSR sentinel from the last count; the empty CSR has sentinel 0.
void launch_write_row_ptr_sentinel(int* row_ptr, const int* pair_counts,
                                   int num_clusters, cudaStream_t stream = 0);

// CUB sort wrapper (manages temp storage via DeviceVector)
void launch_radix_sort(uint32_t* keys_in, uint32_t* keys_out, int* values_in,
                       int* values_out, int num_items, DeviceVector<char>& temp,
                       cudaStream_t stream = 0);

// CUB prefix sum wrapper (manages temp storage via DeviceVector)
void launch_exclusive_sum(const int* input, int* output, int num_items,
                          DeviceVector<char>& temp, cudaStream_t stream = 0);

// Query the maximum temp-storage bytes that CUB radix sort and exclusive
// sum will require for `n_atoms` items / `n_clusters` items, so the
// caller can pre-size DeviceVector<char> temps in build() and avoid any
// cudaMalloc in the per-step launch() path.
//
// Returns {sort_bytes, scan_bytes_atoms, scan_bytes_clusters}. The
// extra `scan_bytes_clusters` covers `prefix_temp` used over
// `pair_counts[num_clusters]` in `launch_exclusive_sum` from
// `neighbor_list.cu::launch()`.
struct CubTempBytes {
    std::size_t sort_bytes;
    std::size_t scan_bytes_atoms;
    std::size_t scan_bytes_clusters;
};
CubTempBytes query_cub_temp_bytes(int num_atoms, int num_clusters);

// Grid CSR build
void launch_build_grid_csr(const int* cluster_cell_idx, int num_clusters,
                           int ncells, int* cell_start, int* cell_clusters,
                           int* histogram, DeviceVector<char>& scan_temp,
                           cudaStream_t stream = 0);

// --- Fused pipeline kernels (Phase 2-4) ---

// Fused AoS→SoA + Hilbert code computation (replaces separate K1+K2)
template <typename work_t>
void launch_aos_to_soa_spatial(const double* positions_aos, int num_atoms,
                               real_t* x, real_t* y, real_t* z,
                               const real_t* inv_cell, bool periodic,
                               uint32_t* spatial_keys, int* atom_indices,
                               cudaStream_t stream = 0);

// Fused scatter-to-cluster + AABB computation (replaces separate K4+K5)
template <typename work_t>
void launch_scatter_and_aabb(const int* sorted_indices, const real_t* pos_x,
                             const real_t* pos_y, const real_t* pos_z,
                             int num_atoms, int num_clusters, real_t* cluster_x,
                             real_t* cluster_y, real_t* cluster_z,
                             int* original_index, int* cluster_count,
                             real_t* aabb_min_x, real_t* aabb_min_y,
                             real_t* aabb_min_z, real_t* aabb_max_x,
                             real_t* aabb_max_y, real_t* aabb_max_z,
                             cudaStream_t stream = 0);

// CSR→Flat expansion (one warp per cluster, parallel copy)
void launch_csr_to_flat(const int* row_ptr, const int* col_idx,
                        const uint64_t* interaction_mask, int num_clusters,
                        int* flat_ci, int* flat_cj, uint64_t* flat_mask,
                        cudaStream_t stream = 0);

// Per-cluster displacement + dirty flags + rebuild flag
template <typename work_t>
void launch_cluster_displacement(
    const real_t* cluster_x, const real_t* cluster_y, const real_t* cluster_z,
    const real_t* ref_cluster_x, const real_t* ref_cluster_y,
    const real_t* ref_cluster_z, const int* cluster_count, int num_clusters,
    work_t dirty_threshold_sq, real_t* cluster_disp_sq, int* dirty_flags,
    int* rebuild_flag, work_t half_skin_sq, cudaStream_t stream = 0);

// Check current periodic displacements. Writes one overwritten 0/1 flag
// per 256-thread block; caller may use mapped pinned storage and must wait
// for completion before reading. No clear or global atomic is needed.
template <typename work_t>
void launch_atom_displacement_max(const real_t* pos_x, const real_t* pos_y,
                                  const real_t* pos_z, const real_t* ref_x,
                                  const real_t* ref_y, const real_t* ref_z,
                                  int num_atoms, work_t half_skin_sq, Box box,
                                  int* block_flags, cudaStream_t stream = 0);

// Snapshot current per-atom positions into ref_{x,y,z} buffers. Called
// after every full neighbor-list rebuild so the next step's
// `launch_atom_displacement_max` measures motion since the rebuild.
void launch_snapshot_ref_atoms(const real_t* pos_x, const real_t* pos_y,
                               const real_t* pos_z, real_t* ref_x,
                               real_t* ref_y, real_t* ref_z, int num_atoms,
                               cudaStream_t stream = 0);

// Smart rolling prune on flat pair list (skip clean pairs). Templated on
// `bool Ortho` inside the .cu so the triclinic branch dead-eliminates in
// the ortho specialization; the dispatcher takes a cuda::Box by value
// and selects the right instantiation based on `box.ortho`
// (cuda-md-review-cleanup-04-box-by-value-pbc).
template <typename work_t>
void launch_smart_rolling_prune(
    const real_t* cluster_x, const real_t* cluster_y, const real_t* cluster_z,
    const int* flat_ci, const int* flat_cj, const int* num_flat_pairs_device,
    int pair_capacity, const int* dirty_flags, work_t cutoff_sq,
    ATV::cuda::Box box, uint64_t* flat_mask, int prune_step, int prune_period,
    cudaStream_t stream = 0);

// Full CSR interaction masks at the search radius (rc+skin). One warp per
// tile: lane i writes its 32-bit j-mask into masks[p * kMaskWords …]. The
// caller supplies the known CSR entry count. No half-Newton (FULL neighbor).
template <typename work_t>
void launch_compute_csr_masks(const real_t* cluster_x, const real_t* cluster_y,
                              const real_t* cluster_z, const int* row_ptr,
                              const int* col_idx, int num_clusters,
                              int num_pairs, work_t search_sq,
                              ATV::cuda::Box box, uint64_t* masks,
                              cudaStream_t stream = 0);

// Legacy flat-list force mask (optional flat export path only).
template <typename work_t>
void launch_compute_force_mask(const real_t* cluster_x, const real_t* cluster_y,
                               const real_t* cluster_z, const int* flat_ci,
                               const int* flat_cj,
                               const int* num_flat_pairs_device, int pair_count,
                               work_t search_sq, ATV::cuda::Box box,
                               uint64_t* flat_mask, cudaStream_t stream = 0);

// Build the per-atom Verlet list (column-major, stride=num_atoms) from the
// cluster CSR + sorted cluster positions at the search radius (rc+skin). One
// thread per cluster slot; writes neighbor global indices into
// verlet_nbr[k*stride + ai] and the count into verlet_cnt[ai]. atomicMax's the
// max count into overflow[0]. Run once per rebuild. See fused_search.cu.
template <typename work_t>
void launch_build_verlet_list(const real_t* cluster_x, const real_t* cluster_y,
                              const real_t* cluster_z, const int* row_ptr,
                              const int* col_idx, const int* original_index,
                              int num_clusters, int stride, int max_neigh,
                              work_t search_sq, ATV::cuda::Box box,
                              int* verlet_nbr, int* verlet_cnt, int* overflow,
                              cudaStream_t stream = 0);

// --- GPU-side setup (setup.cu) ---

void launch_setup_cell_device(const prec_t* d_box, real_t* d_cell,
                              real_t* d_inv_cell, int* d_dims,
                              int* d_stencil_dx, int* d_stencil_dy,
                              int* d_stencil_dz, int* d_n_stencil,
                              int* d_ncells, work_t cutoff_plus_skin,
                              bool periodic, int max_stencil,
                              cudaStream_t stream = 0);

void launch_setup_cell(const real_t* box_data, real_t* d_cell,
                       real_t* d_inv_cell, int* d_dims, int* d_stencil_dx,
                       int* d_stencil_dy, int* d_stencil_dz, int* d_n_stencil,
                       int* d_ncells, work_t cutoff_plus_skin, bool periodic,
                       int max_stencil, cudaStream_t stream = 0);

void launch_write_int(int* arr, int idx, int val, cudaStream_t stream = 0);

// --- Export to atom pairs (export_pairs.cu) ---

// Launcher dispatches on `box.ortho` for compile-time PBC specialization
// (cuda-md-review-cleanup-04-box-by-value-pbc).
void launch_export_atom_pairs(
    const int* flat_ci, const int* flat_cj, const uint64_t* flat_mask,
    const int* num_flat_device, int n_flat, const int* original_index,
    const real_t* pos_x, const real_t* pos_y, const real_t* pos_z,
    bool periodic, ATV::cuda::Box box, int* out_i, int* out_j, real_t* out_dx,
    real_t* out_dy, real_t* out_dz, real_t* out_dist_sq, int* out_count,
    const int* capacity_ptr, cudaStream_t stream = 0);

} // namespace ATV::cuda::md::nlist
