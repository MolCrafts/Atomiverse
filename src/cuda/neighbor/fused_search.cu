/// @file cuda/neighbor/fused_search.cu
/// @brief CSR↔Flat conversion, displacement tracking, and flat-mask refresh kernels.
///
/// Supports the GPU-Resident Fused Neighbor List (GRFNL) pipeline.
/// The build uses the proven legacy search pipeline (coarse count → fill →
/// compact) for full warp utilization, then expands CSR to flat pair format
/// for the smart prune path.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cstdint>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Kernel: CSR → Flat expansion (one warp per cluster, parallel copy)
// ---------------------------------------------------------------------------

__global__ void csr_to_flat(const int* __restrict__ row_ptr,
                            const int* __restrict__ col_idx,
                            const uint64_t* __restrict__ mask_in,
                            int num_clusters, int* __restrict__ flat_ci,
                            int* __restrict__ flat_cj,
                            uint64_t* __restrict__ flat_mask) {

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_clusters)
        return;

    int ci = warp_id;
    int begin = row_ptr[ci];
    int end = row_ptr[ci + 1];

    for (int p = begin + lane; p < end; p += 32) {
        flat_ci[p] = ci;
        flat_cj[p] = col_idx[p];
        // Multi-word mask: copy kMaskWords uint64s per pair.
        const uint64_t* src =
            mask_in + static_cast<std::size_t>(p) * kMaskWords;
        uint64_t* dst = flat_mask + static_cast<std::size_t>(p) * kMaskWords;
#pragma unroll
        for (int w = 0; w < kMaskWords; ++w)
            dst[w] = src[w];
    }
}

void launch_csr_to_flat(const int* row_ptr, const int* col_idx,
                        const uint64_t* interaction_mask, int num_clusters,
                        int* flat_ci, int* flat_cj, uint64_t* flat_mask,
                        cudaStream_t stream) {
    if (num_clusters == 0)
        return;

    int threads_needed = num_clusters * 32;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;

    csr_to_flat<<<grid, block, 0, stream>>>(row_ptr, col_idx, interaction_mask,
                                            num_clusters, flat_ci, flat_cj,
                                            flat_mask);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Per-cluster displacement + dirty flags + rebuild flag
//
// Retained as a cluster-granularity reference; the live Verlet-skin
// gate uses `launch_atom_displacement_max` (defined below) against
// `frame.atoms.x/y/z` vs `NListBuffers::ref_x/y/z` instead. The
// cluster-granular path is kept for future use by a periodic-skip
// smart-prune optimization.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void
cluster_displacement(const pos_t* __restrict__ cluster_x,
                     const pos_t* __restrict__ cluster_y,
                     const pos_t* __restrict__ cluster_z,
                     const pos_t* __restrict__ ref_cluster_x,
                     const pos_t* __restrict__ ref_cluster_y,
                     const pos_t* __restrict__ ref_cluster_z,
                     const int* __restrict__ cluster_count, int num_clusters,
                     T dirty_threshold_sq, force_t* __restrict__ disp_sq_out,
                     int* __restrict__ dirty_flags,
                     int* __restrict__ rebuild_flag, T half_skin_sq) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % kClusterSize;
    int ci = tid / kClusterSize;

    if (ci >= num_clusters)
        return;

    int count = cluster_count[ci];
    int idx = ci * kClusterSize + lane;

    T dx = (lane < count) ? (cluster_x[idx] - ref_cluster_x[idx]) : T(0);
    T dy = (lane < count) ? (cluster_y[idx] - ref_cluster_y[idx]) : T(0);
    T dz = (lane < count) ? (cluster_z[idx] - ref_cluster_z[idx]) : T(0);
    T d2 = dx * dx + dy * dy + dz * dz;

    // Warp-level max reduction (OpenMM CS=32 ⇒ full warp).
    static_assert(kClusterSize == 32, "cluster_displacement assumes CS=32");
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    d2 = cg::reduce(warp, d2, cg::greater<T>{});

    if (lane == 0) {
        disp_sq_out[ci] = d2;
        dirty_flags[ci] = (d2 > dirty_threshold_sq) ? 1 : 0;
        if (d2 > half_skin_sq) {
            atomicMax(rebuild_flag, 1);
        }
    }
}

template <typename T>
void launch_cluster_displacement(
    const pos_t* cluster_x, const pos_t* cluster_y, const pos_t* cluster_z,
    const pos_t* ref_cluster_x, const pos_t* ref_cluster_y,
    const pos_t* ref_cluster_z, const int* cluster_count, int num_clusters,
    T dirty_threshold_sq, force_t* cluster_disp_sq, int* dirty_flags,
    int* rebuild_flag, T half_skin_sq, cudaStream_t stream) {
    if (num_clusters == 0)
        return;

    ATV_CUDA_CHECK(cudaMemsetAsync(rebuild_flag, 0, sizeof(int), stream));

    int threads_needed = num_clusters * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;

    cluster_displacement<T><<<grid, block, 0, stream>>>(
        cluster_x, cluster_y, cluster_z, ref_cluster_x, ref_cluster_y,
        ref_cluster_z, cluster_count, num_clusters, dirty_threshold_sq,
        cluster_disp_sq, dirty_flags, rebuild_flag, half_skin_sq);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Refresh atom-pair masks on the flat pair list.
//
// Correctness note:
// The mask must be recomputed from current coordinates instead of being
// monotonically intersected with the previous mask. Otherwise a pair that
// leaves the cutoff and later re-enters before the next rebuild can be lost
// permanently.
// ---------------------------------------------------------------------------

// Flat-list mask refresh: one warp (CS=32) per flat pair. Writes multi-word
// masks (kMaskWords uint64 / pair). HalfNewton trims the diagonal for export.
//
// Templated on `bool Ortho` so the triclinic branch is dead-eliminated on
// orthorhombic boxes (cuda-md-review-cleanup-04-box-by-value-pbc).
template <bool Ortho, bool HalfNewton, typename T>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void smart_rolling_prune(
    const pos_t* __restrict__ cluster_x, const pos_t* __restrict__ cluster_y,
    const pos_t* __restrict__ cluster_z, const int* __restrict__ flat_ci,
    const int* __restrict__ flat_cj,
    const int* __restrict__ num_flat_pairs_device,
    const int* __restrict__ dirty_flags, T cutoff_sq, cuda::Box box_arg,
    uint64_t* __restrict__ flat_mask, int prune_step, int prune_period) {
    const auto box = box_arg.current().as<T>();
    (void)dirty_flags;
    (void)prune_step;
    (void)prune_period;

    static_assert(kClusterSize == 32, "flat prune assumes OpenMM CS=32");

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = tid / kClusterSize;
    int i_lane = tid % kClusterSize;
    int num_flat_pairs = *num_flat_pairs_device;

    if (pair_idx >= num_flat_pairs)
        return;

    int ci = flat_ci[pair_idx];
    int cj = flat_cj[pair_idx];

    T xi = cluster_x[ci * kClusterSize + i_lane];
    T yi = cluster_y[ci * kClusterSize + i_lane];
    T zi = cluster_z[ci * kClusterSize + i_lane];

    // Full warp owns this pair: each lane holds one j-slot of cj in registers.
    T jx = cluster_x[cj * kClusterSize + i_lane];
    T jy = cluster_y[cj * kClusterSize + i_lane];
    T jz = cluster_z[cj * kClusterSize + i_lane];

    std::uint32_t my_bits = 0;
    const unsigned submask = 0xffffffffu;

    for (int j = 0; j < kClusterSize; ++j) {
        T jx_j = __shfl_sync(submask, jx, j);
        T jy_j = __shfl_sync(submask, jy, j);
        T jz_j = __shfl_sync(submask, jz, j);

        T dx = jx_j - xi;
        T dy = jy_j - yi;
        T dz = jz_j - zi;

        if constexpr (Ortho) {
            dx -= T(box.cell[0]) * nearbyint(dx * T(box.inv_l[0]));
            dy -= T(box.cell[4]) * nearbyint(dy * T(box.inv_l[1]));
            dz -= T(box.cell[8]) * nearbyint(dz * T(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }

        T r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < cutoff_sq)
            my_bits |= (1u << j);
    }

    // Half-Newton: export path only keeps j > i on diagonal tiles.
    if constexpr (HalfNewton) {
        if (ci == cj) {
            std::uint32_t upper =
                (i_lane >= 31) ? 0u : (0xffffffffu << (i_lane + 1));
            my_bits &= upper;
        }
    }

    mask_set_jbits(flat_mask, pair_idx, i_lane, my_bits);
}

template <typename T>
void launch_smart_rolling_prune(const pos_t* cluster_x, const pos_t* cluster_y,
                                const pos_t* cluster_z, const int* flat_ci,
                                const int* flat_cj,
                                const int* num_flat_pairs_device,
                                int pair_capacity, const int* dirty_flags,
                                T cutoff_sq, cuda::Box box, uint64_t* flat_mask,
                                int prune_step, int prune_period,
                                cudaStream_t stream) {
    if (pair_capacity == 0)
        return;

    // 64-bit: at multi-million atoms the launch count (pair_capacity here
    // receives the actual flat-pair count or the capacity) times kClusterSize
    // exceeds INT_MAX (e.g. 425M pairs * 8 = 3.4e9), which wrapped negative and
    // produced a bogus ~4.29e9 grid dimension. The grid itself stays small.
    long long threads_needed =
        static_cast<long long>(pair_capacity) * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = static_cast<int>((threads_needed + block - 1) / block);

    // Dispatch the kernel templated on `box.ortho` so the dead branch
    // is compile-time eliminated.
    if (box.ortho) {
        smart_rolling_prune<true, true, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, dirty_flags, cutoff_sq, box, flat_mask,
            prune_step, prune_period);
    } else {
        smart_rolling_prune<false, true, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, dirty_flags, cutoff_sq, box, flat_mask,
            prune_step, prune_period);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CSR interaction masks at SEARCH radius (rc+skin) — production OpenMM path.
// Geometry assigns one warp per CSR tile; exclusion baking then clears
// exception bits in the existing multi-word masks.
// ---------------------------------------------------------------------------
__global__ void bake_csr_exclusions_k(const int* __restrict__ row_ptr,
                                      const int* __restrict__ col_idx,
                                      const int* __restrict__ original_index,
                                      int num_clusters,
                                      const int* __restrict__ offsets,
                                      const int* __restrict__ partners,
                                      uint64_t* __restrict__ masks) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int ci = slot / kClusterSize, si = slot % kClusterSize;
    if (ci >= num_clusters)
        return;
    const int ai = original_index[slot];
    for (int p = row_ptr[ci]; p < row_ptr[ci + 1]; ++p) {
        std::uint32_t bits = mask_jbits(masks, p, si);
        std::uint32_t retained = 0;
        if (ai >= 0) {
            const int begin = offsets[ai], end = offsets[ai + 1];
            const int cj = col_idx[p];
            while (bits) {
                const int sj = __ffs(static_cast<int>(bits)) - 1;
                bits &= bits - 1;
                const int aj = original_index[cj * kClusterSize + sj];
                if (aj < 0)
                    continue;
                if (!thrust::binary_search(thrust::seq, partners + begin,
                                           partners + end, aj))
                    retained |= (1u << sj);
            }
        }
        mask_set_jbits(masks, p, si, retained);
    }
}

bool launch_bake_csr_exclusions(const int* row_ptr, const int* col_idx,
                                const int* original_index, int num_clusters,
                                const int* excl_offsets,
                                const int* excl_partners, double special_scale,
                                uint64_t* masks, cudaStream_t stream) {
    if (excl_offsets == nullptr || special_scale != 0.0)
        return false;
    if (num_clusters > 0) {
        const int grid = ATV::cuda::grid_size(num_clusters * kClusterSize);
        bake_csr_exclusions_k<<<grid, ATV::cuda::kDefaultBlock, 0, stream>>>(
            row_ptr, col_idx, original_index, num_clusters, excl_offsets,
            excl_partners, masks);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

// A warp owns one CSR tile, matching the OpenMM warp/tile work partition
// (platforms/cuda/src/kernels/nonbonded.cu, SHA
// 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e). The geometric predicate and
// mask layout remain ATV-native; no OpenMM source is copied.
template <bool Ortho, typename T>
__global__ __launch_bounds__(128) void compute_csr_masks_k(
    const pos_t* __restrict__ cluster_x, const pos_t* __restrict__ cluster_y,
    const pos_t* __restrict__ cluster_z, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, int num_clusters, int num_pairs,
    T search_sq, cuda::Box box_arg, uint64_t* __restrict__ masks) {
    const auto box = box_arg.current().as<T>();
    static_assert(kClusterSize == 32, "CSR mask kernel assumes CS=32");
    const int p =
        blockIdx.x * (blockDim.x / kClusterSize) + threadIdx.x / kClusterSize;
    const int lane = threadIdx.x % kClusterSize;
    if (p >= num_pairs)
        return; // Uniform within the warp; no block collective follows.
    int ci = 0;
    if (lane == 0) {
        ci = static_cast<int>(thrust::upper_bound(thrust::seq, row_ptr + 1,
                                                  row_ptr + num_clusters + 1, p)
                              - (row_ptr + 1));
    }
    ci = __shfl_sync(0xffffffffu, ci, 0);
    const int cj = col_idx[p];
    const T xi = cluster_x[ci * kClusterSize + lane];
    const T yi = cluster_y[ci * kClusterSize + lane];
    const T zi = cluster_z[ci * kClusterSize + lane];
    // Coalesced J-block loads are reused through warp exchange for all 32
    // rows; no warp serializes the remaining tiles in the same CSR row.
    const T xj = cluster_x[cj * kClusterSize + lane];
    const T yj = cluster_y[cj * kClusterSize + lane];
    const T zj = cluster_z[cj * kClusterSize + lane];
    std::uint32_t bits = 0;
#pragma unroll 4
    for (int j = 0; j < kClusterSize; ++j) {
        T dx = __shfl_sync(0xffffffffu, xj, j) - xi;
        T dy = __shfl_sync(0xffffffffu, yj, j) - yi;
        T dz = __shfl_sync(0xffffffffu, zj, j) - zi;
        if constexpr (Ortho) {
            dx -= T(box.cell[0]) * nearbyint(dx * T(box.inv_l[0]));
            dy -= T(box.cell[4]) * nearbyint(dy * T(box.inv_l[1]));
            dz -= T(box.cell[8]) * nearbyint(dz * T(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }
        const T r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < search_sq && r2 > T(0))
            bits |= (1u << j);
    }
    // Exclude the same atom on diagonal tiles exactly as the row kernel did.
    if (ci == cj)
        bits &= ~(1u << lane);
    mask_set_jbits(masks, p, lane, bits);
}

template <typename T>
void launch_compute_csr_masks(const pos_t* cluster_x, const pos_t* cluster_y,
                              const pos_t* cluster_z, const int* row_ptr,
                              const int* col_idx, int num_clusters,
                              int num_pairs, T search_sq, cuda::Box box,
                              uint64_t* masks, cudaStream_t stream) {
    if (num_clusters <= 0 || num_pairs <= 0)
        return;
    constexpr int threads = 128;
    constexpr int tiles_per_block = threads / kClusterSize;
    const int grid = (num_pairs - 1) / tiles_per_block + 1;
    if (box.ortho) {
        compute_csr_masks_k<true, T><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            num_pairs, search_sq, box, masks);
    } else {
        compute_csr_masks_k<false, T><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            num_pairs, search_sq, box, masks);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// Flat-list FULL force mask (optional export path; not default MD).
template <typename T>
void launch_compute_force_mask(const pos_t* cluster_x, const pos_t* cluster_y,
                               const pos_t* cluster_z, const int* flat_ci,
                               const int* flat_cj,
                               const int* num_flat_pairs_device, int pair_count,
                               T search_sq, cuda::Box box, uint64_t* flat_mask,
                               cudaStream_t stream) {
    if (pair_count <= 0)
        return;
    long long threads_needed =
        static_cast<long long>(pair_count) * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = static_cast<int>((threads_needed + block - 1) / block);
    if (box.ortho) {
        smart_rolling_prune<true, false, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, /*dirty_flags=*/nullptr, search_sq, box,
            flat_mask, /*prune_step=*/0, /*prune_period=*/1);
    } else {
        smart_rolling_prune<false, false, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, /*dirty_flags=*/nullptr, search_sq, box,
            flat_mask, /*prune_step=*/0, /*prune_period=*/1);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: build per-atom Verlet list from the cluster CSR
//
// One thread per cluster slot owns atom ai = original_index[ci*CS + slot]. It
// walks ci's neighbor-cluster CSR row and tests every neighbor-cluster atom
// against ai at the search radius (rc+skin), appending in-range global indices
// column-major into verlet_nbr[k*stride + ai]. Distances use the sorted cluster
// positions captured at this rebuild — the same data the CSR was built from, so
// the resulting list is a valid superset of cutoff neighbors for the whole
// reneighbor interval. Runs once per rebuild; the per-step force kernel then
// iterates this tight list (one thread per atom) instead of re-walking the
// cluster CSR with 8 redundant threads — the divergence/redundancy that made
// lj_force_cluster spend ~7/32 lanes active.
// ---------------------------------------------------------------------------
template <bool Ortho, typename T>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void build_verlet_list(
    const pos_t* __restrict__ cluster_x, const pos_t* __restrict__ cluster_y,
    const pos_t* __restrict__ cluster_z, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const int* __restrict__ original_index,
    int num_clusters, int stride, int max_neigh, T search_sq, cuda::Box box_arg,
    int* __restrict__ verlet_nbr, int* __restrict__ verlet_cnt,
    int* __restrict__ overflow) {
    const auto box = box_arg.current().as<T>();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / kClusterSize;
    int slot = tid % kClusterSize;
    if (ci >= num_clusters)
        return;
    int ai = original_index[ci * kClusterSize + slot];
    if (ai < 0)
        return;

    T xi = cluster_x[ci * kClusterSize + slot];
    T yi = cluster_y[ci * kClusterSize + slot];
    T zi = cluster_z[ci * kClusterSize + slot];

    int cnt = 0;
    int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
    for (int p = rstart; p < rend; ++p) {
        int cj = col_idx[p];
#pragma unroll
        for (int sj = 0; sj < kClusterSize; ++sj) {
            int aj = original_index[cj * kClusterSize + sj];
            if (aj < 0 || aj == ai)
                continue;
            T dx = cluster_x[cj * kClusterSize + sj] - xi;
            T dy = cluster_y[cj * kClusterSize + sj] - yi;
            T dz = cluster_z[cj * kClusterSize + sj] - zi;
            if constexpr (Ortho) {
                dx -= T(box.cell[0]) * nearbyint(dx * T(box.inv_l[0]));
                dy -= T(box.cell[4]) * nearbyint(dy * T(box.inv_l[1]));
                dz -= T(box.cell[8]) * nearbyint(dz * T(box.inv_l[2]));
            } else {
                box.min_image(dx, dy, dz);
            }
            T r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < search_sq) {
                if (cnt < max_neigh)
                    verlet_nbr[static_cast<std::size_t>(cnt) * stride + ai] =
                        aj;
                ++cnt;
            }
        }
    }
    verlet_cnt[ai] = cnt;
    atomicMax(overflow, cnt);
}

template <typename T>
void launch_build_verlet_list(const pos_t* cluster_x, const pos_t* cluster_y,
                              const pos_t* cluster_z, const int* row_ptr,
                              const int* col_idx, const int* original_index,
                              int num_clusters, int stride, int max_neigh,
                              T search_sq, cuda::Box box, int* verlet_nbr,
                              int* verlet_cnt, int* overflow,
                              cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    long long threads_needed =
        static_cast<long long>(num_clusters) * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = static_cast<int>((threads_needed + block - 1) / block);
    if (box.ortho) {
        build_verlet_list<true, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, original_index,
            num_clusters, stride, max_neigh, search_sq, box, verlet_nbr,
            verlet_cnt, overflow);
    } else {
        build_verlet_list<false, T><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, original_index,
            num_clusters, stride, max_neigh, search_sq, box, verlet_nbr,
            verlet_cnt, overflow);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Current periodic motion: one result per block, written directly into mapped
// pinned memory. All lanes participate, including a partially filled block.
// No global atomic, device clear, or copy-engine transfer is needed.
// ---------------------------------------------------------------------------
// The cell-change test compares the FP64 cells exactly; the motion test runs
// on the primary coordinates against a cell narrowed once to the scratch type.
template <typename T>
__global__ void atom_displacement_max(const pos_t* pos_x, const pos_t* pos_y,
                                      const pos_t* pos_z, const pos_t* ref_x,
                                      const pos_t* ref_y, const pos_t* ref_z,
                                      int num_atoms, T half_skin_sq,
                                      Box box_arg, int* block_flags,
                                      const Box* reference_box) {
    const Box geom = box_arg.current();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool moved = false;
    if (reference_box)
        for (int k = 0; k < 9; ++k)
            moved |= geom.cell[k] != reference_box->cell[k];
    if (i < num_atoms) {
        const auto box = geom.as<T>();
        T dx = pos_x[i] - ref_x[i];
        T dy = pos_y[i] - ref_y[i];
        T dz = pos_z[i] - ref_z[i];
        box.min_image(dx, dy, dz);
        moved |= dx * dx + dy * dy + dz * dz > half_skin_sq;
    }
    const int count = __syncthreads_count(moved);
    if (threadIdx.x == 0)
        block_flags[blockIdx.x] = count != 0;
}

template <typename T>
void launch_atom_displacement_max(const pos_t* pos_x, const pos_t* pos_y,
                                  const pos_t* pos_z, const pos_t* ref_x,
                                  const pos_t* ref_y, const pos_t* ref_z,
                                  int num_atoms, T half_skin_sq, Box box,
                                  int* block_flags, cudaStream_t stream,
                                  const Box* reference_box) {
    if (num_atoms == 0)
        return;
    atom_displacement_max<T><<<(num_atoms + 255) / 256, 256, 0, stream>>>(
        pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, num_atoms, half_skin_sq, box,
        block_flags, reference_box);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Snapshot per-atom positions into ref_{x,y,z}
// ---------------------------------------------------------------------------

void launch_snapshot_ref_atoms(const pos_t* pos_x, const pos_t* pos_y,
                               const pos_t* pos_z, pos_t* ref_x, pos_t* ref_y,
                               pos_t* ref_z, int num_atoms,
                               cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    const std::size_t bytes =
        static_cast<std::size_t>(num_atoms) * sizeof(pos_t);
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(ref_x, pos_x, bytes, cudaMemcpyDeviceToDevice, stream));
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(ref_y, pos_y, bytes, cudaMemcpyDeviceToDevice, stream));
    ATV_CUDA_CHECK(
        cudaMemcpyAsync(ref_z, pos_z, bytes, cudaMemcpyDeviceToDevice, stream));
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the force-path scratch type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_cluster_displacement =
    reinterpret_cast<const void*>(&launch_cluster_displacement<force_t>);
[[maybe_unused]] const void* _inst_launch_smart_rolling_prune =
    reinterpret_cast<const void*>(&launch_smart_rolling_prune<force_t>);
[[maybe_unused]] const void* _inst_launch_compute_force_mask =
    reinterpret_cast<const void*>(&launch_compute_force_mask<force_t>);
[[maybe_unused]] const void* _inst_launch_compute_csr_masks =
    reinterpret_cast<const void*>(&launch_compute_csr_masks<force_t>);
[[maybe_unused]] const void* _inst_launch_build_verlet_list =
    reinterpret_cast<const void*>(&launch_build_verlet_list<force_t>);
[[maybe_unused]] const void* _inst_launch_atom_displacement_max =
    reinterpret_cast<const void*>(&launch_atom_displacement_max<force_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
