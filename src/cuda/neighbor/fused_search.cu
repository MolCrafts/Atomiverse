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
#include <cuda_runtime.h>

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

template <typename work_t>
__global__ void cluster_displacement(
    const real_t* __restrict__ cluster_x, const real_t* __restrict__ cluster_y,
    const real_t* __restrict__ cluster_z,
    const real_t* __restrict__ ref_cluster_x,
    const real_t* __restrict__ ref_cluster_y,
    const real_t* __restrict__ ref_cluster_z,
    const int* __restrict__ cluster_count, int num_clusters,
    work_t dirty_threshold_sq, real_t* __restrict__ disp_sq_out,
    int* __restrict__ dirty_flags, int* __restrict__ rebuild_flag,
    work_t half_skin_sq) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % kClusterSize;
    int ci = tid / kClusterSize;

    if (ci >= num_clusters)
        return;

    int count = cluster_count[ci];
    int idx = ci * kClusterSize + lane;

    work_t dx =
        (lane < count) ? (cluster_x[idx] - ref_cluster_x[idx]) : work_t(0);
    work_t dy =
        (lane < count) ? (cluster_y[idx] - ref_cluster_y[idx]) : work_t(0);
    work_t dz =
        (lane < count) ? (cluster_z[idx] - ref_cluster_z[idx]) : work_t(0);
    work_t d2 = dx * dx + dy * dy + dz * dz;

    // Warp-level max reduction (OpenMM CS=32 ⇒ full warp).
    static_assert(kClusterSize == 32, "cluster_displacement assumes CS=32");
    unsigned mask = 0xffffffffu;
    for (int offset = kClusterSize / 2; offset >= 1; offset >>= 1) {
        work_t tmp = __shfl_down_sync(mask, d2, offset);
        d2 = fmax(d2, tmp);
    }

    if (lane == 0) {
        disp_sq_out[ci] = d2;
        dirty_flags[ci] = (d2 > dirty_threshold_sq) ? 1 : 0;
        if (d2 > half_skin_sq) {
            atomicMax(rebuild_flag, 1);
        }
    }
}

template <typename work_t>
void launch_cluster_displacement(
    const real_t* cluster_x, const real_t* cluster_y, const real_t* cluster_z,
    const real_t* ref_cluster_x, const real_t* ref_cluster_y,
    const real_t* ref_cluster_z, const int* cluster_count, int num_clusters,
    work_t dirty_threshold_sq, real_t* cluster_disp_sq, int* dirty_flags,
    int* rebuild_flag, work_t half_skin_sq, cudaStream_t stream) {
    if (num_clusters == 0)
        return;

    ATV_CUDA_CHECK(cudaMemsetAsync(rebuild_flag, 0, sizeof(int), stream));

    int threads_needed = num_clusters * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;

    cluster_displacement<work_t><<<grid, block, 0, stream>>>(
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
template <bool Ortho, bool HalfNewton, typename work_t>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void smart_rolling_prune(
    const real_t* __restrict__ cluster_x, const real_t* __restrict__ cluster_y,
    const real_t* __restrict__ cluster_z, const int* __restrict__ flat_ci,
    const int* __restrict__ flat_cj,
    const int* __restrict__ num_flat_pairs_device,
    const int* __restrict__ dirty_flags, work_t cutoff_sq, cuda::Box box,
    uint64_t* __restrict__ flat_mask, int prune_step, int prune_period) {
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

    work_t xi = cluster_x[ci * kClusterSize + i_lane];
    work_t yi = cluster_y[ci * kClusterSize + i_lane];
    work_t zi = cluster_z[ci * kClusterSize + i_lane];

    // Full warp owns this pair: each lane holds one j-slot of cj in registers.
    work_t jx = cluster_x[cj * kClusterSize + i_lane];
    work_t jy = cluster_y[cj * kClusterSize + i_lane];
    work_t jz = cluster_z[cj * kClusterSize + i_lane];

    std::uint32_t my_bits = 0;
    const unsigned submask = 0xffffffffu;

    for (int j = 0; j < kClusterSize; ++j) {
        work_t jx_j = __shfl_sync(submask, jx, j);
        work_t jy_j = __shfl_sync(submask, jy, j);
        work_t jz_j = __shfl_sync(submask, jz, j);

        work_t dx = jx_j - xi;
        work_t dy = jy_j - yi;
        work_t dz = jz_j - zi;

        if constexpr (Ortho) {
            dx -= work_t(box.cell[0]) * nearbyint(dx * work_t(box.inv_l[0]));
            dy -= work_t(box.cell[4]) * nearbyint(dy * work_t(box.inv_l[1]));
            dz -= work_t(box.cell[8]) * nearbyint(dz * work_t(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }

        work_t r2 = dx * dx + dy * dy + dz * dz;
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

template <typename work_t>
void launch_smart_rolling_prune(
    const real_t* cluster_x, const real_t* cluster_y, const real_t* cluster_z,
    const int* flat_ci, const int* flat_cj, const int* num_flat_pairs_device,
    int pair_capacity, const int* dirty_flags, work_t cutoff_sq, cuda::Box box,
    uint64_t* flat_mask, int prune_step, int prune_period,
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
        smart_rolling_prune<true, true, work_t><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, dirty_flags, cutoff_sq, box, flat_mask,
            prune_step, prune_period);
    } else {
        smart_rolling_prune<false, true, work_t><<<grid, block, 0, stream>>>(
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
// Warp/tile scheduling follows OpenMM platforms/cuda/src/kernels/nonbonded.cu
// SHA 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e. The existing ATV exclusion
// CSR, binary search and mask semantics are unchanged; no source is copied.
__global__ __launch_bounds__(128) void bake_csr_exclusions_k(
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const int* __restrict__ original_index, int num_clusters, int num_pairs,
    const int* __restrict__ offsets, const int* __restrict__ partners,
    uint64_t* __restrict__ masks) {
    static_assert(kClusterSize == 32, "exclusion tiles assume CS=32");
    const int p =
        blockIdx.x * (blockDim.x / kClusterSize) + threadIdx.x / kClusterSize;
    const int lane = threadIdx.x % kClusterSize;
    if (p >= num_pairs)
        return; // Whole warp; no block collective follows.
    int ci = 0;
    if (lane == 0) {
        int lo = 0, hi = num_clusters;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (row_ptr[mid + 1] <= p)
                lo = mid + 1;
            else
                hi = mid;
        }
        ci = lo;
    }
    ci = __shfl_sync(0xffffffffu, ci, 0);
    const int ai = original_index[ci * kClusterSize + lane];
    const std::uint32_t bits = ai >= 0 ? mask_jbits(masks, p, lane) : 0;
    if (!__any_sync(0xffffffffu, bits != 0)) {
        mask_set_jbits(masks, p, lane, 0);
        return;
    }
    const int begin = ai >= 0 ? offsets[ai] : 0;
    const int end = ai >= 0 ? offsets[ai + 1] : 0;
    const int cj = col_idx[p];
    // Coalesced J indices are reused by every I row of this tile. All lanes
    // reach each shuffle, including padded rows and rows with no set bits.
    const int aj_lane = original_index[cj * kClusterSize + lane];
    std::uint32_t retained = 0;
#pragma unroll
    for (int sj = 0; sj < kClusterSize; ++sj) {
        const int aj = __shfl_sync(0xffffffffu, aj_lane, sj);
        if ((bits & (1u << sj)) == 0 || aj < 0)
            continue;
        int lo = begin, hi = end;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (partners[mid] < aj)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == end || partners[lo] != aj)
            retained |= (1u << sj);
    }
    mask_set_jbits(masks, p, lane, retained);
}

bool launch_bake_csr_exclusions(const int* row_ptr, const int* col_idx,
                                const int* original_index, int num_clusters,
                                int num_pairs, const int* excl_offsets,
                                const int* excl_partners, double special_scale,
                                uint64_t* masks, cudaStream_t stream) {
    if (excl_offsets == nullptr || special_scale != 0.0)
        return false;
    if (num_clusters > 0 && num_pairs > 0) {
        constexpr int threads = 128;
        constexpr int tiles_per_block = threads / kClusterSize;
        const int grid = (num_pairs - 1) / tiles_per_block + 1;
        bake_csr_exclusions_k<<<grid, threads, 0, stream>>>(
            row_ptr, col_idx, original_index, num_clusters, num_pairs,
            excl_offsets, excl_partners, masks);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

// A warp owns one CSR tile, matching the OpenMM warp/tile work partition
// (platforms/cuda/src/kernels/nonbonded.cu, SHA
// 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e). The geometric predicate and
// mask layout remain ATV-native; no OpenMM source is copied.
template <bool Ortho, typename work_t>
__global__ __launch_bounds__(128) void compute_csr_masks_k(
    const real_t* __restrict__ cluster_x, const real_t* __restrict__ cluster_y,
    const real_t* __restrict__ cluster_z, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, int num_clusters, int num_pairs,
    work_t search_sq, cuda::Box box, uint64_t* __restrict__ masks) {
    static_assert(kClusterSize == 32, "CSR mask kernel assumes CS=32");
    const int p =
        blockIdx.x * (blockDim.x / kClusterSize) + threadIdx.x / kClusterSize;
    const int lane = threadIdx.x % kClusterSize;
    if (p >= num_pairs)
        return; // Uniform within the warp; no block collective follows.
    int ci = 0;
    if (lane == 0) {
        int lo = 0, hi = num_clusters;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (row_ptr[mid + 1] <= p)
                lo = mid + 1;
            else
                hi = mid;
        }
        ci = lo;
    }
    ci = __shfl_sync(0xffffffffu, ci, 0);
    const int cj = col_idx[p];
    const work_t xi = cluster_x[ci * kClusterSize + lane];
    const work_t yi = cluster_y[ci * kClusterSize + lane];
    const work_t zi = cluster_z[ci * kClusterSize + lane];
    // Coalesced J-block loads are reused through warp exchange for all 32
    // rows; no warp serializes the remaining tiles in the same CSR row.
    const real_t xj = cluster_x[cj * kClusterSize + lane];
    const real_t yj = cluster_y[cj * kClusterSize + lane];
    const real_t zj = cluster_z[cj * kClusterSize + lane];
    std::uint32_t bits = 0;
#pragma unroll 4
    for (int j = 0; j < kClusterSize; ++j) {
        work_t dx = __shfl_sync(0xffffffffu, xj, j) - xi;
        work_t dy = __shfl_sync(0xffffffffu, yj, j) - yi;
        work_t dz = __shfl_sync(0xffffffffu, zj, j) - zi;
        if constexpr (Ortho) {
            dx -= work_t(box.cell[0]) * nearbyint(dx * work_t(box.inv_l[0]));
            dy -= work_t(box.cell[4]) * nearbyint(dy * work_t(box.inv_l[1]));
            dz -= work_t(box.cell[8]) * nearbyint(dz * work_t(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }
        const work_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < search_sq && r2 > work_t(0))
            bits |= (1u << j);
    }
    // Exclude the same atom on diagonal tiles exactly as the row kernel did.
    if (ci == cj)
        bits &= ~(1u << lane);
    mask_set_jbits(masks, p, lane, bits);
}

template <typename work_t>
void launch_compute_csr_masks(const real_t* cluster_x, const real_t* cluster_y,
                              const real_t* cluster_z, const int* row_ptr,
                              const int* col_idx, int num_clusters,
                              int num_pairs, work_t search_sq, cuda::Box box,
                              uint64_t* masks, cudaStream_t stream) {
    if (num_clusters <= 0 || num_pairs <= 0)
        return;
    constexpr int threads = 128;
    constexpr int tiles_per_block = threads / kClusterSize;
    const int grid = (num_pairs - 1) / tiles_per_block + 1;
    if (box.ortho) {
        compute_csr_masks_k<true, work_t><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            num_pairs, search_sq, box, masks);
    } else {
        compute_csr_masks_k<false, work_t><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            num_pairs, search_sq, box, masks);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// Flat-list FULL force mask (optional export path; not default MD).
template <typename work_t>
void launch_compute_force_mask(const real_t* cluster_x, const real_t* cluster_y,
                               const real_t* cluster_z, const int* flat_ci,
                               const int* flat_cj,
                               const int* num_flat_pairs_device, int pair_count,
                               work_t search_sq, cuda::Box box,
                               uint64_t* flat_mask, cudaStream_t stream) {
    if (pair_count <= 0)
        return;
    long long threads_needed =
        static_cast<long long>(pair_count) * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = static_cast<int>((threads_needed + block - 1) / block);
    if (box.ortho) {
        smart_rolling_prune<true, false, work_t><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, flat_ci, flat_cj,
            num_flat_pairs_device, /*dirty_flags=*/nullptr, search_sq, box,
            flat_mask, /*prune_step=*/0, /*prune_period=*/1);
    } else {
        smart_rolling_prune<false, false, work_t><<<grid, block, 0, stream>>>(
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
template <bool Ortho, typename work_t>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void build_verlet_list(
    const real_t* __restrict__ cluster_x, const real_t* __restrict__ cluster_y,
    const real_t* __restrict__ cluster_z, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, const int* __restrict__ original_index,
    int num_clusters, int stride, int max_neigh, work_t search_sq,
    cuda::Box box, int* __restrict__ verlet_nbr, int* __restrict__ verlet_cnt,
    int* __restrict__ overflow) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = tid / kClusterSize;
    int slot = tid % kClusterSize;
    if (ci >= num_clusters)
        return;
    int ai = original_index[ci * kClusterSize + slot];
    if (ai < 0)
        return;

    work_t xi = cluster_x[ci * kClusterSize + slot];
    work_t yi = cluster_y[ci * kClusterSize + slot];
    work_t zi = cluster_z[ci * kClusterSize + slot];

    int cnt = 0;
    int rstart = row_ptr[ci], rend = row_ptr[ci + 1];
    for (int p = rstart; p < rend; ++p) {
        int cj = col_idx[p];
#pragma unroll
        for (int sj = 0; sj < kClusterSize; ++sj) {
            int aj = original_index[cj * kClusterSize + sj];
            if (aj < 0 || aj == ai)
                continue;
            work_t dx = cluster_x[cj * kClusterSize + sj] - xi;
            work_t dy = cluster_y[cj * kClusterSize + sj] - yi;
            work_t dz = cluster_z[cj * kClusterSize + sj] - zi;
            if constexpr (Ortho) {
                dx -=
                    work_t(box.cell[0]) * nearbyint(dx * work_t(box.inv_l[0]));
                dy -=
                    work_t(box.cell[4]) * nearbyint(dy * work_t(box.inv_l[1]));
                dz -=
                    work_t(box.cell[8]) * nearbyint(dz * work_t(box.inv_l[2]));
            } else {
                box.min_image(dx, dy, dz);
            }
            work_t r2 = dx * dx + dy * dy + dz * dz;
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

template <typename work_t>
void launch_build_verlet_list(const real_t* cluster_x, const real_t* cluster_y,
                              const real_t* cluster_z, const int* row_ptr,
                              const int* col_idx, const int* original_index,
                              int num_clusters, int stride, int max_neigh,
                              work_t search_sq, cuda::Box box, int* verlet_nbr,
                              int* verlet_cnt, int* overflow,
                              cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    long long threads_needed =
        static_cast<long long>(num_clusters) * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = static_cast<int>((threads_needed + block - 1) / block);
    if (box.ortho) {
        build_verlet_list<true, work_t><<<grid, block, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, original_index,
            num_clusters, stride, max_neigh, search_sq, box, verlet_nbr,
            verlet_cnt, overflow);
    } else {
        build_verlet_list<false, work_t><<<grid, block, 0, stream>>>(
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
template <typename work_t>
__global__ void atom_displacement_max(const real_t* pos_x, const real_t* pos_y,
                                      const real_t* pos_z, const real_t* ref_x,
                                      const real_t* ref_y, const real_t* ref_z,
                                      int num_atoms, work_t half_skin_sq,
                                      Box box, int* block_flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool moved = false;
    if (i < num_atoms) {
        work_t dx = pos_x[i] - ref_x[i];
        work_t dy = pos_y[i] - ref_y[i];
        work_t dz = pos_z[i] - ref_z[i];
        box.min_image(dx, dy, dz);
        moved = dx * dx + dy * dy + dz * dz > half_skin_sq;
    }
    const int count = __syncthreads_count(moved);
    if (threadIdx.x == 0)
        block_flags[blockIdx.x] = count != 0;
}

template <typename work_t>
void launch_atom_displacement_max(const real_t* pos_x, const real_t* pos_y,
                                  const real_t* pos_z, const real_t* ref_x,
                                  const real_t* ref_y, const real_t* ref_z,
                                  int num_atoms, work_t half_skin_sq, Box box,
                                  int* block_flags, cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    atom_displacement_max<work_t><<<(num_atoms + 255) / 256, 256, 0, stream>>>(
        pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, num_atoms, half_skin_sq, box,
        block_flags);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Snapshot per-atom positions into ref_{x,y,z}
// ---------------------------------------------------------------------------

__global__ void snapshot_ref_atoms(const real_t* __restrict__ pos_x,
                                   const real_t* __restrict__ pos_y,
                                   const real_t* __restrict__ pos_z,
                                   real_t* __restrict__ ref_x,
                                   real_t* __restrict__ ref_y,
                                   real_t* __restrict__ ref_z, int num_atoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;
    ref_x[i] = pos_x[i];
    ref_y[i] = pos_y[i];
    ref_z[i] = pos_z[i];
}

void launch_snapshot_ref_atoms(const real_t* pos_x, const real_t* pos_y,
                               const real_t* pos_z, real_t* ref_x,
                               real_t* ref_y, real_t* ref_z, int num_atoms,
                               cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    snapshot_ref_atoms<<<grid, block, 0, stream>>>(pos_x, pos_y, pos_z, ref_x,
                                                   ref_y, ref_z, num_atoms);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the configured work_t type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_cluster_displacement =
    reinterpret_cast<const void*>(&launch_cluster_displacement<work_t>);
[[maybe_unused]] const void* _inst_launch_smart_rolling_prune =
    reinterpret_cast<const void*>(&launch_smart_rolling_prune<work_t>);
[[maybe_unused]] const void* _inst_launch_compute_force_mask =
    reinterpret_cast<const void*>(&launch_compute_force_mask<work_t>);
[[maybe_unused]] const void* _inst_launch_compute_csr_masks =
    reinterpret_cast<const void*>(&launch_compute_csr_masks<work_t>);
[[maybe_unused]] const void* _inst_launch_build_verlet_list =
    reinterpret_cast<const void*>(&launch_build_verlet_list<work_t>);
[[maybe_unused]] const void* _inst_launch_atom_displacement_max =
    reinterpret_cast<const void*>(&launch_atom_displacement_max<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
