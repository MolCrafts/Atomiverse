/// @file fused_search.cu
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
        flat_mask[p] = mask_in[p];
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

    // Warp-level max reduction within the kClusterSize-thread group
    unsigned mask =
        0xFFu << (((threadIdx.x % 32) / kClusterSize) * kClusterSize);
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

// `dirty_flags`, `prune_step`, `prune_period` are reserved for a
// future periodic-skip optimization (see cuda-md-review-cleanup-05*).
// The kernel currently re-prunes every step from current cluster
// positions per the correctness note above — the parameters are
// accepted-but-unread.
//
// Templated on `bool Ortho` so the triclinic branch (`box.min_image`'s
// inv_cell-roundtrip path) is dead-eliminated from the orthorhombic
// specialization. The ortho path is 3 FMAs per axis; the triclinic
// path is 12 muls + 6 adds + 3 nearbyint + back-multiply — saving
// ~4-5× FLOPS per pair × kClusterSize lanes × inner loop
// (cuda-md-review-cleanup-04-box-by-value-pbc).
template <bool Ortho, bool HalfNewton, typename work_t>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void smart_rolling_prune(
    const real_t* __restrict__ cluster_x, const real_t* __restrict__ cluster_y,
    const real_t* __restrict__ cluster_z, const int* __restrict__ flat_ci,
    const int* __restrict__ flat_cj,
    const int* __restrict__ num_flat_pairs_device,
    const int* __restrict__ dirty_flags, work_t cutoff_sq, cuda::Box box,
    uint64_t* __restrict__ flat_mask, int prune_step, int prune_period) {

    // kClusterSize threads per pair (8 threads compute the mask)
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

    work_t jx = cluster_x[cj * kClusterSize + i_lane];
    work_t jy = cluster_y[cj * kClusterSize + i_lane];
    work_t jz = cluster_z[cj * kClusterSize + i_lane];

    uint8_t my_bits = 0;
    int group_in_warp = (threadIdx.x % 32) / kClusterSize;
    unsigned submask = 0xFFu << (group_in_warp * kClusterSize);
    int group_base = group_in_warp * kClusterSize;

    for (int j = 0; j < kClusterSize; ++j) {
        work_t jx_j = __shfl_sync(submask, jx, group_base + j);
        work_t jy_j = __shfl_sync(submask, jy, group_base + j);
        work_t jz_j = __shfl_sync(submask, jz, group_base + j);

        work_t dx = jx_j - xi;
        work_t dy = jy_j - yi;
        work_t dz = jz_j - zi;

        // Compile-time-specialized PBC wrap. Ortho path: 3 FMA per axis
        // (no triclinic detour, no inv_cell roundtrip). Triclinic path:
        // full inv_cell·dx → fractional → nearbyint → cell·frac.
        if constexpr (Ortho) {
            dx -= work_t(box.cell[0]) * nearbyint(dx * work_t(box.inv_l[0]));
            dy -= work_t(box.cell[4]) * nearbyint(dy * work_t(box.inv_l[1]));
            dz -= work_t(box.cell[8]) * nearbyint(dz * work_t(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }

        work_t r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < cutoff_sq) {
            my_bits |= (1u << j);
        }
    }

    // Half-Newton exclusion for diagonal pairs (flat/export path only). The
    // atom-centric force kernel consumes a FULL list (each pair owned by both
    // endpoints, energy halved), so its mask must keep both directions —
    // HalfNewton=false leaves the diagonal symmetric.
    if constexpr (HalfNewton) {
        if (ci == cj) {
            uint8_t upper = static_cast<uint8_t>(0xFFu << (i_lane + 1));
            my_bits &= upper;
        }
    }

    uint64_t new_mask = 0;
    for (int k = 0; k < kClusterSize; ++k) {
        uint8_t other = static_cast<uint8_t>(
            __shfl_sync(submask, static_cast<int>(my_bits), group_base + k));
        new_mask |= (static_cast<uint64_t>(other) << (k * kClusterSize));
    }

    if (i_lane == 0) {
        flat_mask[pair_idx] = new_mask;
    }
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
// Force-path interaction mask: full (no half-Newton) per-cluster-pair mask at
// the SEARCH radius (rc+skin), computed once per rebuild. The atom-centric LJ
// force kernel reads it to iterate only the in-range j-slots of each neighbor
// cluster (≈ the real neighbor count) instead of all kClusterSize candidates —
// the dominant cost of the cluster-CSR force traversal. Reuses the
// smart_rolling_prune kernel body with HalfNewton=false. Valid for the whole
// reneighbor interval because the Verlet skin bounds inter-rebuild drift.
// ---------------------------------------------------------------------------
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
// Kernel: Per-atom displacement vs ref_{x,y,z}, atomicMax rebuild_flag
// ---------------------------------------------------------------------------
//
// Reads frame.atoms.x/y/z directly (always live), not cluster_x/y/z
// (sorted layout, stale between rebuilds). Each thread computes one
// atom's squared displacement; any thread whose d² > half_skin_sq
// races to atomicMax(rebuild_flag, 1). No reduction needed — we only
// care "any atom exceeded?", not "what's the max?".
//
// Cost: one global load × 6 (current + ref) + 1 atomicMax on at most
// one thread per launch. ~few µs per call on a 10k-atom system.

template <typename work_t>
__global__ void atom_displacement_max(
    const real_t* __restrict__ pos_x, const real_t* __restrict__ pos_y,
    const real_t* __restrict__ pos_z, const real_t* __restrict__ ref_x,
    const real_t* __restrict__ ref_y, const real_t* __restrict__ ref_z,
    int num_atoms, work_t half_skin_sq, int* __restrict__ rebuild_flag) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;

    work_t dx = pos_x[i] - ref_x[i];
    work_t dy = pos_y[i] - ref_y[i];
    work_t dz = pos_z[i] - ref_z[i];
    work_t d2 = dx * dx + dy * dy + dz * dz;

    if (d2 > half_skin_sq) {
        atomicMax(rebuild_flag, 1);
    }
}

template <typename work_t>
void launch_atom_displacement_max(const real_t* pos_x, const real_t* pos_y,
                                  const real_t* pos_z, const real_t* ref_x,
                                  const real_t* ref_y, const real_t* ref_z,
                                  int num_atoms, work_t half_skin_sq,
                                  int* rebuild_flag, cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    // Caller pre-clears rebuild_flag (the launcher mustn't, because we
    // need to OR into a multi-source flag eventually).
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    atom_displacement_max<work_t>
        <<<grid, block, 0, stream>>>(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z,
                                     num_atoms, half_skin_sq, rebuild_flag);
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
[[maybe_unused]] const void* _inst_launch_build_verlet_list =
    reinterpret_cast<const void*>(&launch_build_verlet_list<work_t>);
[[maybe_unused]] const void* _inst_launch_atom_displacement_max =
    reinterpret_cast<const void*>(&launch_atom_displacement_max<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
