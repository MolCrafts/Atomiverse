/// @file cuda/neighbor/pair_search.cu
/// @brief Hierarchical cluster AABB search with exact CSR count and fill.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Device: Minimum AABB distance² between two clusters
// ---------------------------------------------------------------------------

// Templated on `bool Ortho` so the triclinic min_image branch is
// dead-eliminated from the orthorhombic specialization
// (cuda-md-review-cleanup-04-box-by-value-pbc).
template <bool Ortho, typename work_t>
__device__ work_t min_aabb_dist_sq(work_t ci_min_x, work_t ci_min_y,
                                   work_t ci_min_z, work_t ci_max_x,
                                   work_t ci_max_y, work_t ci_max_z,
                                   work_t cj_min_x, work_t cj_min_y,
                                   work_t cj_min_z, work_t cj_max_x,
                                   work_t cj_max_y, work_t cj_max_z,
                                   bool periodic, cuda::Box box) {
    work_t cx_i = (ci_min_x + ci_max_x) * work_t(0.5);
    work_t cy_i = (ci_min_y + ci_max_y) * work_t(0.5);
    work_t cz_i = (ci_min_z + ci_max_z) * work_t(0.5);
    work_t cx_j = (cj_min_x + cj_max_x) * work_t(0.5);
    work_t cy_j = (cj_min_y + cj_max_y) * work_t(0.5);
    work_t cz_j = (cj_min_z + cj_max_z) * work_t(0.5);

    work_t dx = cx_j - cx_i, dy = cy_j - cy_i, dz = cz_j - cz_i;
    if (periodic) {
        if constexpr (Ortho) {
            dx -= work_t(box.cell[0]) * nearbyint(dx * work_t(box.inv_l[0]));
            dy -= work_t(box.cell[4]) * nearbyint(dy * work_t(box.inv_l[1]));
            dz -= work_t(box.cell[8]) * nearbyint(dz * work_t(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }
    }

    work_t hx_i = (ci_max_x - ci_min_x) * work_t(0.5);
    work_t hy_i = (ci_max_y - ci_min_y) * work_t(0.5);
    work_t hz_i = (ci_max_z - ci_min_z) * work_t(0.5);
    work_t hx_j = (cj_max_x - cj_min_x) * work_t(0.5);
    work_t hy_j = (cj_max_y - cj_min_y) * work_t(0.5);
    work_t hz_j = (cj_max_z - cj_min_z) * work_t(0.5);

    work_t gx = fmax(work_t(0), fabs(dx) - hx_i - hx_j);
    work_t gy = fmax(work_t(0), fabs(dy) - hy_i - hy_j);
    work_t gz = fmax(work_t(0), fabs(dz) - hz_i - hz_j);
    return gx * gx + gy * gy + gz * gz;
}

// Algorithm reference: OpenMM platforms/cuda/src/kernels/findInteractingBlocks.cu
// at 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e, sortBoxData and
// findBlocksWithInteractions: coarse AABBs over 32 blocks, followed by
// warp-parallel block AABB rejection. This ATV implementation reuses its own
// AABB predicate and CSR count/fill primitives; no OpenMM source is copied.
// A centroid grid with an atom-cutoff stencil is not conservative: a cluster
// can extend outside its center's cell by arbitrarily many bins.

__global__ void large_cluster_bounds_k(ClusterBounds b, int num_clusters,
                                       real_t* __restrict__ large_bounds) {
    int lane = threadIdx.x % 32;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_groups = (num_clusters + 31) / 32;
    if (group >= num_groups)
        return;
    int ci = group * 32 + lane;
    real_t lo_x = ci < num_clusters ? b.min_x[ci] : real_t(INFINITY);
    real_t lo_y = ci < num_clusters ? b.min_y[ci] : real_t(INFINITY);
    real_t lo_z = ci < num_clusters ? b.min_z[ci] : real_t(INFINITY);
    real_t hi_x = ci < num_clusters ? b.max_x[ci] : real_t(-INFINITY);
    real_t hi_y = ci < num_clusters ? b.max_y[ci] : real_t(-INFINITY);
    real_t hi_z = ci < num_clusters ? b.max_z[ci] : real_t(-INFINITY);
    for (int offset = 16; offset; offset >>= 1) {
        lo_x = fmin(lo_x, __shfl_down_sync(0xffffffffu, lo_x, offset));
        lo_y = fmin(lo_y, __shfl_down_sync(0xffffffffu, lo_y, offset));
        lo_z = fmin(lo_z, __shfl_down_sync(0xffffffffu, lo_z, offset));
        hi_x = fmax(hi_x, __shfl_down_sync(0xffffffffu, hi_x, offset));
        hi_y = fmax(hi_y, __shfl_down_sync(0xffffffffu, hi_y, offset));
        hi_z = fmax(hi_z, __shfl_down_sync(0xffffffffu, hi_z, offset));
    }
    if (lane == 0) {
        large_bounds[group] = lo_x;
        large_bounds[num_groups + group] = lo_y;
        large_bounds[2 * num_groups + group] = lo_z;
        large_bounds[3 * num_groups + group] = hi_x;
        large_bounds[4 * num_groups + group] = hi_y;
        large_bounds[5 * num_groups + group] = hi_z;
    }
}

void launch_large_cluster_bounds(ClusterBounds bounds, int num_clusters,
                                 real_t* large_bounds, cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int num_groups = (num_clusters + 31) / 32;
    int grid = ATV::cuda::grid_size(num_groups * 32);
    large_cluster_bounds_k<<<grid, ATV::cuda::kDefaultBlock, 0, stream>>>(
        bounds, num_clusters, large_bounds);
    ATV_CUDA_CHECK(cudaGetLastError());
}

template <bool Ortho, typename work_t>
__device__ bool bounds_interact(ClusterBounds a, int ai, ClusterBounds b,
                                int bi, bool periodic, work_t cutoff_sq,
                                cuda::Box box) {
    // Axis-aligned Cartesian bounds under a skew lattice do not guarantee
    // that center minimum-image selects the closest box image. Let the exact
    // per-atom mask resolve these candidates rather than dropping any tile.
    if constexpr (!Ortho) {
        if (periodic)
            return true;
    }
    return min_aabb_dist_sq<Ortho, work_t>(
               a.min_x[ai], a.min_y[ai], a.min_z[ai], a.max_x[ai], a.max_y[ai],
               a.max_z[ai], b.min_x[bi], b.min_y[bi], b.min_z[bi], b.max_x[bi],
               b.max_y[bi], b.max_z[bi], periodic, box)
           <= cutoff_sq;
}

template <bool Ortho, bool CountOnly, typename work_t>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void fill_pairs(
    ClusterBounds bounds, const real_t* __restrict__ large_bounds,
    int num_clusters, bool periodic, work_t cutoff_plus_skin_sq, cuda::Box box,
    const int* __restrict__ row_ptr, int* __restrict__ col_idx,
    int* __restrict__ pair_counts) {
    int ci = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (ci >= num_clusters)
        return;
    const int num_groups = (num_clusters + 31) / 32;
    ClusterBounds large{large_bounds,
                        large_bounds + num_groups,
                        large_bounds + 2 * num_groups,
                        large_bounds + 3 * num_groups,
                        large_bounds + 4 * num_groups,
                        large_bounds + 5 * num_groups};
    int count = 0;
    int write_offset = CountOnly ? 0 : row_ptr[ci];
    for (int base = 0; base < num_groups; base += 32) {
        int group = base + lane;
        bool possible =
            group < num_groups
            && bounds_interact<Ortho, work_t>(
                bounds, ci, large, group, periodic, cutoff_plus_skin_sq, box);
        unsigned groups = __ballot_sync(0xffffffffu, possible);
        while (groups) {
            int group_idx = base + __ffs(static_cast<int>(groups)) - 1;
            groups &= groups - 1;
            int cj = group_idx * 32 + lane;
            bool passes =
                cj < num_clusters
                && bounds_interact<Ortho, work_t>(
                    bounds, ci, bounds, cj, periodic, cutoff_plus_skin_sq, box);
            unsigned accepted = __ballot_sync(0xffffffffu, passes);
            int batch_count = __popc(accepted);
            if constexpr (!CountOnly) {
                if (passes) {
                    int prefix = __popc(accepted & ((1u << lane) - 1));
                    col_idx[write_offset + prefix] = cj;
                }
                write_offset += batch_count;
            }
            count += batch_count;
        }
    }
    if constexpr (CountOnly) {
        if (lane == 0)
            pair_counts[ci] = count;
    }
}

template <typename work_t>
void launch_fill_cluster_pairs(ClusterBounds bounds, const real_t* large_bounds,
                               int num_clusters, bool periodic,
                               work_t cutoff_plus_skin_sq, cuda::Box box,
                               const int* row_ptr, int* col_idx,
                               cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int grid = ATV::cuda::grid_size(num_clusters * 32);
    int block = ATV::cuda::kDefaultBlock;
    if (box.ortho) {
        fill_pairs<true, false, work_t><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, row_ptr, col_idx, nullptr);
    } else {
        fill_pairs<false, false, work_t><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, row_ptr, col_idx, nullptr);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

template <typename work_t>
void launch_count_cluster_pairs(ClusterBounds bounds,
                                const real_t* large_bounds, int num_clusters,
                                bool periodic, work_t cutoff_plus_skin_sq,
                                cuda::Box box, int* pair_counts,
                                cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int grid = ATV::cuda::grid_size(num_clusters * 32);
    int block = ATV::cuda::kDefaultBlock;
    if (box.ortho) {
        fill_pairs<true, true, work_t><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, nullptr, nullptr, pair_counts);
    } else {
        fill_pairs<false, true, work_t><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, nullptr, nullptr, pair_counts);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

__global__ void write_row_ptr_sentinel(int* __restrict__ row_ptr,
                                       const int* __restrict__ pair_counts,
                                       int num_clusters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        row_ptr[num_clusters] =
            num_clusters > 0
                ? row_ptr[num_clusters - 1] + pair_counts[num_clusters - 1]
                : 0;
    }
}

void launch_write_row_ptr_sentinel(int* row_ptr, const int* pair_counts,
                                   int num_clusters, cudaStream_t stream) {
    write_row_ptr_sentinel<<<1, 1, 0, stream>>>(row_ptr, pair_counts,
                                                num_clusters);
    ATV_CUDA_CHECK(cudaGetLastError());
}

namespace {
[[maybe_unused]] const void* _inst_launch_fill_cluster_pairs =
    reinterpret_cast<const void*>(&launch_fill_cluster_pairs<work_t>);
[[maybe_unused]] const void* _inst_launch_count_cluster_pairs =
    reinterpret_cast<const void*>(&launch_count_cluster_pairs<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
