/// @file cuda/neighbor/pair_search.cu
/// @brief Hierarchical cluster AABB search with exact CSR count and fill.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/limits>
#include <cuda_runtime.h>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Device: Minimum AABB distance² between two clusters
// ---------------------------------------------------------------------------

// Templated on `bool Ortho` so the triclinic min_image branch is
// dead-eliminated from the orthorhombic specialization
// (cuda-md-review-cleanup-04-box-by-value-pbc).
template <bool Ortho, typename T>
__device__ T min_aabb_dist_sq(T ci_min_x, T ci_min_y, T ci_min_z, T ci_max_x,
                              T ci_max_y, T ci_max_z, T cj_min_x, T cj_min_y,
                              T cj_min_z, T cj_max_x, T cj_max_y, T cj_max_z,
                              bool periodic, BoxGeometry<T> box) {
    T cx_i = (ci_min_x + ci_max_x) * T(0.5);
    T cy_i = (ci_min_y + ci_max_y) * T(0.5);
    T cz_i = (ci_min_z + ci_max_z) * T(0.5);
    T cx_j = (cj_min_x + cj_max_x) * T(0.5);
    T cy_j = (cj_min_y + cj_max_y) * T(0.5);
    T cz_j = (cj_min_z + cj_max_z) * T(0.5);

    T dx = cx_j - cx_i, dy = cy_j - cy_i, dz = cz_j - cz_i;
    if (periodic) {
        if constexpr (Ortho) {
            dx -= T(box.cell[0]) * nearbyint(dx * T(box.inv_l[0]));
            dy -= T(box.cell[4]) * nearbyint(dy * T(box.inv_l[1]));
            dz -= T(box.cell[8]) * nearbyint(dz * T(box.inv_l[2]));
        } else {
            box.min_image(dx, dy, dz);
        }
    }

    T hx_i = (ci_max_x - ci_min_x) * T(0.5);
    T hy_i = (ci_max_y - ci_min_y) * T(0.5);
    T hz_i = (ci_max_z - ci_min_z) * T(0.5);
    T hx_j = (cj_max_x - cj_min_x) * T(0.5);
    T hy_j = (cj_max_y - cj_min_y) * T(0.5);
    T hz_j = (cj_max_z - cj_min_z) * T(0.5);

    T gx = fmax(T(0), fabs(dx) - hx_i - hx_j);
    T gy = fmax(T(0), fabs(dy) - hy_i - hy_j);
    T gz = fmax(T(0), fabs(dz) - hz_i - hz_j);
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
                                       force_t* __restrict__ large_bounds) {
    int lane = threadIdx.x % 32;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_groups = (num_clusters + 31) / 32;
    if (group >= num_groups)
        return;
    int ci = group * 32 + lane;
    force_t lo_x = ci < num_clusters ? b.min_x[ci] : force_t(INFINITY);
    force_t lo_y = ci < num_clusters ? b.min_y[ci] : force_t(INFINITY);
    force_t lo_z = ci < num_clusters ? b.min_z[ci] : force_t(INFINITY);
    force_t hi_x = ci < num_clusters ? b.max_x[ci] : force_t(-INFINITY);
    force_t hi_y = ci < num_clusters ? b.max_y[ci] : force_t(-INFINITY);
    force_t hi_z = ci < num_clusters ? b.max_z[ci] : force_t(-INFINITY);
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    lo_x = cg::reduce(warp, lo_x, cg::less<force_t>{});
    hi_x = cg::reduce(warp, hi_x, cg::greater<force_t>{});
    lo_y = cg::reduce(warp, lo_y, cg::less<force_t>{});
    hi_y = cg::reduce(warp, hi_y, cg::greater<force_t>{});
    lo_z = cg::reduce(warp, lo_z, cg::less<force_t>{});
    hi_z = cg::reduce(warp, hi_z, cg::greater<force_t>{});

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
                                 force_t* large_bounds, cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int num_groups = (num_clusters + 31) / 32;
    int grid = ATV::cuda::grid_size(num_groups * 32);
    large_cluster_bounds_k<<<grid, ATV::cuda::kDefaultBlock, 0, stream>>>(
        bounds, num_clusters, large_bounds);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// Conservative periodic lower bound for skew cells. Any separation q − p of
// points in the two boxes is d + e with d the center difference and
// |e_j| <= h_j (summed half-extents). Lattice vectors are the cell columns,
// so the fractional coordinate along reciprocal vector b_k (row k of the
// inverse cell) of every periodic image lies in [s_k − r_k, s_k + r_k] + Z,
// r_k = Σ_j h_j |b_kj|.
//  1. Projection bound, valid for every image: |v| >= |v·b_k| / |b_k|, so
//     the distance of that interval to the nearest integer times the plane
//     spacing w_k = 1/|b_k| bounds all images; the largest k is kept.
//  2. Image bound: an image n further than one cell from the rounded center
//     difference has |s_k + n_k| >= 1.5 along some k; when (1.5 − r_k) w_k
//     exceeds the search radius for every k, only the 27 nearest images can
//     interact, and the Cartesian AABB gap of each bounds its distance.
// The larger of the two is returned. Intervals are widened by a rounding
// margin, so a doubtful pair is kept for the exact per-atom mask.
template <typename T>
__device__ T skew_periodic_lower_bound_sq(T ci_min_x, T ci_min_y, T ci_min_z,
                                          T ci_max_x, T ci_max_y, T ci_max_z,
                                          T cj_min_x, T cj_min_y, T cj_min_z,
                                          T cj_max_x, T cj_max_y, T cj_max_z,
                                          const BoxGeometry<T>& box,
                                          T search_sq) {
    const T d[3] = {T(0.5) * (cj_min_x + cj_max_x - ci_min_x - ci_max_x),
                    T(0.5) * (cj_min_y + cj_max_y - ci_min_y - ci_max_y),
                    T(0.5) * (cj_min_z + cj_max_z - ci_min_z - ci_max_z)};
    const T h[3] = {T(0.5) * (ci_max_x - ci_min_x + cj_max_x - cj_min_x),
                    T(0.5) * (ci_max_y - ci_min_y + cj_max_y - cj_min_y),
                    T(0.5) * (ci_max_z - ci_min_z + cj_max_z - cj_min_z)};
    constexpr T kSlack = T(64) * ::cuda::std::numeric_limits<T>::epsilon();
    T projection = T(0), s_round[3];
    bool images_bounded = true;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        T s = T(0), r = T(0), norm2 = T(0);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const T b = box.inv_cell[3 * k + j];
            s += b * d[j];
            r += fabs(b) * h[j];
            norm2 += b * b;
        }
        s_round[k] = nearbyint(s);
        const T margin = kSlack * (fabs(s) + r + T(1));
        const T spacing = rsqrt(norm2);
        const T gap = fabs(s - s_round[k]) - r - margin;
        if (gap > T(0))
            projection = fmax(projection, gap * spacing);
        const T far = (T(1.5) - r - margin) * spacing;
        images_bounded = images_bounded && far > T(0) && far * far > search_sq;
    }
    T bound = projection * projection;
    if (images_bounded) {
        T nearest = ::cuda::std::numeric_limits<T>::infinity();
        for (int n0 = -1; n0 <= 1; ++n0)
            for (int n1 = -1; n1 <= 1; ++n1)
                for (int n2 = -1; n2 <= 1; ++n2) {
                    const T t[3] = {T(n0) - s_round[0], T(n1) - s_round[1],
                                    T(n2) - s_round[2]};
                    T gap2 = T(0);
#pragma unroll
                    for (int j = 0; j < 3; ++j) {
                        const T v = d[j] + box.cell[3 * j] * t[0]
                                    + box.cell[3 * j + 1] * t[1]
                                    + box.cell[3 * j + 2] * t[2];
                        const T g =
                            fabs(v) - h[j] - kSlack * (fabs(v) + h[j] + T(1));
                        if (g > T(0))
                            gap2 += g * g;
                    }
                    nearest = fmin(nearest, gap2);
                }
        bound = fmax(bound, nearest);
    }
    return bound;
}

template <bool Ortho, typename T>
__device__ bool bounds_interact(ClusterBounds a, int ai, ClusterBounds b,
                                int bi, bool periodic, T cutoff_sq,
                                BoxGeometry<T> box) {
    // Axis-aligned Cartesian bounds under a skew lattice do not guarantee
    // that center minimum-image selects the closest box image; use the
    // fractional-projection lower bound instead.
    if constexpr (!Ortho) {
        if (periodic) {
            const T search_sq = cutoff_sq * (T(1) + T(1e-5));
            return skew_periodic_lower_bound_sq<T>(
                       a.min_x[ai], a.min_y[ai], a.min_z[ai], a.max_x[ai],
                       a.max_y[ai], a.max_z[ai], b.min_x[bi], b.min_y[bi],
                       b.min_z[bi], b.max_x[bi], b.max_y[bi], b.max_z[bi], box,
                       search_sq)
                   <= search_sq;
        }
    }
    return min_aabb_dist_sq<Ortho, T>(
               a.min_x[ai], a.min_y[ai], a.min_z[ai], a.max_x[ai], a.max_y[ai],
               a.max_z[ai], b.min_x[bi], b.min_y[bi], b.min_z[bi], b.max_x[bi],
               b.max_y[bi], b.max_z[bi], periodic, box)
           <= cutoff_sq;
}

template <bool Ortho, bool CountOnly, typename T>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void fill_pairs(
    ClusterBounds bounds, const force_t* __restrict__ large_bounds,
    int num_clusters, bool periodic, T cutoff_plus_skin_sq, cuda::Box box_arg,
    const int* __restrict__ row_ptr, int* __restrict__ col_idx,
    int* __restrict__ pair_counts) {
    const auto box = box_arg.current().as<T>();
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
            && bounds_interact<Ortho, T>(bounds, ci, large, group, periodic,
                                         cutoff_plus_skin_sq, box);
        unsigned groups = __ballot_sync(0xffffffffu, possible);
        while (groups) {
            int group_idx = base + __ffs(static_cast<int>(groups)) - 1;
            groups &= groups - 1;
            int cj = group_idx * 32 + lane;
            bool passes =
                cj < num_clusters
                && bounds_interact<Ortho, T>(bounds, ci, bounds, cj, periodic,
                                             cutoff_plus_skin_sq, box);
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

template <typename T>
void launch_fill_cluster_pairs(ClusterBounds bounds,
                               const force_t* large_bounds, int num_clusters,
                               bool periodic, T cutoff_plus_skin_sq,
                               cuda::Box box, const int* row_ptr, int* col_idx,
                               cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int grid = ATV::cuda::grid_size(num_clusters * 32);
    int block = ATV::cuda::kDefaultBlock;
    if (box.ortho) {
        fill_pairs<true, false, T><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, row_ptr, col_idx, nullptr);
    } else {
        fill_pairs<false, false, T><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, row_ptr, col_idx, nullptr);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_count_cluster_pairs(ClusterBounds bounds,
                                const force_t* large_bounds, int num_clusters,
                                bool periodic, T cutoff_plus_skin_sq,
                                cuda::Box box, int* pair_counts,
                                cudaStream_t stream) {
    if (num_clusters <= 0)
        return;
    int grid = ATV::cuda::grid_size(num_clusters * 32);
    int block = ATV::cuda::kDefaultBlock;
    if (box.ortho) {
        fill_pairs<true, true, T><<<grid, block, 0, stream>>>(
            bounds, large_bounds, num_clusters, periodic, cutoff_plus_skin_sq,
            box, nullptr, nullptr, pair_counts);
    } else {
        fill_pairs<false, true, T><<<grid, block, 0, stream>>>(
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
    reinterpret_cast<const void*>(&launch_fill_cluster_pairs<force_t>);
[[maybe_unused]] const void* _inst_launch_count_cluster_pairs =
    reinterpret_cast<const void*>(&launch_count_cluster_pairs<force_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
