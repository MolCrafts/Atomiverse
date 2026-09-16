/// @file cuda/neighbor/pair_search.cu
/// @brief Stencil-based cluster pair search — single-pass with coarse count.
///
/// Pipeline: coarse_count → prefix_sum → fill(AABB) → prefix_sum → compact.
/// The expensive AABB traversal happens only ONCE (in the fill kernel).

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
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

// ---------------------------------------------------------------------------
// Kernel: Fill cluster pairs (single AABB pass)
//
// Writes pairs at row_ptr_upper[ci] offsets (pre-allocated from coarse count).
// Outputs actual pair_counts[ci] for subsequent exact prefix sum.
// Uses cluster_cell_idx to avoid recomputing home cell (P5).
// ---------------------------------------------------------------------------

// Templated on `bool Ortho` — propagates through to `min_aabb_dist_sq`
// so the triclinic branch dead-eliminates in the ortho specialization.
// Templated on `bool CountOnly`: when true the kernel runs the identical
// AABB-filtered stencil walk but writes ONLY pair_counts[ci] (no col_idx /
// row_ptr_upper touched) — the exact-count pre-pass that sizes the CSR.
// Running the SAME predicate in both passes guarantees the fill pass writes
// exactly row_ptr_upper[ci+1]-row_ptr_upper[ci] entries per cluster, so the
// CSR can never overflow. (The old coarse counter was an unfiltered
// stencil-cell sum — a wildly loose upper bound that overflowed int32 and
// wasted gigabytes at ~1M dense atoms.)
template <bool Ortho, bool CountOnly, typename work_t>
__global__ __launch_bounds__(ATV::cuda::kDefaultBlock) void fill_pairs(
    const int* __restrict__ cell_start, const int* __restrict__ cell_clusters,
    const int* __restrict__ cluster_cell_idx, int dims_x, int dims_y,
    int dims_z, const int* __restrict__ stencil_dx,
    const int* __restrict__ stencil_dy, const int* __restrict__ stencil_dz,
    const int* __restrict__ num_stencil_ptr,
    const real_t* __restrict__ aabb_min_x,
    const real_t* __restrict__ aabb_min_y,
    const real_t* __restrict__ aabb_min_z,
    const real_t* __restrict__ aabb_max_x,
    const real_t* __restrict__ aabb_max_y,
    const real_t* __restrict__ aabb_max_z, int num_clusters, bool periodic,
    work_t cutoff_plus_skin_sq, cuda::Box box,
    const int* __restrict__ row_ptr_upper, int* __restrict__ col_idx,
    int* __restrict__ pair_counts) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_clusters)
        return;

    int num_stencil = *num_stencil_ptr;
    int ci = warp_id;

    // Home cell from pre-computed cluster_cell_idx (P5: no cart_to_frac)
    int home = cluster_cell_idx[ci];
    int ncells_yz = dims_y * dims_z;
    int hx = home / ncells_yz;
    int hy = (home / dims_z) % dims_y;
    int hz = home % dims_z;

    int write_offset = CountOnly ? 0 : row_ptr_upper[ci];
    int count = 0;

    for (int s = 0; s < num_stencil; ++s) {
        int nx = hx + stencil_dx[s];
        int ny = hy + stencil_dy[s];
        int nz = hz + stencil_dz[s];

        if (periodic) {
            nx = ((nx % dims_x) + dims_x) % dims_x;
            ny = ((ny % dims_y) + dims_y) % dims_y;
            nz = ((nz % dims_z) + dims_z) % dims_z;
        } else {
            if (nx < 0 || nx >= dims_x || ny < 0 || ny >= dims_y || nz < 0
                || nz >= dims_z)
                continue;
        }

        int cell_idx = nx * ncells_yz + ny * dims_z + nz;
        int start = cell_start[cell_idx];
        int end = cell_start[cell_idx + 1];
        int num_cj = end - start;

        for (int batch = 0; batch < (num_cj + 31) / 32; ++batch) {
            int local = batch * 32 + lane;
            bool passes = false;
            int cj_val = -1;

            if (local < num_cj) {
                cj_val = cell_clusters[start + local];
                // FULL neighbor list: keep every neighbor cluster (cj < ci,
                // cj == ci, cj > ci), not just the cj >= ci upper triangle.
                // The atom-centric LJ force kernel owns each atom's force and
                // accumulates it in registers from ITS cluster's full neighbor
                // set, so it needs cj < ci entries too — in exchange it does
                // zero atomicAdds (vs the old half-list + 6-atomic Newton).
                work_t dist_sq = min_aabb_dist_sq<Ortho, work_t>(
                    aabb_min_x[ci], aabb_min_y[ci], aabb_min_z[ci],
                    aabb_max_x[ci], aabb_max_y[ci], aabb_max_z[ci],
                    aabb_min_x[cj_val], aabb_min_y[cj_val], aabb_min_z[cj_val],
                    aabb_max_x[cj_val], aabb_max_y[cj_val], aabb_max_z[cj_val],
                    periodic, box);
                passes = (dist_sq <= cutoff_plus_skin_sq);
            }

            unsigned ballot = __ballot_sync(0xFFFFFFFF, passes);
            int batch_count = __popc(ballot);

            if (!CountOnly) {
                int prefix = __popc(ballot & ((1u << lane) - 1));
                if (passes) {
                    col_idx[write_offset + prefix] = cj_val;
                }
                write_offset += batch_count;
            }
            count += batch_count;
        }
    }

    if (lane == 0) {
        pair_counts[ci] = count;
    }
}

template <typename work_t>
void launch_fill_cluster_pairs(
    const int* cell_start, const int* cell_clusters,
    const int* cluster_cell_idx, int dims_x, int dims_y, int dims_z,
    const int* stencil_dx, const int* stencil_dy, const int* stencil_dz,
    const int* num_stencil_ptr, const real_t* aabb_min_x,
    const real_t* aabb_min_y, const real_t* aabb_min_z,
    const real_t* aabb_max_x, const real_t* aabb_max_y,
    const real_t* aabb_max_z, int num_clusters, bool periodic,
    work_t cutoff_plus_skin_sq, cuda::Box box, const int* row_ptr_upper,
    int* col_idx, int* pair_counts, cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int threads_needed = num_clusters * 32;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;
    if (box.ortho) {
        fill_pairs<true, false, work_t><<<grid, block, 0, stream>>>(
            cell_start, cell_clusters, cluster_cell_idx, dims_x, dims_y, dims_z,
            stencil_dx, stencil_dy, stencil_dz, num_stencil_ptr, aabb_min_x,
            aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z,
            num_clusters, periodic, cutoff_plus_skin_sq, box, row_ptr_upper,
            col_idx, pair_counts);
    } else {
        fill_pairs<false, false, work_t><<<grid, block, 0, stream>>>(
            cell_start, cell_clusters, cluster_cell_idx, dims_x, dims_y, dims_z,
            stencil_dx, stencil_dy, stencil_dz, num_stencil_ptr, aabb_min_x,
            aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z,
            num_clusters, periodic, cutoff_plus_skin_sq, box, row_ptr_upper,
            col_idx, pair_counts);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// Exact-count pre-pass: same AABB-filtered stencil walk as the fill, writing
// only pair_counts[ci]. Its prefix sum (row_ptr_upper) is therefore the EXACT
// number of pairs the fill will emit — never an overflowing loose bound.
template <typename work_t>
void launch_count_cluster_pairs(
    const int* cell_start, const int* cell_clusters,
    const int* cluster_cell_idx, int dims_x, int dims_y, int dims_z,
    const int* stencil_dx, const int* stencil_dy, const int* stencil_dz,
    const int* num_stencil_ptr, const real_t* aabb_min_x,
    const real_t* aabb_min_y, const real_t* aabb_min_z,
    const real_t* aabb_max_x, const real_t* aabb_max_y,
    const real_t* aabb_max_z, int num_clusters, bool periodic,
    work_t cutoff_plus_skin_sq, cuda::Box box, int* pair_counts,
    cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int threads_needed = num_clusters * 32;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;
    if (box.ortho) {
        fill_pairs<true, true, work_t><<<grid, block, 0, stream>>>(
            cell_start, cell_clusters, cluster_cell_idx, dims_x, dims_y, dims_z,
            stencil_dx, stencil_dy, stencil_dz, num_stencil_ptr, aabb_min_x,
            aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z,
            num_clusters, periodic, cutoff_plus_skin_sq, box, /*row_ptr_upper*/
            nullptr, /*col_idx*/ nullptr, pair_counts);
    } else {
        fill_pairs<false, true, work_t><<<grid, block, 0, stream>>>(
            cell_start, cell_clusters, cluster_cell_idx, dims_x, dims_y, dims_z,
            stencil_dx, stencil_dy, stencil_dz, num_stencil_ptr, aabb_min_x,
            aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z,
            num_clusters, periodic, cutoff_plus_skin_sq, box, /*row_ptr_upper*/
            nullptr, /*col_idx*/ nullptr, pair_counts);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Write row_ptr sentinel on device (avoids D→H readback)
// ---------------------------------------------------------------------------

__global__ void write_row_ptr_sentinel(int* __restrict__ row_ptr,
                                       const int* __restrict__ pair_counts,
                                       int num_clusters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        row_ptr[num_clusters] =
            row_ptr[num_clusters - 1] + pair_counts[num_clusters - 1];
    }
}

void launch_write_row_ptr_sentinel(int* row_ptr, const int* pair_counts,
                                   int num_clusters, cudaStream_t stream) {
    write_row_ptr_sentinel<<<1, 1, 0, stream>>>(row_ptr, pair_counts,
                                                num_clusters);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: In-place compaction of col_idx from upper-bound to exact CSR layout.
// Safe because row_ptr[ci] <= row_ptr_upper[ci] for all ci (exact count <=
// coarse count implies smaller prefix sums).
// ---------------------------------------------------------------------------

__global__ void compact_pairs(int* __restrict__ col_idx,
                              const int* __restrict__ row_ptr_upper,
                              const int* __restrict__ row_ptr,
                              const int* __restrict__ pair_counts,
                              int num_clusters) {
    // One warp per cluster for parallel copy
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_clusters)
        return;

    int ci = warp_id;
    int src = row_ptr_upper[ci];
    int dst = row_ptr[ci];
    int n = pair_counts[ci];

    // Warp-cooperative copy: each lane handles elements at stride 32
    for (int i = lane; i < n; i += 32) {
        col_idx[dst + i] = col_idx[src + i];
    }
}

void launch_compact_pairs(int* col_idx, const int* row_ptr_upper,
                          const int* row_ptr, const int* pair_counts,
                          int num_clusters, cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int threads_needed = num_clusters * 32;
    int grid = (threads_needed + block - 1) / block;
    compact_pairs<<<grid, block, 0, stream>>>(col_idx, row_ptr_upper, row_ptr,
                                              pair_counts, num_clusters);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the configured work_t type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_fill_cluster_pairs =
    reinterpret_cast<const void*>(&launch_fill_cluster_pairs<work_t>);
[[maybe_unused]] const void* _inst_launch_count_cluster_pairs =
    reinterpret_cast<const void*>(&launch_count_cluster_pairs<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
