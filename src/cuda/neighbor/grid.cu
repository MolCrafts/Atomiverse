/// @file grid.cu
/// @brief Cluster spatial grid construction (histogram + scan + scatter).
#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Kernel: Assign each cluster to a grid cell based on its AABB centroid.
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void
assign_clusters_to_cells(const real_t* __restrict__ aabb_min_x,
                         const real_t* __restrict__ aabb_min_y,
                         const real_t* __restrict__ aabb_min_z,
                         const real_t* __restrict__ aabb_max_x,
                         const real_t* __restrict__ aabb_max_y,
                         const real_t* __restrict__ aabb_max_z,
                         int num_clusters, const real_t* __restrict__ inv_cell,
                         int dims_x, int dims_y, int dims_z, bool periodic,
                         int* __restrict__ cluster_cell_idx) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= num_clusters)
        return;

    // Centroid in Cartesian
    work_t cx = (aabb_min_x[ci] + aabb_max_x[ci]) * work_t(0.5);
    work_t cy = (aabb_min_y[ci] + aabb_max_y[ci]) * work_t(0.5);
    work_t cz = (aabb_min_z[ci] + aabb_max_z[ci]) * work_t(0.5);

    // Cartesian → fractional
    Vec3<work_t> frac = cart_to_frac(cx, cy, cz, inv_cell);

    // Wrap or clamp fractional coordinates
    if (periodic) {
        frac.x = wrap_frac(frac.x);
        frac.y = wrap_frac(frac.y);
        frac.z = wrap_frac(frac.z);
    } else {
        frac.x = clamp01(frac.x);
        frac.y = clamp01(frac.y);
        frac.z = clamp01(frac.z);
    }

    // Grid indices
    int ix = min(max(int(frac.x * dims_x), 0), dims_x - 1);
    int iy = min(max(int(frac.y * dims_y), 0), dims_y - 1);
    int iz = min(max(int(frac.z * dims_z), 0), dims_z - 1);

    cluster_cell_idx[ci] = ix * dims_y * dims_z + iy * dims_z + iz;
}

template <typename work_t>
void launch_assign_clusters_to_cells(
    const real_t* aabb_min_x, const real_t* aabb_min_y,
    const real_t* aabb_min_z, const real_t* aabb_max_x,
    const real_t* aabb_max_y, const real_t* aabb_max_z, int num_clusters,
    const real_t* inv_cell, int dims_x, int dims_y, int dims_z, bool periodic,
    int* cluster_cell_idx, cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_clusters + block - 1) / block;
    assign_clusters_to_cells<work_t><<<grid, block, 0, stream>>>(
        aabb_min_x, aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z,
        num_clusters, inv_cell, dims_x, dims_y, dims_z, periodic,
        cluster_cell_idx);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CUB exclusive sum wrapper — temp storage owned by DeviceVector
// ---------------------------------------------------------------------------

void launch_exclusive_sum(const int* input, int* output, int num_items,
                          DeviceVector<char>& temp, cudaStream_t stream) {
    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, input, output, num_items,
                                  stream);
    if (needed > temp.capacity()) {
        temp.resize(needed);
    }
    size_t avail = temp.capacity();
    cub::DeviceScan::ExclusiveSum(temp.data(), avail, input, output, num_items,
                                  stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Build grid CSR: histogram → prefix sum → scatter
// ---------------------------------------------------------------------------

__global__ void build_histogram(const int* __restrict__ cell_idx,
                                int num_clusters, int* __restrict__ hist) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci < num_clusters) {
        atomicAdd(&hist[cell_idx[ci]], 1);
    }
}

__global__ void scatter_to_grid(const int* __restrict__ cell_idx,
                                int num_clusters,
                                const int* __restrict__ cell_start,
                                int* __restrict__ cell_clusters,
                                int* __restrict__ counter) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci < num_clusters) {
        int cell = cell_idx[ci];
        int pos = cell_start[cell] + atomicAdd(&counter[cell], 1);
        cell_clusters[pos] = ci;
    }
}

void launch_build_grid_csr(const int* cluster_cell_idx, int num_clusters,
                           int ncells, int* cell_start, int* cell_clusters,
                           int* histogram, DeviceVector<char>& scan_temp,
                           cudaStream_t stream) {
    if (num_clusters == 0 || ncells == 0)
        return;

    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_clusters + block - 1) / block;

    // Step 1: Zero histogram
    ATV_CUDA_CHECK(cudaMemsetAsync(histogram, 0, ncells * sizeof(int), stream));

    // Step 2: Build histogram
    build_histogram<<<grid, block, 0, stream>>>(cluster_cell_idx, num_clusters,
                                                histogram);
    ATV_CUDA_CHECK(cudaGetLastError());

    // Step 3: Prefix sum → cell_start[0..ncells-1]
    launch_exclusive_sum(histogram, cell_start, ncells, scan_temp, stream);

    // cell_start[ncells] = num_clusters (total count, written on device)
    launch_write_int(cell_start, ncells, num_clusters, stream);

    // Step 4: Scatter clusters into grid-sorted order
    // Re-use histogram as atomic counter (zero it first)
    ATV_CUDA_CHECK(cudaMemsetAsync(histogram, 0, ncells * sizeof(int), stream));
    scatter_to_grid<<<grid, block, 0, stream>>>(
        cluster_cell_idx, num_clusters, cell_start, cell_clusters, histogram);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the configured work_t type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_assign_clusters_to_cells =
    reinterpret_cast<const void*>(&launch_assign_clusters_to_cells<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
