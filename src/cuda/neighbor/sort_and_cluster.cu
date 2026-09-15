/// @file cuda/neighbor/sort_and_cluster.cu
/// @brief Morton-code sorting and cluster grouping kernels.
#include "morton.cuh"
#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Kernel: AoS double* → SoA real_t* (x, y, z separate arrays)
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void aos_to_soa(const double* __restrict__ positions, int num_atoms,
                           real_t* __restrict__ x, real_t* __restrict__ y,
                           real_t* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;
    x[i] = static_cast<work_t>(positions[i * 3 + 0]);
    y[i] = static_cast<work_t>(positions[i * 3 + 1]);
    z[i] = static_cast<work_t>(positions[i * 3 + 2]);
}

template <typename work_t>
void launch_aos_to_soa(const double* positions_aos, int num_atoms, real_t* x,
                       real_t* y, real_t* z, cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    aos_to_soa<work_t>
        <<<grid, block, 0, stream>>>(positions_aos, num_atoms, x, y, z);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: Compute Morton codes from SoA positions
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void compute_morton(const real_t* __restrict__ pos_x,
                               const real_t* __restrict__ pos_y,
                               const real_t* __restrict__ pos_z, int num_atoms,
                               const real_t* __restrict__ inv_cell,
                               bool periodic,
                               uint32_t* __restrict__ morton_codes,
                               int* __restrict__ atom_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;

    work_t rx = pos_x[i], ry = pos_y[i], rz = pos_z[i];

    // Cartesian → fractional
    Vec3<work_t> frac = cart_to_frac(rx, ry, rz, inv_cell);

    if (periodic) {
        frac.x = wrap_frac(frac.x);
        frac.y = wrap_frac(frac.y);
        frac.z = wrap_frac(frac.z);
    } else {
        // For non-periodic, clamp to [0, 1)
        frac.x = clamp01(frac.x);
        frac.y = clamp01(frac.y);
        frac.z = clamp01(frac.z);
    }

    morton_codes[i] = morton_from_frac(frac.x, frac.y, frac.z);
    atom_indices[i] = i;
}

template <typename work_t>
void launch_compute_morton(const real_t* pos_x, const real_t* pos_y,
                           const real_t* pos_z, int num_atoms,
                           const real_t* inv_cell, bool periodic,
                           uint32_t* morton_codes, int* atom_indices,
                           cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    compute_morton<work_t>
        <<<grid, block, 0, stream>>>(pos_x, pos_y, pos_z, num_atoms, inv_cell,
                                     periodic, morton_codes, atom_indices);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CUB radix sort wrapper — temp storage owned by DeviceVector
// ---------------------------------------------------------------------------

void launch_radix_sort(uint32_t* keys_in, uint32_t* keys_out, int* values_in,
                       int* values_out, int num_items, DeviceVector<char>& temp,
                       cudaStream_t stream) {
    // Query required temp size
    size_t needed = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, needed, keys_in, keys_out,
                                    values_in, values_out, num_items, 0, 30,
                                    stream);
    if (needed > temp.capacity()) {
        temp.resize(needed);
    }
    size_t avail = temp.capacity();
    cub::DeviceRadixSort::SortPairs(temp.data(), avail, keys_in, keys_out,
                                    values_in, values_out, num_items, 0, 30,
                                    stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CUB temp byte query — used by NeighborList::build() to pre-allocate
// sort/scan temps so launch() never trips the resize-cudaMalloc path.
// ---------------------------------------------------------------------------

CubTempBytes query_cub_temp_bytes(int num_atoms, int num_clusters) {
    CubTempBytes out{0, 0, 0};
    if (num_atoms > 0) {
        // Radix sort of `num_atoms` Morton codes (uint32 keys + int values).
        uint32_t* dummy_keys = nullptr;
        int* dummy_values = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, out.sort_bytes, dummy_keys,
                                        dummy_keys, dummy_values, dummy_values,
                                        num_atoms, 0, 30, cudaStream_t{0});
        // Exclusive sum over `num_atoms` ints (largest scan path).
        int* dummy_in = nullptr;
        int* dummy_out = nullptr;
        cub::DeviceScan::ExclusiveSum(nullptr, out.scan_bytes_atoms, dummy_in,
                                      dummy_out, num_atoms, cudaStream_t{0});
    }
    if (num_clusters > 0) {
        // Exclusive sum over `num_clusters` ints (used by row_ptr build).
        int* dummy_in = nullptr;
        int* dummy_out = nullptr;
        cub::DeviceScan::ExclusiveSum(nullptr, out.scan_bytes_clusters,
                                      dummy_in, dummy_out, num_clusters,
                                      cudaStream_t{0});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Kernel: Scatter sorted atoms into cluster SoA layout
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void scatter_to_soa(
    const int* __restrict__ sorted_indices, const real_t* __restrict__ src_x,
    const real_t* __restrict__ src_y, const real_t* __restrict__ src_z,
    int num_atoms, int num_clusters, real_t* __restrict__ dst_x,
    real_t* __restrict__ dst_y, real_t* __restrict__ dst_z,
    int* __restrict__ original_index, int* __restrict__ cluster_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int padded = num_clusters * kClusterSize;

    if (i < num_atoms) {
        int old_idx = sorted_indices[i];
        dst_x[i] = src_x[old_idx];
        dst_y[i] = src_y[old_idx];
        dst_z[i] = src_z[old_idx];
        original_index[i] = old_idx;
    } else if (i < padded) {
        // Padding: set to huge value so AABB excludes them
        work_t huge = work_t(1e30);
        dst_x[i] = huge;
        dst_y[i] = huge;
        dst_z[i] = huge;
        original_index[i] = -1;
    }

    // Compute cluster_count: thread 0 of each cluster
    if (i < padded && (i % kClusterSize) == 0) {
        int ci = i / kClusterSize;
        int first = ci * kClusterSize;
        int last = first + kClusterSize;
        if (last > num_atoms)
            last = num_atoms;
        cluster_count[ci] = (last > first) ? (last - first) : 0;
    }
}

template <typename work_t>
void launch_scatter_to_soa(const int* sorted_indices, const real_t* pos_x,
                           const real_t* pos_y, const real_t* pos_z,
                           int num_atoms, int num_clusters, real_t* cluster_x,
                           real_t* cluster_y, real_t* cluster_z,
                           int* original_index, int* cluster_count,
                           cudaStream_t stream) {
    int padded = num_clusters * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (padded + block - 1) / block;
    scatter_to_soa<work_t><<<grid, block, 0, stream>>>(
        sorted_indices, pos_x, pos_y, pos_z, num_atoms, num_clusters, cluster_x,
        cluster_y, cluster_z, original_index, cluster_count);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Fused kernel: AoS→SoA + Morton code (replaces separate K1+K2)
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void
aos_to_soa_morton(const double* __restrict__ positions_aos, int num_atoms,
                  real_t* __restrict__ x, real_t* __restrict__ y,
                  real_t* __restrict__ z, const real_t* __restrict__ inv_cell,
                  bool periodic, uint32_t* __restrict__ morton_codes,
                  int* __restrict__ atom_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;

    // AoS double → SoA work_t
    work_t rx = static_cast<work_t>(positions_aos[i * 3 + 0]);
    work_t ry = static_cast<work_t>(positions_aos[i * 3 + 1]);
    work_t rz = static_cast<work_t>(positions_aos[i * 3 + 2]);
    x[i] = rx;
    y[i] = ry;
    z[i] = rz;

    // Morton code
    Vec3<work_t> frac = cart_to_frac(rx, ry, rz, inv_cell);

    if (periodic) {
        frac.x = wrap_frac(frac.x);
        frac.y = wrap_frac(frac.y);
        frac.z = wrap_frac(frac.z);
    } else {
        frac.x = clamp01(frac.x);
        frac.y = clamp01(frac.y);
        frac.z = clamp01(frac.z);
    }

    morton_codes[i] = morton_from_frac(frac.x, frac.y, frac.z);
    atom_indices[i] = i;
}

template <typename work_t>
void launch_aos_to_soa_morton(const double* positions_aos, int num_atoms,
                              real_t* x, real_t* y, real_t* z,
                              const real_t* inv_cell, bool periodic,
                              uint32_t* morton_codes, int* atom_indices,
                              cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    aos_to_soa_morton<work_t><<<grid, block, 0, stream>>>(
        positions_aos, num_atoms, x, y, z, inv_cell, periodic, morton_codes,
        atom_indices);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Fused kernel: Scatter to cluster SoA + AABB (replaces separate K4+K5)
// ---------------------------------------------------------------------------

template <typename work_t>
__global__ void scatter_and_aabb(
    const int* __restrict__ sorted_indices, const real_t* __restrict__ src_x,
    const real_t* __restrict__ src_y, const real_t* __restrict__ src_z,
    int num_atoms, int num_clusters, real_t* __restrict__ dst_x,
    real_t* __restrict__ dst_y, real_t* __restrict__ dst_z,
    int* __restrict__ original_index, int* __restrict__ cluster_count,
    real_t* __restrict__ min_x, real_t* __restrict__ min_y,
    real_t* __restrict__ min_z, real_t* __restrict__ max_x,
    real_t* __restrict__ max_y, real_t* __restrict__ max_z) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % kClusterSize;
    int ci = tid / kClusterSize;

    if (ci >= num_clusters)
        return;

    int i = ci * kClusterSize + lane;
    bool valid = (i < num_atoms);

    // Phase 1: Scatter
    work_t vx, vy, vz;
    work_t huge = work_t(1e30);

    if (valid) {
        int old_idx = sorted_indices[i];
        vx = src_x[old_idx];
        vy = src_y[old_idx];
        vz = src_z[old_idx];
        dst_x[i] = vx;
        dst_y[i] = vy;
        dst_z[i] = vz;
        original_index[i] = old_idx;
    } else {
        vx = huge;
        vy = huge;
        vz = huge;
        dst_x[i] = huge;
        dst_y[i] = huge;
        dst_z[i] = huge;
        original_index[i] = -1;
    }

    // Cluster count
    if (lane == 0) {
        int first = ci * kClusterSize;
        int last = first + kClusterSize;
        if (last > num_atoms)
            last = num_atoms;
        cluster_count[ci] = (last > first) ? (last - first) : 0;
    }

    // Phase 2: AABB via warp-shuffle reduction.
    // CS=32 ⇒ one cluster per full warp (mask 0xffffffff). Sub-warp grouping
    // is only needed if kClusterSize < 32 (historical CS=8).
    work_t lo_x = valid ? vx : huge;
    work_t lo_y = valid ? vy : huge;
    work_t lo_z = valid ? vz : huge;
    work_t hi_x = valid ? vx : -huge;
    work_t hi_y = valid ? vy : -huge;
    work_t hi_z = valid ? vz : -huge;

    static_assert(kClusterSize == 32 || (32 % kClusterSize) == 0,
                  "kClusterSize must divide 32 for sub-warp AABB");
    // OpenMM CS=32: one cluster = full warp. (If CS is ever <32, use a
    // dedicated sub-warp mask path — do not `1u << kClusterSize` when CS==32.)
    static_assert(kClusterSize == 32, "AABB reduction assumes OpenMM CS=32");
    unsigned mask = 0xffffffffu;
    for (int offset = kClusterSize / 2; offset >= 1; offset >>= 1) {
        work_t tmp;
        tmp = __shfl_down_sync(mask, lo_x, offset);
        lo_x = fmin(lo_x, tmp);
        tmp = __shfl_down_sync(mask, hi_x, offset);
        hi_x = fmax(hi_x, tmp);
        tmp = __shfl_down_sync(mask, lo_y, offset);
        lo_y = fmin(lo_y, tmp);
        tmp = __shfl_down_sync(mask, hi_y, offset);
        hi_y = fmax(hi_y, tmp);
        tmp = __shfl_down_sync(mask, lo_z, offset);
        lo_z = fmin(lo_z, tmp);
        tmp = __shfl_down_sync(mask, hi_z, offset);
        hi_z = fmax(hi_z, tmp);
    }

    if (lane == 0) {
        min_x[ci] = lo_x;
        min_y[ci] = lo_y;
        min_z[ci] = lo_z;
        max_x[ci] = hi_x;
        max_y[ci] = hi_y;
        max_z[ci] = hi_z;
    }
}

template <typename work_t>
void launch_scatter_and_aabb(const int* sorted_indices, const real_t* pos_x,
                             const real_t* pos_y, const real_t* pos_z,
                             int num_atoms, int num_clusters, real_t* cluster_x,
                             real_t* cluster_y, real_t* cluster_z,
                             int* original_index, int* cluster_count,
                             real_t* aabb_min_x, real_t* aabb_min_y,
                             real_t* aabb_min_z, real_t* aabb_max_x,
                             real_t* aabb_max_y, real_t* aabb_max_z,
                             cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int threads_needed = num_clusters * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;
    scatter_and_aabb<work_t><<<grid, block, 0, stream>>>(
        sorted_indices, pos_x, pos_y, pos_z, num_atoms, num_clusters, cluster_x,
        cluster_y, cluster_z, original_index, cluster_count, aabb_min_x,
        aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the configured work_t type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_aos_to_soa =
    reinterpret_cast<const void*>(&launch_aos_to_soa<work_t>);
[[maybe_unused]] const void* _inst_launch_compute_morton =
    reinterpret_cast<const void*>(&launch_compute_morton<work_t>);
[[maybe_unused]] const void* _inst_launch_scatter_to_soa =
    reinterpret_cast<const void*>(&launch_scatter_to_soa<work_t>);
[[maybe_unused]] const void* _inst_launch_aos_to_soa_morton =
    reinterpret_cast<const void*>(&launch_aos_to_soa_morton<work_t>);
[[maybe_unused]] const void* _inst_launch_scatter_and_aabb =
    reinterpret_cast<const void*>(&launch_scatter_and_aabb<work_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
