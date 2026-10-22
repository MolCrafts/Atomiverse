/// @file cuda/neighbor/sort_and_cluster.cu
/// @brief Hilbert-key sorting and cluster grouping kernels.
#include "hilbert.cuh"
#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ATV::cuda::md::nlist {

// ---------------------------------------------------------------------------
// Kernel: Compute Hilbert keys from SoA positions
// ---------------------------------------------------------------------------

template <typename T>
__global__ void compute_spatial_keys(
    const pos_t* __restrict__ pos_x, const pos_t* __restrict__ pos_y,
    const pos_t* __restrict__ pos_z, int num_atoms,
    const force_t* __restrict__ inv_cell, bool periodic,
    uint32_t* __restrict__ spatial_keys, int* __restrict__ atom_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms)
        return;

    T rx = pos_x[i], ry = pos_y[i], rz = pos_z[i];

    // Cartesian → fractional
    Vec3<T> frac = cart_to_frac(rx, ry, rz, inv_cell);

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

    spatial_keys[i] = hilbert_from_frac(frac.x, frac.y, frac.z);
    atom_indices[i] = i;
}

template <typename T>
void launch_compute_spatial_keys(const pos_t* pos_x, const pos_t* pos_y,
                                 const pos_t* pos_z, int num_atoms,
                                 const force_t* inv_cell, bool periodic,
                                 uint32_t* spatial_keys, int* atom_indices,
                                 cudaStream_t stream) {
    if (num_atoms == 0)
        return;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (num_atoms + block - 1) / block;
    compute_spatial_keys<T>
        <<<grid, block, 0, stream>>>(pos_x, pos_y, pos_z, num_atoms, inv_cell,
                                     periodic, spatial_keys, atom_indices);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// CUB radix sort wrapper — temp storage owned by DeviceVector
// ---------------------------------------------------------------------------

void launch_exclusive_sum(const int* input, int* output, int num_items,
                          DeviceVector<char>& temp, cudaStream_t stream) {
    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, input, output, num_items,
                                  stream);
    if (needed > temp.capacity()) {
        throw std::runtime_error(
            "NeighborList: CUB scan scratch exceeds build-time capacity");
    }
    size_t avail = temp.capacity();
    cub::DeviceScan::ExclusiveSum(temp.data(), avail, input, output, num_items,
                                  stream);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void launch_radix_sort(uint32_t* keys_in, uint32_t* keys_out, int* values_in,
                       int* values_out, int num_items, DeviceVector<char>& temp,
                       cudaStream_t stream) {
    // Query required temp size
    size_t needed = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, needed, keys_in, keys_out,
                                    values_in, values_out, num_items, 0, 30,
                                    stream);
    if (needed > temp.capacity()) {
        throw std::runtime_error(
            "NeighborList: CUB sort scratch exceeds build-time capacity");
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
        // Radix sort of `num_atoms` Hilbert keys (uint32 keys + int values).
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
// Fused kernel: Scatter to cluster SoA + AABB (replaces separate K4+K5)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void scatter_and_aabb(
    const int* __restrict__ sorted_indices, const pos_t* __restrict__ src_x,
    const pos_t* __restrict__ src_y, const pos_t* __restrict__ src_z,
    int num_atoms, int num_clusters, pos_t* __restrict__ dst_x,
    pos_t* __restrict__ dst_y, pos_t* __restrict__ dst_z,
    int* __restrict__ original_index, int* __restrict__ cluster_count,
    force_t* __restrict__ min_x, force_t* __restrict__ min_y,
    force_t* __restrict__ min_z, force_t* __restrict__ max_x,
    force_t* __restrict__ max_y, force_t* __restrict__ max_z) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % kClusterSize;
    int ci = tid / kClusterSize;

    if (ci >= num_clusters)
        return;

    int i = ci * kClusterSize + lane;
    bool valid = (i < num_atoms);

    // Phase 1: Scatter
    T vx, vy, vz;
    T huge = T(1e30);

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

    // Phase 2: AABB via Cooperative Groups warp reductions.
    // CS=32 ⇒ one cluster per full warp (mask 0xffffffff). Sub-warp grouping
    // is only needed if kClusterSize < 32 (historical CS=8).
    T lo_x = valid ? vx : huge;
    T lo_y = valid ? vy : huge;
    T lo_z = valid ? vz : huge;
    T hi_x = valid ? vx : -huge;
    T hi_y = valid ? vy : -huge;
    T hi_z = valid ? vz : -huge;

    static_assert(kClusterSize == 32 || (32 % kClusterSize) == 0,
                  "kClusterSize must divide 32 for sub-warp AABB");
    // OpenMM CS=32: one cluster = full warp. (If CS is ever <32, use a
    // dedicated sub-warp mask path — do not `1u << kClusterSize` when CS==32.)
    static_assert(kClusterSize == 32, "AABB reduction assumes OpenMM CS=32");
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    lo_x = cg::reduce(warp, lo_x, cg::less<T>{});
    hi_x = cg::reduce(warp, hi_x, cg::greater<T>{});
    lo_y = cg::reduce(warp, lo_y, cg::less<T>{});
    hi_y = cg::reduce(warp, hi_y, cg::greater<T>{});
    lo_z = cg::reduce(warp, lo_z, cg::less<T>{});
    hi_z = cg::reduce(warp, hi_z, cg::greater<T>{});

    if (lane == 0) {
        min_x[ci] = lo_x;
        min_y[ci] = lo_y;
        min_z[ci] = lo_z;
        max_x[ci] = hi_x;
        max_y[ci] = hi_y;
        max_z[ci] = hi_z;
    }
}

template <typename T>
void launch_scatter_and_aabb(const int* sorted_indices, const pos_t* pos_x,
                             const pos_t* pos_y, const pos_t* pos_z,
                             int num_atoms, int num_clusters, pos_t* cluster_x,
                             pos_t* cluster_y, pos_t* cluster_z,
                             int* original_index, int* cluster_count,
                             force_t* aabb_min_x, force_t* aabb_min_y,
                             force_t* aabb_min_z, force_t* aabb_max_x,
                             force_t* aabb_max_y, force_t* aabb_max_z,
                             cudaStream_t stream) {
    if (num_clusters == 0)
        return;
    int threads_needed = num_clusters * kClusterSize;
    int block = ATV::cuda::kDefaultBlock;
    int grid = (threads_needed + block - 1) / block;
    scatter_and_aabb<T><<<grid, block, 0, stream>>>(
        sorted_indices, pos_x, pos_y, pos_z, num_atoms, num_clusters, cluster_x,
        cluster_y, cluster_z, original_index, cluster_count, aabb_min_x,
        aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── Forced template instantiations ───────────────────────────────────
// Instantiate each launcher template over the force-path scratch type
// (the only argument the call sites use). Address-of ODR-uses the
// template, so the compiler-deduced signature is instantiated without
// a hand-written parameter list that could drift out of sync.
namespace {
[[maybe_unused]] const void* _inst_launch_compute_spatial_keys =
    reinterpret_cast<const void*>(&launch_compute_spatial_keys<force_t>);
[[maybe_unused]] const void* _inst_launch_scatter_and_aabb =
    reinterpret_cast<const void*>(&launch_scatter_and_aabb<force_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
