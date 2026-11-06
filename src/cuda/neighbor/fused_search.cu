/// @file cuda/neighbor/fused_search.cu
/// @brief Rebuild-time CSR masks and exclusion baking, and the per-step skin
///        gate that decides a rebuild on the device.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/core/neighbor_check.h"

#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ATV::cuda::md::nlist {

namespace {
// Device-wide publication ids survive provider replacement and allocator
// address reuse. Zero denotes a list whose masks have not been built.
__device__ unsigned long long g_mask_generation = 0;
} // namespace

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
// mask layout remain ATV-native; no OpenMM source is copied. The live entry
// count is read on the device and the grid strides over it, so a rebuild
// never reports its size to the host.
template <bool Ortho, typename T>
__global__ __launch_bounds__(128) void compute_csr_masks_k(
    const pos_t* __restrict__ cluster_x, const pos_t* __restrict__ cluster_y,
    const pos_t* __restrict__ cluster_z, const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx, int num_clusters, T search_sq,
    cuda::Box box_arg, uint64_t* __restrict__ masks) {
    const auto box = box_arg.current().as<T>();
    static_assert(kClusterSize == 32, "CSR mask kernel assumes CS=32");
    const int warps_per_block = blockDim.x / kClusterSize;
    const int lane = threadIdx.x % kClusterSize;
    const int num_pairs = row_ptr[num_clusters];
    // Uniform within the warp; no block collective follows.
    for (int p = blockIdx.x * warps_per_block + threadIdx.x / kClusterSize;
         p < num_pairs; p += gridDim.x * warps_per_block) {
        int ci = 0;
        if (lane == 0) {
            ci = static_cast<int>(
                thrust::upper_bound(thrust::seq, row_ptr + 1,
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
        // Exclude the same atom on diagonal tiles exactly as the row kernel
        // did.
        if (ci == cj)
            bits &= ~(1u << lane);
        mask_set_jbits(masks, p, lane, bits);
    }
}

template <typename T>
void launch_compute_csr_masks(const pos_t* cluster_x, const pos_t* cluster_y,
                              const pos_t* cluster_z, const int* row_ptr,
                              const int* col_idx, int num_clusters,
                              int pair_capacity, T search_sq, cuda::Box box,
                              uint64_t* masks, cudaStream_t stream) {
    if (num_clusters <= 0 || pair_capacity <= 0)
        return;
    constexpr int threads = 128;
    constexpr int tiles_per_block = threads / kClusterSize;
    // Half the reserve per pass: the default reserve is twice the measured
    // count, so a typical rebuild is one stride and an outgrown one two.
    const long long tiles = (static_cast<long long>(pair_capacity) + 1) / 2;
    const int grid = static_cast<int>(
        std::max(1LL, (tiles + tiles_per_block - 1) / tiles_per_block));
    if (box.ortho) {
        compute_csr_masks_k<true, T><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            search_sq, box, masks);
    } else {
        compute_csr_masks_k<false, T><<<grid, threads, 0, stream>>>(
            cluster_x, cluster_y, cluster_z, row_ptr, col_idx, num_clusters,
            search_sq, box, masks);
    }
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Skin gate: current periodic motion against the rebuild reference, decided
// on the device. Blocks fold their verdict into `control->motion`; the last
// block (ticket counter) consumes motion and the forced flag, publishes the
// decision and selects the conditional rebuild body. No host read.
//
// Box change follows LAMMPS Neighbor::check_distance (src/neighbor.cpp,
// SHA e8c4c278a8599165a0321ebfcb4e4046ef79ad79; Copyright (2003) Sandia
// Corporation, GPL-2.0-only): the half-skin trigger shrinks by the motion
// of the cell corners since the reference build (`core/neighbor_check.h`),
// and a cell change that alone consumes the skin rebuilds. A changing cell
// therefore no longer rebuilds on every step.
// ---------------------------------------------------------------------------
// The trigger is computed once per block in FP64 from the FP64 cells; the
// motion test runs on the primary coordinates against a cell narrowed once
// to the scratch type.
template <typename T>
constexpr T kInfinity = std::numeric_limits<T>::infinity();
template <typename T>
constexpr T kEpsilon = std::numeric_limits<T>::epsilon();

// OpenMM's singlePeriodicCopy criterion from one cluster's live coordinates
// (one warp = one cluster; every lane participates, pad lanes as `live =
// false`). A reused Hilbert cluster then needs no assumption about its
// motion within the skin; wide clusters keep the per-pair minimum image.
template <typename T>
__device__ __forceinline__ void
publish_cluster_image(bool live, T x, T y, T z, const BoxGeometry<T>& box,
                      T cutoff, ClusterImage* images, int ci) {
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    constexpr T inf = kInfinity<T>;
    constexpr T eps = kEpsilon<T>;
    ClusterImage image{};
    if (box.ortho) {
        const T position[3] = {x, y, z};
        bool single = true;
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            const T lo =
                cg::reduce(warp, live ? position[d] : inf, cg::less<T>{});
            const T hi =
                cg::reduce(warp, live ? position[d] : -inf, cg::greater<T>{});
            const T length = T(box.cell[4 * d]);
            const T halfwidth = (hi - lo) * T(0.5);
            const T margin =
                T(32) * eps * (fabs(lo) + fabs(hi) + fabs(length) + cutoff);
            // Every cutoff neighbour then lies strictly inside the half cell
            // centred here. Warp-uniform, including pad lanes.
            single = single && halfwidth + cutoff + margin < length * T(0.5);
            image.center[d] = force_t(lo + halfwidth);
        }
        image.single = single ? 1 : 0;
    }
    if (warp.thread_rank() == 0)
        images[ci] = image;
}

// One thread per cluster slot: read the slot's atom once, publish it as the
// live cluster position and fold it into its cluster's image. With `Gate`
// the same read drives the skin test against the slot-ordered reference of
// the last rebuild — every block reduces "an atom moved more than the LAMMPS
// check_distance trigger (`core/neighbor_check.h`; a cell change that alone
// consumes the skin rebuilds)" and the last block sets the conditional
// rebuild handle. The trigger is computed once per block in FP64 from the
// FP64 cells; the motion test runs on the primary coordinates against a
// cell narrowed once to the scratch type.
template <typename T, bool Gate>
__global__ void
skin_gate_k(const int* __restrict__ orig_index, int num_slots,
            const pos_t* __restrict__ pos_x, const pos_t* __restrict__ pos_y,
            const pos_t* __restrict__ pos_z, const pos_t* __restrict__ ref_x,
            const pos_t* __restrict__ ref_y, const pos_t* __restrict__ ref_z,
            pos_t* __restrict__ cl_x, pos_t* __restrict__ cl_y,
            pos_t* __restrict__ cl_z, ClusterImage* __restrict__ images,
            T image_cut, double skin, Box box_arg, const Box* reference_box,
            Control* control, cudaGraphConditionalHandle handle) {
    const Box geom = box_arg.current();
    const auto box = geom.as<T>();
    __shared__ double trigger_sq;
    if constexpr (Gate) {
        if (threadIdx.x == 0)
            trigger_sq = reference_box
                             ? ATV::md::skin_trigger_sq(reference_box->cell,
                                                        geom.cell, skin)
                             : 0.25 * skin * skin;
        __syncthreads();
    }
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = k < num_slots ? orig_index[k] : -1;
    T x = 0, y = 0, z = 0;
    if (a >= 0) {
        x = pos_x[a];
        y = pos_y[a];
        z = pos_z[a];
        cl_x[k] = pos_t(x);
        cl_y[k] = pos_t(y);
        cl_z[k] = pos_t(z);
    }
    // Slots are whole clusters and blocks whole warps: a warp is either a
    // full cluster or entirely past the end.
    if (k - static_cast<int>(threadIdx.x % 32) < num_slots)
        publish_cluster_image<T>(a >= 0, x, y, z, box, image_cut, images,
                                 k / 32);
    if constexpr (Gate) {
        bool moved = trigger_sq < 0.0;
        if (a >= 0 && !moved) {
            T dx = x - ref_x[k];
            T dy = y - ref_y[k];
            T dz = z - ref_z[k];
            box.min_image(dx, dy, dz);
            moved = dx * dx + dy * dy + dz * dz > T(trigger_sq);
        }
        const int count = __syncthreads_count(moved);
        if (threadIdx.x != 0)
            return;
        if (count != 0)
            atomicOr(&control->motion, 1);
        __threadfence();
        // atomicInc wraps to zero on the last ticket, re-arming the next step.
        if (atomicInc(&control->done, gridDim.x - 1) != gridDim.x - 1)
            return;
        const bool rebuild =
            (atomicExch(&control->motion, 0) | atomicExch(&control->force, 0))
            != 0;
        control->rebuilt = rebuild;
        if (rebuild) {
            ++control->rebuilds;
            control->generation = atomicAdd(&g_mask_generation, 1ULL) + 1ULL;
        }
        cudaGraphSetConditional(handle, rebuild ? 1u : 0u);
    }
}

template <typename T>
void launch_skin_gate(const ClusterSlots& slots, const pos_t* pos_x,
                      const pos_t* pos_y, const pos_t* pos_z, double skin,
                      Box box, const Box* reference_box, Control* control,
                      cudaGraphConditionalHandle handle, cudaStream_t stream) {
    const int grid = ATV::cuda::grid_size(std::max(slots.num_slots, 1), 256);
    skin_gate_k<T, true><<<grid, 256, 0, stream>>>(
        slots.orig_index, slots.num_slots, pos_x, pos_y, pos_z, slots.ref_x,
        slots.ref_y, slots.ref_z, slots.cl_x, slots.cl_y, slots.cl_z,
        slots.images, T(slots.image_cut), skin, box, reference_box, control,
        handle);
    ATV_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_refresh_clusters(const ClusterSlots& slots, const pos_t* pos_x,
                             const pos_t* pos_y, const pos_t* pos_z, Box box,
                             cudaStream_t stream) {
    if (slots.num_slots <= 0)
        return;
    const int grid = ATV::cuda::grid_size(slots.num_slots, 256);
    skin_gate_k<T, false><<<grid, 256, 0, stream>>>(
        slots.orig_index, slots.num_slots, pos_x, pos_y, pos_z, nullptr,
        nullptr, nullptr, slots.cl_x, slots.cl_y, slots.cl_z, slots.images,
        T(slots.image_cut), 0.0, box, nullptr, nullptr, {});
    ATV_CUDA_CHECK(cudaGetLastError());
}

__global__ void publish_generation_k(Control* control) {
    control->rebuilt = 1;
    ++control->rebuilds;
    control->generation = atomicAdd(&g_mask_generation, 1ULL) + 1ULL;
}

void launch_publish_generation(Control* control, cudaStream_t stream) {
    publish_generation_k<<<1, 1, 0, stream>>>(control);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Snapshot the rebuilt slot positions into ref_{x,y,z}
// ---------------------------------------------------------------------------

void launch_snapshot_ref_slots(const pos_t* pos_x, const pos_t* pos_y,
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
[[maybe_unused]] const void* _inst_launch_compute_csr_masks =
    reinterpret_cast<const void*>(&launch_compute_csr_masks<force_t>);
[[maybe_unused]] const void* _inst_launch_skin_gate =
    reinterpret_cast<const void*>(&launch_skin_gate<force_t>);
[[maybe_unused]] const void* _inst_launch_refresh_clusters =
    reinterpret_cast<const void*>(&launch_refresh_clusters<force_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
