/// @file cuda/neighbor/fused_search.cu
/// @brief Rebuild-time CSR masks and exclusion baking, and the per-step skin
///        gate that decides a rebuild on the device.

#include "nlist_impl.cuh"
#include "triclinic.cuh"

#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cstdint>

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
// ---------------------------------------------------------------------------
// The cell-change test compares the FP64 cells exactly; the motion test runs
// on the primary coordinates against a cell narrowed once to the scratch type.
template <typename T>
__global__ void
skin_gate_k(const pos_t* __restrict__ pos_x, const pos_t* __restrict__ pos_y,
            const pos_t* __restrict__ pos_z, const pos_t* __restrict__ ref_x,
            const pos_t* __restrict__ ref_y, const pos_t* __restrict__ ref_z,
            int num_atoms, T half_skin_sq, Box box_arg,
            const Box* reference_box, Control* control,
            cudaGraphConditionalHandle handle) {
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
    if (threadIdx.x != 0)
        return;
    if (count != 0)
        atomicOr(&control->motion, 1);
    __threadfence();
    // atomicInc wraps to zero on the last ticket, re-arming the next step.
    if (atomicInc(&control->done, gridDim.x - 1) != gridDim.x - 1)
        return;
    const bool rebuild =
        (atomicExch(&control->motion, 0) | atomicExch(&control->force, 0)) != 0;
    control->rebuilt = rebuild;
    if (rebuild) {
        ++control->rebuilds;
        control->generation = atomicAdd(&g_mask_generation, 1ULL) + 1ULL;
    }
    cudaGraphSetConditional(handle, rebuild ? 1u : 0u);
}

template <typename T>
void launch_skin_gate(const pos_t* pos_x, const pos_t* pos_y,
                      const pos_t* pos_z, const pos_t* ref_x,
                      const pos_t* ref_y, const pos_t* ref_z, int num_atoms,
                      T half_skin_sq, Box box, const Box* reference_box,
                      Control* control, cudaGraphConditionalHandle handle,
                      cudaStream_t stream) {
    const int grid = ATV::cuda::grid_size(num_atoms, 256);
    skin_gate_k<T><<<grid, 256, 0, stream>>>(
        pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, num_atoms, half_skin_sq, box,
        reference_box, control, handle);
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
[[maybe_unused]] const void* _inst_launch_compute_csr_masks =
    reinterpret_cast<const void*>(&launch_compute_csr_masks<force_t>);
[[maybe_unused]] const void* _inst_launch_skin_gate =
    reinterpret_cast<const void*>(&launch_skin_gate<force_t>);
} // namespace

} // namespace ATV::cuda::md::nlist
