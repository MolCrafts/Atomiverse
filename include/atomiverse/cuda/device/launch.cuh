/// @file cuda/device/launch.cuh
/// @brief Kernel-author helpers: launch geometry + block-level reductions.
///
/// Single hub for the small device/host helpers that every CUDA kernel
/// in this codebase reaches for. Two flavors:
///
///   - **Launch geometry** (`kDefaultBlock`, `grid_size`) — host-only,
///     no CUDA runtime dependency beyond the include itself. Used at
///     kernel launch sites to size the grid.
///   - **Block reduction** (`block_reduce_sum`, `block_reduce_sum_n`) —
///     device-side, depends on `<cooperative_groups>`. Replaces the
///     legacy "extern __shared__ tree-reduction" pattern; each force /
///     KE kernel calls it to pack the per-thread energy + virial /
///     KE components into a single warp-shuffle-backed reduction.
///
/// Naming: helpers live in `ATV::cuda::` directly (the namespace itself
/// disambiguates them — e.g. `cuda::block_reduce_sum` is unambiguous
/// against CG-flavored `cuda` / `coarse-grain` concepts elsewhere in
/// the project).
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Default thread block size for 1D CUDA kernels.
///
/// 256 threads per block balances occupancy and register pressure on
/// Volta / Ampere / Ada / Hopper architectures. Changing this single
/// constant rescales every kernel in the codebase.
inline constexpr int kDefaultBlock = 256;

/// @brief CUDA warp size — fixed at 32 for every NVIDIA GPU generation
///        through Hopper. Named so kernel-launch sites don't have to
///        spell `>> 5` or `/ 32` repeatedly when sizing per-warp
///        scratch.
inline constexpr int kWarpSize = 32;

/// @brief Number of warps per thread block (ceil division).
__host__ __device__ inline int warps_per_block(int block = kDefaultBlock) {
    return (block + kWarpSize - 1) / kWarpSize;
}

/// @brief Compute 1D grid dimension for @p n_elements.
/// @return At least 1 block, even when @p n_elements is 0.
inline int grid_size(int n_elements, int block = kDefaultBlock) {
    int g = (n_elements + block - 1) / block;
    return g < 1 ? 1 : g;
}

// ─── Block-level sum reductions (device-side) ──────────────────────────
//
// Replaces the legacy "extern __shared__ work_t sdata[]" tree-reduction
// pattern with a warp-shuffle reduction. The legacy pattern took
// blockDim.x × sizeof(work_t) bytes of shared memory and log₂(blockDim.x)
// __syncthreads-gated stages per scalar; for the 6+1 = 7-scalar packed
// reduction in pair / bonded force kernels that meant ~48 barriers /
// 2 KiB per call.
//
// The warp-shuffle path here:
//   (1) `cooperative_groups::reduce` over a 32-lane warp tile —
//       compiles to a single `__reduce_add_sync` instruction on sm_80+
//       (GH200 is sm_90), or a shfl_xor fallback on older hardware.
//   (2) Lane 0 of each warp writes its partial to a small per-warp
//       shared scratch (≤ 8 cells for the 256-thread block).
//   (3) One `__syncthreads`. Warp 0 re-reduces the per-warp partials
//       via another `cg::reduce` call.
//
// Cost per scalar: 5 shuffle stages + 1 barrier + 1 cross-warp shuffle
// — independent of N scalars. Shared footprint: (blockDim.x / 32) ×
// sizeof(T) bytes total, regardless of how many scalars share the call.

/// @brief Sum-reduce one scalar across all threads in a thread block.
///
/// Returns the block-wide sum on thread 0; undefined on other threads
/// (caller typically only reads from thread 0 to emit an `atomicAdd`).
/// Caller provides one `__shared__` cell per warp (so the shared
/// footprint is `(blockDim.x / 32) * sizeof(T)` — ≤ 64 bytes for the
/// standard 256-thread block).
///
/// @tparam T Scalar type (typically `work_t` = double).
/// @param value Per-thread contribution.
/// @param warp_scratch Per-warp shared-memory scratch
///                     (≥ blockDim.x / 32 cells, reusable across calls).
/// @return Block-wide sum (valid only on thread 0).
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T value, T* warp_scratch) {
    namespace cg = cooperative_groups;
    auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    T warp_sum = cg::reduce(tile, value, cg::plus<T>{});

    int lane = tile.thread_rank();
    int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0)
        warp_scratch[warp_id] = warp_sum;
    __syncthreads();

    // Single warp at the front re-reduces the per-warp partials. The
    // staging buffer is small (≤ warps_per_block entries), so one warp
    // is more than enough — lanes past the warp count read zero.
    T block_sum = T(0);
    if (warp_id == 0) {
        int n_warps = warps_per_block(blockDim.x);
        T v = (lane < n_warps) ? warp_scratch[lane] : T(0);
        block_sum = cg::reduce(tile, v, cg::plus<T>{});
    }
    // Barrier before returning: warp 0's read of warp_scratch above must
    // complete on ALL lanes before any warp overwrites the scratch. Without
    // it, a repeated-use caller (block_reduce_sum_n loops over N scalars
    // reusing this same buffer) lets a non-zero warp race ahead to the next
    // scalar's `warp_scratch[warp_id] = ...` write while warp 0 is still
    // reading the previous scalar's partials — a write-after-read hazard that
    // corrupts the reduced energy/stress/virial (compute-sanitizer racecheck).
    __syncthreads();
    return block_sum;
}

/// @brief Max-reduce one scalar across all threads in a thread block.
///
/// Same warp-shuffle shape as `block_reduce_sum`, with `cg::greater`
/// in place of `cg::plus`. Returns the block-wide maximum on thread 0;
/// undefined on other threads. Out-of-range threads should contribute
/// the identity for the caller's domain (0 for magnitudes).
///
/// @tparam T Scalar type (typically `work_t`).
/// @param value Per-thread contribution.
/// @param warp_scratch Per-warp shared-memory scratch
///                     (≥ blockDim.x / 32 cells, reusable across calls).
/// @return Block-wide maximum (valid only on thread 0).
template <typename T>
__device__ __forceinline__ T block_reduce_max(T value, T* warp_scratch) {
    namespace cg = cooperative_groups;
    auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
    T warp_max = cg::reduce(tile, value, cg::greater<T>{});

    int lane = tile.thread_rank();
    int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0)
        warp_scratch[warp_id] = warp_max;
    __syncthreads();

    T block_max = T(0);
    if (warp_id == 0) {
        int n_warps = warps_per_block(blockDim.x);
        T v = (lane < n_warps) ? warp_scratch[lane] : T(0);
        block_max = cg::reduce(tile, v, cg::greater<T>{});
    }
    // Same write-after-read fence as block_reduce_sum: warp 0 must finish
    // reading warp_scratch before a repeated-use caller overwrites it.
    __syncthreads();
    return block_max;
}

/// @brief Atomic max for NON-NEGATIVE floating-point accumulators.
///
/// CUDA has no native floating-point `atomicMax`; for non-negative
/// values the IEEE-754 bit pattern is monotonic under integer
/// comparison, so the integer `atomicMax` on the reinterpreted bits is
/// exact. Callers must guarantee `val >= 0` and that the accumulator
/// was initialized to a non-negative value (a zeroed buffer is fine).
__device__ __forceinline__ void atomic_max_nonneg(double* addr, double val) {
    atomicMax(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(__double_as_longlong(val)));
}

/// @overload
__device__ __forceinline__ void atomic_max_nonneg(float* addr, float val) {
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
}

/// @brief Sum-reduce N scalars across all threads in a thread block.
///
/// Loops over scalars; each call shares the same `warp_scratch` buffer
/// (one cell per warp) and the final results land in `out[i]` on thread 0.
/// Each `block_reduce_sum` runs two `__syncthreads()` (one staging the
/// per-warp partials, one fencing the scratch read before the next scalar
/// reuses the buffer) — cost: 2N barriers, still far below the
/// N × log₂(blockDim) of a per-scalar shared-mem tree reduction.
///
/// @tparam N Compile-time number of scalars.
/// @tparam T Scalar type.
/// @param values Per-thread contributions (length N).
/// @param warp_scratch Per-warp shared-memory scratch
///                     (≥ blockDim.x / 32 cells).
/// @param out Per-block sums (length N; valid only on thread 0).
template <int N, typename T>
__device__ __forceinline__ void
block_reduce_sum_n(const T (&values)[N], T* warp_scratch, T (&out)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i)
        out[i] = block_reduce_sum<T>(values[i], warp_scratch);
}

} // namespace cuda
} // namespace ATV
