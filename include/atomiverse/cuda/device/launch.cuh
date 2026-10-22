/// @file cuda/device/launch.cuh
/// @brief Launch geometry and component-wise reduction operator for CUB.
#pragma once

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/array>
#include <cuda_runtime.h>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV::cuda {

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

/// @brief Component-wise sum for CUB reductions of energy/stress or moments.
/// @tparam T Accumulation scalar type.
/// @tparam N Number of independent components; components never mix.
template <typename T, std::size_t N>
struct SumComponents {
    /// @brief Add corresponding components, preserving their scalar precision.
    /// @param a First contribution.
    /// @param b Second contribution.
    /// @return Component-wise sum.
    __host__ __device__ ::cuda::std::array<T, N>
    operator()(const ::cuda::std::array<T, N>& a,
               const ::cuda::std::array<T, N>& b) const {
        ::cuda::std::array<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
            result[i] = a[i] + b[i];
        return result;
    }
};

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

} // namespace ATV::cuda
