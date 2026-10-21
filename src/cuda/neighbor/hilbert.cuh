/// @file cuda/neighbor/hilbert.cuh
/// @brief 3D Hilbert spatial keys for compact CUDA atom clusters.
/// Global Gray transform: J. Skilling, AIP Conf. Proc. 707, 381 (2004),
/// doi:10.1063/1.1751381. OpenMM ComputeContext.cpp uses Hilbert ordering
/// for large molecules (SHA 3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e).
/// This bit transform is implemented independently; no OpenMM code is copied.
#pragma once
#include "morton.cuh"

namespace ATV::cuda::md::nlist {
template <int Bits>
__host__ __device__ inline uint32_t hilbert_3d(uint32_t x, uint32_t y,
                                               uint32_t z) {
    static_assert(Bits >= 1 && Bits <= 10);
    uint32_t axis[3] = {x, y, z};
    constexpr uint32_t top = 1u << (Bits - 1);
    // Undo the lower-axis reflections/exchanges, from coarse to fine cells.
    for (uint32_t level = top; level > 1; level >>= 1) {
        const uint32_t lower = level - 1;
        for (int d = 0; d < 3; ++d) {
            if (axis[d] & level)
                axis[0] ^= lower;
            else {
                const uint32_t exchange = (axis[0] ^ axis[d]) & lower;
                axis[0] ^= exchange;
                axis[d] ^= exchange;
            }
        }
    }
    axis[1] ^= axis[0];
    axis[2] ^= axis[1];
    uint32_t reflection = 0;
    for (uint32_t level = top; level > 1; level >>= 1)
        if (axis[2] & level)
            reflection ^= level - 1;
    // Reuse only the existing bit-interleaving primitive, not Z-order keys.
    return morton_3d(axis[0] ^ reflection, axis[1] ^ reflection,
                     axis[2] ^ reflection);
}

template <typename T>
__host__ __device__ inline uint32_t hilbert_from_frac(T x, T y, T z) {
    auto quantize = [](T v) {
        v *= T(1024);
        return uint32_t(v < T(0) ? T(0) : (v > T(1023) ? T(1023) : v));
    };
    return hilbert_3d<10>(quantize(x), quantize(y), quantize(z));
}
} // namespace ATV::cuda::md::nlist
