#pragma once

/// @file cuda/neighbor/morton.cuh
/// @brief Morton (Z-order) code bit-interleaving for spatial sorting.

#include <cstdint>

namespace ATV::cuda::md::nlist {

/// @brief Spread 10 bits across 30 bits for 3D Morton code.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// @brief Compute 30-bit Morton code from 3 x 10-bit integer coordinates.
__host__ __device__ inline uint32_t morton_3d(uint32_t x, uint32_t y,
                                              uint32_t z) {
    return (expand_bits(x) << 2) | (expand_bits(y) << 1) | expand_bits(z);
}

/// @brief Compute Morton code from fractional coordinates in [0,1).
template <typename work_t>
__host__ __device__ inline uint32_t morton_from_frac(work_t fx, work_t fy,
                                                     work_t fz) {
    const uint32_t resolution = 1023; // 10-bit per axis
    auto clamp = [](work_t v, work_t lo, work_t hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    uint32_t ix =
        static_cast<uint32_t>(clamp(fx * 1024, work_t(0), work_t(resolution)));
    uint32_t iy =
        static_cast<uint32_t>(clamp(fy * 1024, work_t(0), work_t(resolution)));
    uint32_t iz =
        static_cast<uint32_t>(clamp(fz * 1024, work_t(0), work_t(resolution)));
    return morton_3d(ix, iy, iz);
}

} // namespace ATV::cuda::md::nlist
