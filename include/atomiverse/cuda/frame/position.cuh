/// @file cuda/frame/position.cuh
/// @brief Reconstruct and split the physical coordinate representation.
#pragma once

#include "atomiverse/cuda/frame/box.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ATV::cuda {

/// @brief Load a motion-precision coordinate [Angstrom].
/// @param main Main coordinate array [Angstrom].
/// @param correction Residual array [Angstrom]; required only in mixed mode.
/// @param i Local atom row [dimensionless].
/// @return Reconstructed physical coordinate [Angstrom].
__host__ __device__ inline motion_t
load_position(const real_t* main, const real_t* correction, int i) {
    if constexpr (kMixedPrecision)
        return motion_t(main[i]) + motion_t(correction[i]);
    else
        return motion_t(main[i]);
}

/// @brief Store a physical coordinate without losing its mixed residual.
/// @param main Main coordinate array [Angstrom].
/// @param correction Residual array [Angstrom]; required only in mixed mode.
/// @param i Local atom row [dimensionless].
/// @param value Updated coordinate [Angstrom].
__host__ __device__ inline void store_position(real_t* main, real_t* correction,
                                               int i, motion_t value) {
    main[i] = real_t(value);
    if constexpr (kMixedPrecision)
        correction[i] = real_t(value - motion_t(main[i]));
}

/// @brief Wrap a physical position and accumulate lattice crossings.
/// @param image Interleaved signed crossing counts [3 * n_atoms].
/// @param box Current simulation cell, including per-axis periodicity.
/// @param px,py,pz Proposed physical position [Angstrom].
/// Nonperiodic fractional axes retain their coordinates and image counts.
/// Crossing counts must fit int8; overflow fails before storing any state.
__host__ __device__ inline void
wrap_position(real_t* x, real_t* y, real_t* z, real_t* xc, real_t* yc,
              real_t* zc, std::int8_t* image, int i, const MotionBox& box,
              motion_t px, motion_t py, motion_t pz) {
    motion_t pos[3] = {px, py, pz};
    motion_t shift[3] = {};
    if (box.ortho) {
        for (int d = 0; d < 3; ++d) {
            const motion_t L = box.cell[4 * d];
            if (!box.periodic[d] || L <= motion_t(0))
                continue;
            const motion_t raw = pos[d];
            shift[d] = floor(pos[d] * box.inv_l[d]);
            pos[d] -= shift[d] * L;
            // Correct a rounded division/multiply at a cell face.
            if (pos[d] < motion_t(0)) {
                pos[d] += L;
                shift[d] -= motion_t(1);
                // A tiny negative coordinate can round to L after adding L.
                // Keep its negative crossing while retaining a wrapped value.
                if (pos[d] >= L)
                    pos[d] = nextafter(L, motion_t(0));
            } else if (pos[d] >= L) {
                // A negative coordinate smaller than one ulp of L can also
                // round to L in the initial subtraction. Distinguish that
                // rounded remainder from a genuinely underestimated crossing.
                if (raw < (shift[d] + motion_t(1)) * L)
                    pos[d] = nextafter(L, motion_t(0));
                else {
                    pos[d] -= L;
                    shift[d] += motion_t(1);
                }
            }
        }
    } else {
        const motion_t raw[3] = {px, py, pz};
        for (int d = 0; d < 3; ++d) {
            const motion_t fraction = box.inv_cell[3 * d] * raw[0]
                                      + box.inv_cell[3 * d + 1] * raw[1]
                                      + box.inv_cell[3 * d + 2] * raw[2];
            if (box.periodic[d])
                shift[d] = floor(fraction);
        }
        for (int d = 0; d < 3; ++d)
            pos[d] -= box.cell[3 * d] * shift[0]
                      + box.cell[3 * d + 1] * shift[1]
                      + box.cell[3 * d + 2] * shift[2];
    }
    int next[3] = {};
    for (int d = 0; d < 3; ++d) {
        const motion_t count =
            motion_t(image ? image[3 * i + d] : 0) + shift[d];
        if (!(count >= motion_t(-128) && count <= motion_t(127))
            || (!image && shift[d] != motion_t(0))) {
#if defined(__CUDA_ARCH__)
            asm volatile("trap;");
#else
            throw std::overflow_error(
                "cuda::wrap_position: image crossing count overflows int8 "
                "or image storage is absent");
#endif
        }
        next[d] = static_cast<int>(count);
    }
    if (image)
        for (int d = 0; d < 3; ++d)
            image[3 * i + d] = static_cast<std::int8_t>(next[d]);
    store_position(x, xc, i, pos[0]);
    store_position(y, yc, i, pos[1]);
    store_position(z, zc, i, pos[2]);
}

} // namespace ATV::cuda
