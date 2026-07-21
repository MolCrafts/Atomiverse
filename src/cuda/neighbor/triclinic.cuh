#pragma once

/// @file triclinic.cuh
/// @brief Fractional coordinate transforms + 3x3 inverse helpers for
///        neighbor-list internals.
///
/// Cell convention: row-major 3x3 matrix, r_cart = cell * r_frac.
/// Rows of cell are lattice vectors a, b, c.
///
/// **Minimum image:** call `box.min_image(dx, dy, dz)` on
/// `ATV::cuda::Box` directly. The old raw-pointer / Box-forwarding
/// `minimum_image` shims were retired in
/// `cuda-md-review-cleanup-04-box-by-value-pbc` once all pair / fill /
/// export kernels migrated to the by-value-Box signature.

#include <cmath>
#include <cstdint>

#include "atomiverse/cuda/frame/box.cuh"

namespace ATV::cuda::md::nlist {

/// @brief 3-component vector for device-side math.
template <typename work_t>
struct Vec3 {
    work_t x, y, z;
};

/// @brief Compute the inverse of a 3x3 row-major matrix.
/// @param m Input 3x3 matrix (row-major)
/// @param inv Output inverse matrix (row-major)
/// @return determinant
template <typename work_t>
__host__ __device__ inline work_t invert_3x3(const real_t* m, work_t* inv) {
    work_t det = m[0] * (m[4] * m[8] - m[5] * m[7])
                 - m[1] * (m[3] * m[8] - m[5] * m[6])
                 + m[2] * (m[3] * m[7] - m[4] * m[6]);

    work_t inv_det = work_t(1) / det;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return det;
}

/// @brief Cartesian to fractional: r_frac = inv_cell * r_cart
/// where cell is row-major, inv_cell is inverse of cell.
template <typename work_t>
__host__ __device__ inline Vec3<work_t>
cart_to_frac(work_t rx, work_t ry, work_t rz, const real_t* inv_cell) {
    return {inv_cell[0] * rx + inv_cell[1] * ry + inv_cell[2] * rz,
            inv_cell[3] * rx + inv_cell[4] * ry + inv_cell[5] * rz,
            inv_cell[6] * rx + inv_cell[7] * ry + inv_cell[8] * rz};
}

/// @brief Clamp fractional coordinate to [0, 1).
/// @brief Boundary epsilon for clamping fractional coordinates to [0,1).
template <typename work_t>
inline constexpr work_t kFracBoundaryEps = work_t(1e-6);

template <typename work_t>
__device__ inline work_t clamp01(work_t v) {
    if (v < work_t(0))
        return work_t(0);
    if (v >= work_t(1))
        return work_t(1) - kFracBoundaryEps<work_t>;
    return v;
}

/// @brief Wrap fractional coordinate to [0, 1).
template <typename work_t>
__host__ __device__ inline work_t wrap_frac(work_t f) {
    f -= floor(f);
    // Guard against numerical edge case where floor gives exactly 1.0
    if (f >= work_t(1))
        f -= work_t(1);
    if (f < work_t(0))
        f += work_t(1);
    return f;
}

} // namespace ATV::cuda::md::nlist
