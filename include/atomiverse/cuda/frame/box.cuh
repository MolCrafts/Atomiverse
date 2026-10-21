/// @file cuda/frame/box.cuh
/// @brief cuda::Box — canonical simulation cell + min-image + wrap on the GPU.
///
/// Plain trivially-copyable struct so the type can be uploaded to
/// `__constant__` memory or passed by value through a kernel argument.
/// Member methods (`length / volume / min_image / wrap_axis`) mirror the
/// `cpu::Box` API (CLAUDE.md `## Architecture § Backend directory layering`).
///
/// **Dual storage convention.** `cuda::Frame::box` remains a `prec_t*`
/// 9-element row-major device cell — AsyncBridge observers (`cuda::RDF`)
/// and `cuda::md::NoseHooverNPT::propagate_x_k` (thread-0 box-diagonal
/// write) read/write it directly on device. The host-side authoritative
/// copies live on `cuda::Context::host_box` (force arithmetic) and
/// `host_motion_box` (integration arithmetic), populated by
/// `Backend::materialize()` and mutated by box-changing components
/// (`NoseHooverNPT`, `Deform`, and `Barostat`).
/// Integrator + bonded + neighbor-list-pair kernels take `cuda::Box` by
/// value at launch and route PBC through `box.min_image(dx, dy, dz)`.
///
/// @see include/atomiverse/cpu/frame/box.h for the CPU mirror.
#pragma once

#include <type_traits>

#include "atomiverse/cuda/frame/types.cuh"

// Host TUs (the core Driver template) include this header. nvcc defines the
// execution-space decorators; the host compiler does not.
#if !defined(__CUDACC__)
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#if !defined(__forceinline__)
#define __forceinline__ inline
#endif
#include <cmath>
#endif

namespace ATV {
namespace cuda {

/// @brief Plain device-uploadable cell + cached inverse + ortho fast-path scalars.
///
/// Row-major 3×3 storage matching `triclinic.cuh`. Trivially copyable
/// so the struct can travel through `cudaMemcpyToSymbol` or kernel-arg
/// pass-by-value without surgery.
template <typename CellReal>
struct BasicBox {
    /// @brief Floating-point tolerance used by `Backend::materialize`
    ///        (and any future setter) to classify an off-diagonal cell
    ///        as orthorhombic. Must match `cpu::Box`'s ortho threshold
    ///        so a borderline-triclinic frame doesn't get classified
    ///        differently on the two backends.
    static constexpr CellReal kOrthoEps = CellReal(1e-12);

    CellReal cell[9] = {};
    CellReal inv_cell[9] = {};
    CellReal half_l[3] = {};
    CellReal inv_l[3] = {};
    bool periodic[3] = {true, true, true};
    bool ortho = true;

    /// Recompute geometry caches after changing the canonical cell.
    __host__ __device__ void refresh() noexcept {
        ortho = true;
        for (int i = 0; i < 9; ++i)
            if (i != 0 && i != 4 && i != 8
                && (cell[i] <= -kOrthoEps || cell[i] >= kOrthoEps))
                ortho = false;
        for (int d = 0; d < 3; ++d) {
            const CellReal L = cell[d * 3 + d];
            inv_l[d] = L > CellReal(0) ? CellReal(1) / L : CellReal(0);
            half_l[d] = CellReal(0.5) * L;
        }
        const CellReal a = cell[0], b = cell[1], c = cell[2];
        const CellReal d = cell[3], e = cell[4], f = cell[5];
        const CellReal g = cell[6], h = cell[7], i = cell[8];
        const CellReal det =
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        for (int k = 0; k < 9; ++k)
            inv_cell[k] = CellReal(0);
        if (det > CellReal(0) || det < CellReal(0)) {
            const CellReal inv = CellReal(1) / det;
            inv_cell[0] = (e * i - f * h) * inv;
            inv_cell[1] = (c * h - b * i) * inv;
            inv_cell[2] = (b * f - c * e) * inv;
            inv_cell[3] = (f * g - d * i) * inv;
            inv_cell[4] = (a * i - c * g) * inv;
            inv_cell[5] = (c * d - a * f) * inv;
            inv_cell[6] = (d * h - e * g) * inv;
            inv_cell[7] = (b * g - a * h) * inv;
            inv_cell[8] = (a * e - b * d) * inv;
        }
    }

    /// @brief Edge length along axis `d`.
    __host__ __device__ __forceinline__ CellReal length(int d) const noexcept {
        return cell[d * 3 + d];
    }

    /// @brief Cell volume.
    ///
    /// Orthorhombic fast path: product of diagonal entries. Triclinic:
    /// |det(cell)|.
    __host__ __device__ __forceinline__ CellReal volume() const noexcept {
        if (ortho) {
            return cell[0] * cell[4] * cell[8];
        }
        const CellReal a = cell[0], bb = cell[1], c = cell[2];
        const CellReal d = cell[3], e = cell[4], f = cell[5];
        const CellReal g = cell[6], h = cell[7], i = cell[8];
        const CellReal det =
            a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
        return det < CellReal(0) ? -det : det;
    }

    /// @brief Apply minimum-image convention, in place.
    ///
    /// Orthorhombic path uses the `nearbyint` form (branch-free,
    /// identical across host/device); triclinic uses the inv_cell
    /// roundtrip.
    template <typename T>
    __host__ __device__ __forceinline__ void min_image(T& dx, T& dy,
                                                       T& dz) const noexcept {
        if (ortho) {
            dx -= T(cell[0]) * nearbyint(dx * T(inv_l[0]));
            dy -= T(cell[4]) * nearbyint(dy * T(inv_l[1]));
            dz -= T(cell[8]) * nearbyint(dz * T(inv_l[2]));
        } else {
            T fx =
                T(inv_cell[0]) * dx + T(inv_cell[1]) * dy + T(inv_cell[2]) * dz;
            T fy =
                T(inv_cell[3]) * dx + T(inv_cell[4]) * dy + T(inv_cell[5]) * dz;
            T fz =
                T(inv_cell[6]) * dx + T(inv_cell[7]) * dy + T(inv_cell[8]) * dz;
            fx -= nearbyint(fx);
            fy -= nearbyint(fy);
            fz -= nearbyint(fz);
            dx = T(cell[0]) * fx + T(cell[1]) * fy + T(cell[2]) * fz;
            dy = T(cell[3]) * fx + T(cell[4]) * fy + T(cell[5]) * fz;
            dz = T(cell[6]) * fx + T(cell[7]) * fy + T(cell[8]) * fz;
        }
    }

    /// @brief Wrap one axis into `[0, L_d)`. Orthorhombic only.
    template <typename T>
    __host__ __device__ __forceinline__ T wrap_axis(T q, int d) const noexcept {
        const T L = T(cell[d * 3 + d]);
        if (L <= T(0))
            return q;
        T w = q - L * floor(q * T(inv_l[d]));
        if (w < T(0))
            w += L;
        if (w >= L)
            w -= L;
        return w;
    }
};

using Box = BasicBox<real_t>;
using MotionBox = BasicBox<motion_t>;

/// Derive a geometry cache in another arithmetic precision from its cell.
template <typename To, typename From>
inline BasicBox<To> box_cast(const BasicBox<From>& source) noexcept {
    BasicBox<To> result;
    for (int i = 0; i < 9; ++i)
        result.cell[i] = static_cast<To>(source.cell[i]);
    for (int d = 0; d < 3; ++d)
        result.periodic[d] = source.periodic[d];
    result.refresh();
    return result;
}

static_assert(std::is_trivially_copyable<MotionBox>::value,
              "cuda::MotionBox must be trivially copyable");

static_assert(
    std::is_trivially_copyable<Box>::value,
    "cuda::Box must be trivially copyable for kernel-arg pass-by-value");

} // namespace cuda
} // namespace ATV
