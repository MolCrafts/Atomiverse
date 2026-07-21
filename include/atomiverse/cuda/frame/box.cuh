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
/// copy lives on `cuda::Context::host_box`, populated by
/// `Backend::materialize()` and mutated by box-changing components
/// (currently `cuda::md::NoseHooverNPT`; `cuda::md::Deform`,
/// `cuda::md::Barostat` are open follow-ups).
/// Integrator + bonded + neighbor-list-pair kernels take `cuda::Box` by
/// value at launch and route PBC through `box.min_image(dx, dy, dz)`.
///
/// @see include/atomiverse/cpu/frame/box.h for the CPU mirror.
#pragma once

#include <type_traits>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Plain device-uploadable cell + cached inverse + ortho fast-path scalars.
///
/// Row-major 3×3 storage matching `triclinic.cuh`. Trivially copyable
/// so the struct can travel through `cudaMemcpyToSymbol` or kernel-arg
/// pass-by-value without surgery.
struct Box {
    /// @brief Floating-point tolerance used by `Backend::materialize`
    ///        (and any future setter) to classify an off-diagonal cell
    ///        as orthorhombic. Must match `cpu::Box`'s ortho threshold
    ///        so a borderline-triclinic frame doesn't get classified
    ///        differently on the two backends.
    static constexpr real_t kOrthoEps = real_t(1e-12);

    real_t cell[9] = {};
    real_t inv_cell[9] = {};
    real_t half_l[3] = {};
    real_t inv_l[3] = {};
    bool periodic[3] = {true, true, true};
    bool ortho = true;

    /// @brief Edge length along axis `d`.
    __host__ __device__ __forceinline__ real_t length(int d) const noexcept {
        return cell[d * 3 + d];
    }

    /// @brief Cell volume.
    ///
    /// Orthorhombic fast path: product of diagonal entries. Triclinic:
    /// |det(cell)|.
    __host__ __device__ __forceinline__ real_t volume() const noexcept {
        if (ortho) {
            return cell[0] * cell[4] * cell[8];
        }
        const real_t a = cell[0], bb = cell[1], c = cell[2];
        const real_t d = cell[3], e = cell[4], f = cell[5];
        const real_t g = cell[6], h = cell[7], i = cell[8];
        const real_t det =
            a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
        return det < real_t(0) ? -det : det;
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

static_assert(
    std::is_trivially_copyable<Box>::value,
    "cuda::Box must be trivially copyable for kernel-arg pass-by-value");

} // namespace cuda
} // namespace ATV
