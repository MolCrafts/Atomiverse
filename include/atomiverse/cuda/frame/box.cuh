/// @file cuda/frame/box.cuh
/// @brief cuda::Box — canonical simulation cell + min-image + wrap on the GPU.
///
/// Plain trivially-copyable struct so the type can be uploaded to
/// `__constant__` memory or passed by value through a kernel argument.
/// Member methods (`length / volume / min_image / wrap_axis`) mirror the
/// `cpu::Box` API (docs/developer/project-contract.md `## Architecture § Backend directory layering`).
///
/// The physical cell is cuda::Frame::box. Backend::materialize publishes a
/// derived device Box; box-changing bindings declare changes_box(), and the
/// backend refreshes that cache in stream order. Kernels resolve current()
/// once at entry, keeping inverse-cell work out of atom and pair loops.
/// Standalone component fixtures may instead pass a value with resident=null.
///
/// The cell is geometry (`geom_t`, FP64 in every mode). Integrators wrap
/// reconstructed coordinates against it directly; force kernels take the
/// narrowed `as<force_t>()` copy once at entry so the pair loop never mixes
/// precisions.
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

/// @brief Cell matrix, cached inverse and orthorhombic scalars in precision T.
///
/// Row-major 3×3 storage matching `triclinic.cuh`. Lattice vectors are the
/// cell **columns** (molrs/LAMMPS H, upper triangular for tilted cells):
/// r = cell · s, fractional s = inv_cell · r.
template <typename T>
struct BoxGeometry {
    T cell[9] = {};
    T inv_cell[9] = {};
    T half_l[3] = {};
    T inv_l[3] = {};
    bool periodic[3] = {true, true, true};
    bool ortho = true;

    /// @brief Edge length along axis `d`.
    __host__ __device__ __forceinline__ T length(int d) const noexcept {
        return cell[d * 3 + d];
    }

    /// @brief Cell volume.
    ///
    /// Orthorhombic fast path: product of diagonal entries. Triclinic:
    /// |det(cell)|.
    __host__ __device__ __forceinline__ T volume() const noexcept {
        if (ortho) {
            return cell[0] * cell[4] * cell[8];
        }
        const T a = cell[0], bb = cell[1], c = cell[2];
        const T d = cell[3], e = cell[4], f = cell[5];
        const T g = cell[6], h = cell[7], i = cell[8];
        const T det =
            a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
        return det < T(0) ? -det : det;
    }

    /// @brief Apply minimum-image convention, in place.
    ///
    /// Orthorhombic path uses the `nearbyint` form (branch-free,
    /// identical across host/device); triclinic uses the inv_cell
    /// roundtrip.
    template <typename U>
    __host__ __device__ __forceinline__ void min_image(U& dx, U& dy,
                                                       U& dz) const noexcept {
        if (ortho) {
            dx -= U(cell[0]) * nearbyint(dx * U(inv_l[0]));
            dy -= U(cell[4]) * nearbyint(dy * U(inv_l[1]));
            dz -= U(cell[8]) * nearbyint(dz * U(inv_l[2]));
        } else {
            U fx =
                U(inv_cell[0]) * dx + U(inv_cell[1]) * dy + U(inv_cell[2]) * dz;
            U fy =
                U(inv_cell[3]) * dx + U(inv_cell[4]) * dy + U(inv_cell[5]) * dz;
            U fz =
                U(inv_cell[6]) * dx + U(inv_cell[7]) * dy + U(inv_cell[8]) * dz;
            fx -= nearbyint(fx);
            fy -= nearbyint(fy);
            fz -= nearbyint(fz);
            dx = U(cell[0]) * fx + U(cell[1]) * fy + U(cell[2]) * fz;
            dy = U(cell[3]) * fx + U(cell[4]) * fy + U(cell[5]) * fz;
            dz = U(cell[6]) * fx + U(cell[7]) * fy + U(cell[8]) * fz;
        }
    }

    /// @brief Map a position into the primary cell by a lattice translation.
    ///
    /// Orthorhombic cells wrap each axis independently. Skew cells wrap the
    /// fractional coordinates s = cell⁻¹·r into [0, 1) (lattice vectors are
    /// the cell columns): only whole lattice vectors are subtracted, so the
    /// position stays a valid periodic image.
    template <typename U>
    __host__ __device__ __forceinline__ void
    wrap_position(U& x, U& y, U& z) const noexcept {
        if (ortho) {
            x -= U(cell[0]) * floor(x * U(inv_l[0]));
            y -= U(cell[4]) * floor(y * U(inv_l[1]));
            z -= U(cell[8]) * floor(z * U(inv_l[2]));
            return;
        }
        U s[3];
        for (int k = 0; k < 3; ++k)
            s[k] = floor(U(inv_cell[3 * k]) * x + U(inv_cell[3 * k + 1]) * y
                         + U(inv_cell[3 * k + 2]) * z);
        x -= U(cell[0]) * s[0] + U(cell[1]) * s[1] + U(cell[2]) * s[2];
        y -= U(cell[3]) * s[0] + U(cell[4]) * s[1] + U(cell[5]) * s[2];
        z -= U(cell[6]) * s[0] + U(cell[7]) * s[1] + U(cell[8]) * s[2];
    }

    /// @brief Wrap one axis into `[0, L_d)`. Orthorhombic only.
    template <typename U>
    __host__ __device__ __forceinline__ U wrap_axis(U q, int d) const noexcept {
        const U L = U(cell[d * 3 + d]);
        if (L <= U(0))
            return q;
        U w = q - L * floor(q * U(inv_l[d]));
        if (w < U(0))
            w += L;
        if (w >= L)
            w -= L;
        return w;
    }
};

/// @brief Plain device-uploadable cell + cached inverse + ortho fast-path scalars.
struct Box : BoxGeometry<geom_t> {
    /// @brief Floating-point tolerance used by `Backend::materialize`
    ///        (and any future setter) to classify an off-diagonal cell
    ///        as orthorhombic. Must match `cpu::Box`'s ortho threshold
    ///        so a borderline-triclinic frame doesn't get classified
    ///        differently on the two backends.
    static constexpr geom_t kOrthoEps = geom_t(1e-12);

    /// Backend-owned derived cell cache. Kernel arguments resolve this view
    /// once at entry; the physical cell remains cuda::Frame::box.
    const Box* resident = nullptr;

    /// Cell revision: the backend increments it whenever a publish changes
    /// the cell. Consumers that keep cell-derived tables (PME influence)
    /// compare it instead of copying the geometry. Constant for by-value
    /// boxes without a resident cache.
    unsigned long long revision = 0;

    __host__ __device__ __forceinline__ Box current() const noexcept {
#if defined(__CUDA_ARCH__)
        return resident ? *resident : *this;
#else
        return *this;
#endif
    }

    /// @brief Narrowed copy of the geometry for kernels computing in T.
    /// @return Cell, inverse and orthorhombic scalars rounded to T.
    template <typename T>
    __host__ __device__ __forceinline__ BoxGeometry<T> as() const noexcept {
        BoxGeometry<T> out;
        for (int k = 0; k < 9; ++k) {
            out.cell[k] = T(cell[k]);
            out.inv_cell[k] = T(inv_cell[k]);
        }
        for (int d = 0; d < 3; ++d) {
            out.half_l[d] = T(half_l[d]);
            out.inv_l[d] = T(inv_l[d]);
            out.periodic[d] = periodic[d];
        }
        out.ortho = ortho;
        return out;
    }

    /// @brief Replace the cell and refresh every derived quantity.
    /// @param values Row-major 3×3 cell [Angstrom].
    __host__ __device__ void assign_cell(const geom_t* values) noexcept {
        for (int k = 0; k < 9; ++k)
            cell[k] = values[k];
        ortho = fabs(cell[1]) < kOrthoEps && fabs(cell[2]) < kOrthoEps
                && fabs(cell[3]) < kOrthoEps && fabs(cell[5]) < kOrthoEps
                && fabs(cell[6]) < kOrthoEps && fabs(cell[7]) < kOrthoEps;
        for (int axis = 0; axis < 3; ++axis) {
            const geom_t length = cell[4 * axis];
            inv_l[axis] = length > geom_t(0) ? geom_t(1) / length : geom_t(0);
            half_l[axis] = geom_t(0.5) * length;
        }
        const geom_t a = cell[0], b = cell[1], c = cell[2];
        const geom_t d = cell[3], e = cell[4], f = cell[5];
        const geom_t g = cell[6], h = cell[7], i = cell[8];
        const geom_t det =
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        for (auto& value : inv_cell)
            value = geom_t(0);
        if (det != geom_t(0)) {
            const geom_t r = geom_t(1) / det;
            inv_cell[0] = (e * i - f * h) * r;
            inv_cell[1] = (c * h - b * i) * r;
            inv_cell[2] = (b * f - c * e) * r;
            inv_cell[3] = (f * g - d * i) * r;
            inv_cell[4] = (a * i - c * g) * r;
            inv_cell[5] = (c * d - a * f) * r;
            inv_cell[6] = (d * h - e * g) * r;
            inv_cell[7] = (b * g - a * h) * r;
            inv_cell[8] = (a * e - b * d) * r;
        }
    }
};

static_assert(
    std::is_trivially_copyable<Box>::value,
    "cuda::Box must be trivially copyable for kernel-arg pass-by-value");

} // namespace cuda
} // namespace ATV
