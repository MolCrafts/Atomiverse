/// @file cpu/frame/box.h
/// @brief cpu::Box -- canonical simulation cell + min-image + wrap.
///
/// Single source of truth for cell data and periodic-boundary helpers on
/// the CPU backend. Lives in the `frame/` pillar because it is part of the
/// Frame's physical `box` region and is consumed across method domains.
///
/// **Hot-path discipline.** Box stores fundamental cell data
/// (`Mat3 cell_`) plus a per-step-immutable cache (`Mat3 inv_cell_`,
/// per-axis `inv_L`, face heights and reduced lattice). The cache is refreshed
/// exactly when a mutation method is called (`set_cell`, `scale_*`),
/// which is at most once per MD step under NPT. No allocation, no
/// indirection in any inline method.
///
/// Cell columns are lattice vectors: Cartesian r = H * fractional s.
/// Orthogonal cells retain an inline scalar fast path. General triclinic
/// minimum images use a cached reduced lattice and exact sphere search, since rounding
/// fractional coordinates alone does not select the nearest Cartesian image.
///
/// @see include/atomiverse/cpu/frame/box_simd.h for the xsimd batch overload
/// @see include/atomiverse/cuda/frame/box.cuh for the GPU mirror
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"

namespace ATV {
namespace cpu {

/// @brief Simulation cell with cached inverse and ortho fast-path scalars.
///
/// All hot-path queries are `[[gnu::always_inline]]` and branch-free in
/// the orthorhombic case (the dominant case in classical MD and AIMD).
/// The triclinic branch searches a cached lattice-reduced QR basis.
class Box {
  public:
    /// @brief Default-construct an absent periodic cell (all entries zero).
    Box() noexcept = default;

    /// @brief Construct from a row-major 3×3 cell matrix.
    /// @param cell Cell matrix [Angstrom]; columns are lattice vectors a, b, c.
    explicit Box(const Mat3& cell) { set_cell(cell); }

    Box(const Box&) noexcept = default;
    Box(Box&&) noexcept = default;
    Box& operator=(const Box& other) noexcept;
    Box& operator=(Box&& other) noexcept;

    /// @brief Construct an orthorhombic box.
    /// @param a Edge length along x [Angstrom].
    /// @param b Edge length along y [Angstrom].
    /// @param c Edge length along z [Angstrom].
    /// @return Orthorhombic Box with the given edge lengths.
    static Box orthorhombic(double a, double b, double c) {
        if (!(a > 0) || !(b > 0) || !(c > 0))
            throw std::invalid_argument(
                "Box: orthorhombic lengths must be positive");
        Box bx;
        bx.set_cell(Mat3::diagonal(a, b, c));
        return bx;
    }

    /// @brief Construct a cubic box.
    /// @param l Edge length [Angstrom].
    /// @return Cubic Box with edge length `l`.
    static Box cubic(double l) { return orthorhombic(l, l, l); }

    // ── Hot-path queries (always-inline, branch-free) ──────────────────

    /// @brief Diagonal cell entry along Cartesian axis `d`.
    /// @param d Axis index in `[0, 2]`.
    /// @return Diagonal cell entry `cell(d, d)` [Angstrom].
    [[gnu::always_inline]] inline double length(int d) const noexcept {
        return cell_(d, d);
    }

    /// @brief Cell volume (determinant for triclinic, product of edges for ortho).
    /// @return Volume [Angstrom^3].
    [[gnu::always_inline]] inline double volume() const noexcept {
        if (ortho_) {
            return cell_(0, 0) * cell_(1, 1) * cell_(2, 2);
        }
        // Triclinic: |det(cell)| via expansion along row 0.
        const double a = cell_(0, 0), b = cell_(0, 1), c = cell_(0, 2);
        const double d = cell_(1, 0), e = cell_(1, 1), f = cell_(1, 2);
        const double g = cell_(2, 0), h = cell_(2, 1), i = cell_(2, 2);
        const double det =
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        return det < 0.0 ? -det : det;
    }

    /// @brief Whether the cell uses the positive diagonal fast path (or is absent).
    /// @return `true` for a positive diagonal or absent cell; otherwise `false`.
    [[gnu::always_inline]] inline bool is_orthorhombic() const noexcept {
        return ortho_;
    }

    /// @brief Convert Cartesian coordinates to fractional lattice coordinates.
    /// @param r Cartesian position or displacement [Angstrom].
    /// @return Dimensionless fractional coordinates; zero for an absent cell.
    Vec3 to_fractional(const Vec3& r) const noexcept {
        if (ortho_)
            return {r[0] * inv_lx_, r[1] * inv_ly_, r[2] * inv_lz_};
        return inv_cell_ * r;
    }
    /// @brief Convert fractional lattice coordinates to Cartesian coordinates.
    /// @param s Dimensionless fractional position or displacement.
    /// @return Cartesian coordinates [Angstrom].
    Vec3 to_cartesian(const Vec3& s) const noexcept {
        if (ortho_)
            return {s[0] * cell_(0, 0), s[1] * cell_(1, 1), s[2] * cell_(2, 2)};
        return cell_ * s;
    }

    /// @brief Perpendicular distance between the two faces normal to axis d.
    /// Zero denotes an absent periodic cell. A physical halo R spans R/height
    /// fractional units along that lattice direction.
    /// @param d Lattice direction in `[0, 2]`.
    /// @return Face separation [Angstrom].
    double face_height(int d) const noexcept { return face_heights_[d]; }

    /// @brief Cartesian bounds of a fractional parallelepiped, including tilt.
    /// Derived from the eight corners, as in LAMMPS Domain::bbox.
    /// @param lo Lower fractional bounds (dimensionless).
    /// @param hi Upper fractional bounds (dimensionless).
    /// @return Cartesian lower and upper bounds [Angstrom].
    std::pair<Vec3, Vec3> cartesian_bounds(const Vec3& lo,
                                           const Vec3& hi) const noexcept;

    /// @brief Nearest periodic Cartesian displacement, for any lattice image.
    /// Ties retain the initial nearest-plane image. An absent cell leaves d unchanged.
    [[gnu::always_inline]] inline void min_image(double& dx, double& dy,
                                                 double& dz) const noexcept {
        if (ortho_) {
            if (cell_(0, 0) > 0)
                dx -= cell_(0, 0) * std::nearbyint(dx / cell_(0, 0));
            if (cell_(1, 1) > 0)
                dy -= cell_(1, 1) * std::nearbyint(dy / cell_(1, 1));
            if (cell_(2, 2) > 0)
                dz -= cell_(2, 2) * std::nearbyint(dz / cell_(2, 2));
        } else {
            min_image_triclinic_(dx, dy, dz);
        }
    }

    /// @brief Wrap a single Cartesian coordinate into `[0, L_d)`.
    ///
    /// Orthorhombic only. Used by MPI domain decomposition (`Rebalance`,
    /// `Comm::exchange`) and 1-D histogramming. Use wrap_into for triclinic
    /// positions, whose Cartesian coordinates cannot be wrapped independently.
    ///
    /// @param q Coordinate value [Angstrom].
    /// @param d Axis index in `[0, 2]`.
    /// @return `q` shifted by an integer multiple of `L_d` to lie in `[0, L_d)`.
    ///         If `L_d <= 0`, returns `q` unchanged.
    [[gnu::always_inline]] inline double wrap_axis(double q,
                                                   int d) const noexcept {
        const double L = cell_(d, d);
        if (L <= 0.0)
            return q;
        const double inv_L = (d == 0) ? inv_lx_ : (d == 1) ? inv_ly_ : inv_lz_;
        double w = q - std::floor(q * inv_L) * L;
        if (w < 0.0)
            w += L;
        if (w >= L)
            w -= L;
        return w;
    }

    /// @brief Wrap a Cartesian point into the primary fractional cell [0,1)^3.
    [[gnu::always_inline]] inline void wrap_into(double& x, double& y,
                                                 double& z) const noexcept {
        if (ortho_) {
            x = wrap_axis(x, 0);
            y = wrap_axis(y, 1);
            z = wrap_axis(z, 2);
            return;
        }
        Vec3 s = to_fractional({x, y, z});
        for (int d = 0; d < 3; ++d) {
            s[d] -= std::floor(s[d]);
            if (s[d] >= 1 || s[d] < 0)
                s[d] = 0;
        }
        const Vec3 r = to_cartesian(s);
        x = r[0];
        y = r[1];
        z = r[2];
    }

    /// @brief xsimd-batched orthorhombic minimum-image. **Defined in box_simd.h.**
    ///
    /// Precondition: `is_orthorhombic()`. Triclinic batch is intentionally
    /// unsupported (no-fallback rule). Debug builds assert; release is UB.
    ///
    /// @tparam B xsimd batch type (e.g. `xsimd::batch<double>`).
    /// @param bdx Batch of x displacements, modified in place.
    /// @param bdy Batch of y displacements, modified in place.
    /// @param bdz Batch of z displacements, modified in place.
    template <class B>
    inline void min_image_batch(B& bdx, B& bdy, B& bdz) const noexcept;

    // ── Mutation (cold path) ───────────────────────────────────────────

    /// @brief Replace the cell matrix and refresh the cache.
    /// @param cell Finite invertible cell [Angstrom], or all zero for no cell.
    /// @throws std::invalid_argument for invalid cell entries or lattice cache.
    /// @throws std::runtime_error if Mat3 inversion rejects a singular cell.
    /// The cell, caches and revision remain unchanged on failure.
    void set_cell(const Mat3& cell);

    /// @brief Multiply each diagonal entry by the corresponding scale factor.
    /// @param sx Scale along x.
    /// @param sy Scale along y.
    /// @param sz Scale along z.
    void scale_diagonal(double sx, double sy, double sz);

    /// @brief Isotropic volume scale: multiply the full cell matrix by `s`.
    ///
    /// \(h \leftarrow s\,h\) (all nine entries). For axis-aligned orthorhombic
    /// boxes this equals diagonal-only scaling; for fcc/triclinic primitives
    /// it is required so off-diagonal lattice vectors actually move.
    /// @param s Isotropic scale (dimensionless).
    void scale_isotropic(double s);

    /// @brief Multiply a single diagonal entry by `s`.
    /// @param d Axis index in `[0, 2]`.
    /// @param s Scale.
    void scale_axis(int d, double s);

    // ── Explicit cell accessors ────────────────────────────────────────

    /// @brief Read-only access to the cell matrix.
    /// @return Reference to the underlying `Mat3` cell [Angstrom].
    [[gnu::always_inline]] inline const Mat3& cell() const noexcept {
        return cell_;
    }

    /// @brief Read-only access to the cached inverse cell matrix.
    /// @return Reference to `cell^{-1}` [Angstrom^{-1}].
    [[gnu::always_inline]] inline const Mat3& inv_cell() const noexcept {
        return inv_cell_;
    }

    // ── Derived geometry (cold; returns a small value) ──────────────────

    /// @brief Reciprocal lattice `2*pi * (cell^{-1})^T`.
    /// @return Reciprocal cell matrix [Angstrom^{-1}] (columns are reciprocal lattice vectors).
    Mat3 recip_cell() const noexcept {
        return inv_cell_.transpose() * AccReal(2.0 * units::pi);
    }

    /// @brief Euclidean lengths of the three lattice vectors (cell columns).
    Vec3 lengths() const noexcept {
        return {std::hypot(cell_(0, 0), cell_(1, 0), cell_(2, 0)),
                std::hypot(cell_(0, 1), cell_(1, 1), cell_(2, 1)),
                std::hypot(cell_(0, 2), cell_(1, 2), cell_(2, 2))};
    }

    /// @brief Monotonic mutation revision for cell-dependent component caches.
    ///
    /// Every explicit mutation and assignment advances the destination Box's
    /// revision.  Cell-dependent components compare this scalar rather than
    /// duplicating nine-entry cell snapshots and equality policies.
    [[gnu::always_inline]] inline std::uint64_t revision() const noexcept {
        return revision_;
    }

  private:
    Mat3 cell_{};
    Mat3 inv_cell_{};
    double inv_lx_{0.0}, inv_ly_{0.0}, inv_lz_{0.0};
    Vec3 face_heights_{};
    // Private QR of a lattice-equivalent reduced basis; public H never rotates.
    std::array<long double, 9> mic_q_{};
    std::array<long double, 9> mic_r_{};
    bool ortho_{true};
    std::uint64_t revision_{1};

    void min_image_triclinic_(double& dx, double& dy,
                              double& dz) const noexcept;
    void bump_revision_() noexcept { ++revision_; }
};

} // namespace cpu
} // namespace ATV
