/// @file cpu/frame/box.h
/// @brief cpu::Box -- canonical simulation cell + min-image + wrap.
///
/// Single source of truth for cell data and periodic-boundary helpers on
/// the CPU backend. Lives in the `frame/` pillar because it is part of the
/// Frame's physical `box` region and is consumed across method domains.
///
/// **Hot-path discipline.** Box stores fundamental cell data
/// (`Mat3 cell_`) plus a per-step-immutable cache (`Mat3 inv_cell_`,
/// per-axis `inv_L`, `half_L`, `bool ortho`). The cache is refreshed
/// exactly when a mutation method is called (`set_cell`, `scale_*`),
/// which is at most once per MD step under NPT. No allocation, no
/// indirection in any inline method.
///
/// **Bit-identicality.** The orthorhombic `min_image` uses the same
/// branchy `if (dx > L/2) dx -= L;` form as the pre-refactor static-
/// inline copies in bonded/ and the LJ scalar tail, and the SIMD
/// `min_image_batch` in `box_simd.h` uses the matching `xsimd::select`
/// form. This preserves ULP-exact equivalence with pre-refactor
/// behavior at every callsite. The triclinic path uses the
/// `nearbyint` fractional-roundtrip form (the CUDA reference).
///
/// **No-fallback discipline.** `min_image_batch` requires
/// `is_orthorhombic()`. Calling it on a triclinic box is a
/// precondition violation; debug builds assert.
///
/// @see include/atomiverse/cpu/frame/box_simd.h for the xsimd batch overload
/// @see include/atomiverse/cuda/frame/box.cuh for the GPU mirror
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"

namespace ATV {
namespace cpu {

/// @brief Simulation cell with cached inverse and ortho fast-path scalars.
///
/// All hot-path queries are `[[gnu::always_inline]]` and branch-free in
/// the orthorhombic case (the dominant case in classical MD and AIMD).
/// The triclinic branch uses a Cartesian↔fractional roundtrip with the
/// cached `inv_cell`.
class Box {
  public:
    /// @brief Default-construct an identity box (cell = 0, ortho = true).
    Box() noexcept = default;

    /// @brief Construct from a row-major 3×3 cell matrix.
    /// @param cell Cell matrix [Angstrom]; rows are lattice vectors a, b, c.
    explicit Box(const Mat3& cell) noexcept { set_cell(cell); }

    Box(const Box&) noexcept = default;
    Box(Box&&) noexcept = default;
    Box& operator=(const Box& other) noexcept;
    Box& operator=(Box&& other) noexcept;

    /// @brief Construct an orthorhombic box.
    /// @param a Edge length along x [Angstrom].
    /// @param b Edge length along y [Angstrom].
    /// @param c Edge length along z [Angstrom].
    /// @return Orthorhombic Box with the given edge lengths.
    static Box orthorhombic(double a, double b, double c) noexcept {
        Box bx;
        bx.set_cell(Mat3::diagonal(a, b, c));
        return bx;
    }

    /// @brief Construct a cubic box.
    /// @param l Edge length [Angstrom].
    /// @return Cubic Box with edge length `l`.
    static Box cubic(double l) noexcept { return orthorhombic(l, l, l); }

    // ── Hot-path queries (always-inline, branch-free) ──────────────────

    /// @brief Edge length along axis `d`.
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

    /// @brief Whether the cell is orthorhombic (off-diagonal entries are zero).
    /// @return `true` for orthorhombic, `false` for triclinic.
    [[gnu::always_inline]] inline bool is_orthorhombic() const noexcept {
        return ortho_;
    }

    /// @brief Apply minimum-image convention to a displacement, in place.
    ///
    /// Orthorhombic path uses the canonical branchy form
    /// (`if (dx > L/2) dx -= L; if (dx < -L/2) dx += L;`) to remain
    /// bit-identical with the pre-refactor static-inline copies and the
    /// LJ scalar tail. Triclinic path uses the Cartesian↔fractional
    /// roundtrip with cached `inv_cell` (`nearbyint` form, matches the
    /// CUDA reference).
    ///
    /// **Precondition (ortho path):** `|d_*| ≤ 1.5 · L_*`. This holds
    /// for all physical MD displacements after neighbor-list pruning.
    ///
    /// @param dx Cartesian x displacement [Angstrom], modified in place.
    /// @param dy Cartesian y displacement [Angstrom], modified in place.
    /// @param dz Cartesian z displacement [Angstrom], modified in place.
    [[gnu::always_inline]] inline void min_image(double& dx, double& dy,
                                                 double& dz) const noexcept {
        if (ortho_) {
            const double lx = cell_(0, 0), ly = cell_(1, 1), lz = cell_(2, 2);
            // Round-to-nearest-image (matches the triclinic branch below and
            // the CUDA mirror). Wraps Δ into [-L/2, L/2] in O(1) for ANY
            // multiple of L, so multi-rank ghosts at ±2L are handled in a
            // single op — and, critically, a pathological Δ (e.g. a bonded
            // partner that has stretched beyond the halo and resolves to a
            // stale ghost slot, observed at ~1e11 Å in the 64-rank deform
            // stage) is wrapped in one step instead of spinning the old
            // `while (Δ > L/2) Δ -= L` loop ~1e10 times — that loop was the
            // measured 64-rank Bonded straggler (one rank ~480s vs ~4s). For
            // the common |Δ| <= 1.5L case this is bit-identical to the legacy
            // single-wrap `if` form (nearbyint(Δ/L) ∈ {-1,0,1}).
            dx -= lx * std::nearbyint(dx / lx);
            dy -= ly * std::nearbyint(dy / ly);
            dz -= lz * std::nearbyint(dz / lz);
        } else {
            // Triclinic: f = inv_cell · d, f -= round(f), d = cell · f.
            const double fx = inv_cell_(0, 0) * dx + inv_cell_(0, 1) * dy
                              + inv_cell_(0, 2) * dz;
            const double fy = inv_cell_(1, 0) * dx + inv_cell_(1, 1) * dy
                              + inv_cell_(1, 2) * dz;
            const double fz = inv_cell_(2, 0) * dx + inv_cell_(2, 1) * dy
                              + inv_cell_(2, 2) * dz;
            const double rfx = fx - std::nearbyint(fx);
            const double rfy = fy - std::nearbyint(fy);
            const double rfz = fz - std::nearbyint(fz);
            dx = cell_(0, 0) * rfx + cell_(0, 1) * rfy + cell_(0, 2) * rfz;
            dy = cell_(1, 0) * rfx + cell_(1, 1) * rfy + cell_(1, 2) * rfz;
            dz = cell_(2, 0) * rfx + cell_(2, 1) * rfy + cell_(2, 2) * rfz;
        }
    }

    /// @brief Wrap a single Cartesian coordinate into `[0, L_d)`.
    ///
    /// Orthorhombic only. Used by MPI domain decomposition (`rcb`,
    /// `Comm::exchange`) and 1-D histogramming. Triclinic wrap is a
    /// separate concern (see DD spec).
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

    /// @brief Wrap a Cartesian point into `[0, L) × [0, L) × [0, L)`.
    /// @param x x coordinate [Angstrom], modified in place.
    /// @param y y coordinate [Angstrom], modified in place.
    /// @param z z coordinate [Angstrom], modified in place.
    [[gnu::always_inline]] inline void wrap_into(double& x, double& y,
                                                 double& z) const noexcept {
        x = wrap_axis(x, 0);
        y = wrap_axis(y, 1);
        z = wrap_axis(z, 2);
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
    /// @param cell New cell [Angstrom].
    void set_cell(const Mat3& cell) noexcept;

    /// @brief Multiply each diagonal entry by the corresponding scale factor.
    /// @param sx Scale along x.
    /// @param sy Scale along y.
    /// @param sz Scale along z.
    void scale_diagonal(double sx, double sy, double sz) noexcept;

    /// @brief Isotropic volume scale: multiply the full cell matrix by `s`.
    ///
    /// \(h \leftarrow s\,h\) (all nine entries). For axis-aligned orthorhombic
    /// boxes this equals diagonal-only scaling; for fcc/triclinic primitives
    /// it is required so off-diagonal lattice vectors actually move.
    /// @param s Isotropic scale (dimensionless).
    void scale_isotropic(double s) noexcept;

    /// @brief Multiply a single diagonal entry by `s`.
    /// @param d Axis index in `[0, 2]`.
    /// @param s Scale.
    void scale_axis(int d, double s) noexcept;

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
    /// @return Reciprocal cell matrix [Angstrom^{-1}] (rows are b0, b1, b2).
    Mat3 recip_cell() const noexcept {
        return inv_cell_.transpose() * AccReal(2.0 * units::pi);
    }

    /// @brief Euclidean lengths of the three lattice vectors (cell rows).
    /// @return Vector of edge lengths [Angstrom]. For an orthorhombic cell
    ///         these equal `length(d)`; for triclinic they are the row norms.
    Vec3 lengths() const noexcept {
        return {std::sqrt(cell_(0, 0) * cell_(0, 0) + cell_(0, 1) * cell_(0, 1)
                          + cell_(0, 2) * cell_(0, 2)),
                std::sqrt(cell_(1, 0) * cell_(1, 0) + cell_(1, 1) * cell_(1, 1)
                          + cell_(1, 2) * cell_(1, 2)),
                std::sqrt(cell_(2, 0) * cell_(2, 0) + cell_(2, 1) * cell_(2, 1)
                          + cell_(2, 2) * cell_(2, 2))};
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
    double half_lx_{0.0}, half_ly_{0.0}, half_lz_{0.0};
    double inv_lx_{0.0}, inv_ly_{0.0}, inv_lz_{0.0};
    bool ortho_{true};
    std::uint64_t revision_{1};

    void recompute_cache_() noexcept;
    void bump_revision_() noexcept { ++revision_; }
};

} // namespace cpu
} // namespace ATV
