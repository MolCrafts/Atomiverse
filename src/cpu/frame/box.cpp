/// @file src/cpu/frame/box.cpp
/// @brief Cold-path mutation + cache rebuild for `cpu::Box`.

#include "atomiverse/cpu/frame/box.h"

#include <cmath>

namespace ATV {
namespace cpu {

namespace {

/// @brief Numerical zero threshold for off-diagonal entries.
///
/// `Mat3::diagonal(...)` and `Mat3::cubic(...)` zero off-diagonals
/// exactly, so the strict `== 0.0` test covers the orthorhombic case
/// without a tolerance. Triclinic cells with intentional off-diagonal
/// entries fall through to the general path.
inline bool is_diagonal(const Mat3& m) noexcept {
    return m(0, 1) == 0.0 && m(0, 2) == 0.0 && m(1, 0) == 0.0 && m(1, 2) == 0.0
           && m(2, 0) == 0.0 && m(2, 1) == 0.0;
}

/// @brief Invert a 3×3 row-major matrix via cofactor expansion.
///
/// Cheaper than calling out to xtensor-blas for a fixed 3×3, and lets
/// `set_cell` stay free of heavy dependencies.
void invert_3x3(const Mat3& m, Mat3& inv) noexcept {
    const double a = m(0, 0), b = m(0, 1), c = m(0, 2);
    const double d = m(1, 0), e = m(1, 1), f = m(1, 2);
    const double g = m(2, 0), h = m(2, 1), i = m(2, 2);

    const double det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const double inv_det = 1.0 / det;

    inv(0, 0) = (e * i - f * h) * inv_det;
    inv(0, 1) = (c * h - b * i) * inv_det;
    inv(0, 2) = (b * f - c * e) * inv_det;
    inv(1, 0) = (f * g - d * i) * inv_det;
    inv(1, 1) = (a * i - c * g) * inv_det;
    inv(1, 2) = (c * d - a * f) * inv_det;
    inv(2, 0) = (d * h - e * g) * inv_det;
    inv(2, 1) = (b * g - a * h) * inv_det;
    inv(2, 2) = (a * e - b * d) * inv_det;
}

} // namespace

Box& Box::operator=(const Box& other) noexcept {
    if (this == &other)
        return *this;
    cell_ = other.cell_;
    recompute_cache_();
    bump_revision_();
    return *this;
}

Box& Box::operator=(Box&& other) noexcept {
    if (this == &other)
        return *this;
    cell_ = other.cell_;
    recompute_cache_();
    bump_revision_();
    return *this;
}

void Box::recompute_cache_() noexcept {
    ortho_ = is_diagonal(cell_);
    if (ortho_) {
        const double lx = cell_(0, 0);
        const double ly = cell_(1, 1);
        const double lz = cell_(2, 2);
        half_lx_ = 0.5 * lx;
        half_ly_ = 0.5 * ly;
        half_lz_ = 0.5 * lz;
        inv_lx_ = (lx > 0.0) ? 1.0 / lx : 0.0;
        inv_ly_ = (ly > 0.0) ? 1.0 / ly : 0.0;
        inv_lz_ = (lz > 0.0) ? 1.0 / lz : 0.0;
        // Diagonal inverse for compat readers of `inv_cell()`.
        inv_cell_ = Mat3::diagonal(inv_lx_, inv_ly_, inv_lz_);
    } else {
        invert_3x3(cell_, inv_cell_);
        // Per-axis ortho scratch left in a defined state but never read in
        // the triclinic path.
        half_lx_ = 0.5 * cell_(0, 0);
        half_ly_ = 0.5 * cell_(1, 1);
        half_lz_ = 0.5 * cell_(2, 2);
        inv_lx_ = inv_ly_ = inv_lz_ = 0.0;
    }
}

void Box::set_cell(const Mat3& cell) noexcept {
    cell_ = cell;
    recompute_cache_();
    bump_revision_();
}

void Box::scale_diagonal(double sx, double sy, double sz) noexcept {
    cell_(0, 0) *= sx;
    cell_(1, 1) *= sy;
    cell_(2, 2) *= sz;
    recompute_cache_();
    bump_revision_();
}

void Box::scale_isotropic(double s) noexcept {
    cell_(0, 0) *= s;
    cell_(1, 1) *= s;
    cell_(2, 2) *= s;
    recompute_cache_();
    bump_revision_();
}

void Box::scale_axis(int d, double s) noexcept {
    cell_(d, d) *= s;
    recompute_cache_();
    bump_revision_();
}

} // namespace cpu
} // namespace ATV
