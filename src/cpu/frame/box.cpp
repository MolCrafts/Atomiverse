/// @file src/cpu/frame/box.cpp
/// @brief Cell mutation, triclinic geometry and exact minimum images.
#include "atomiverse/cpu/frame/box.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ATV::cpu {
namespace {
using MicMatrix = std::array<long double, 9>;

// Reorthogonalized modified Gram-Schmidt, only on cold 3x3 geometry changes.
void mic_qr(const MicMatrix& basis, MicMatrix& q, MicMatrix& r) {
    q.fill(0);
    r.fill(0);
    for (int j = 0; j < 3; ++j) {
        long double v[3] = {basis[j], basis[3 + j], basis[6 + j]};
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < j; ++i) {
                long double dot = 0;
                for (int d = 0; d < 3; ++d)
                    dot += q[3 * d + i] * v[d];
                r[3 * i + j] += dot;
                for (int d = 0; d < 3; ++d)
                    v[d] -= dot * q[3 * d + i];
            }
        const long double norm = std::hypot(v[0], v[1], v[2]);
        if (!(norm > 0) || !std::isfinite(norm))
            throw std::invalid_argument(
                "Box: lattice QR is not finite and nonsingular");
        r[3 * j + j] = norm;
        for (int d = 0; d < 3; ++d)
            q[3 * d + j] = v[d] / norm;
    }
}

void mic_reduce(const Mat3& cell, MicMatrix& q, MicMatrix& r) {
    MicMatrix basis;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            basis[3 * i + j] = cell(i, j);
    mic_qr(basis, q, r);
    // LLL integer column operations preserve the lattice. Size reduction
    // removes huge shear before per-displacement search; Lovasz swaps put
    // short directions first, so reverse QR enumeration branches on long ones.
    for (int k = 1; k < 3;) {
        for (int j = k - 1; j >= 0; --j) {
            const long double multiple =
                std::round(r[3 * j + k] / r[3 * j + j]);
            if (multiple == 0)
                continue;
            for (int d = 0; d < 3; ++d)
                basis[3 * d + k] =
                    std::fma(-multiple, basis[3 * d + j], basis[3 * d + k]);
            mic_qr(basis, q, r);
        }
        const long double mu = r[3 * (k - 1) + k] / r[3 * (k - 1) + (k - 1)];
        const long double ratio = r[3 * k + k] / r[3 * (k - 1) + (k - 1)];
        if (ratio * ratio >= 0.75L - mu * mu) {
            ++k;
        } else {
            for (int d = 0; d < 3; ++d)
                std::swap(basis[3 * d + k], basis[3 * d + k - 1]);
            mic_qr(basis, q, r);
            k = std::max(1, k - 1);
        }
    }
}
} // namespace

Box& Box::operator=(const Box& other) noexcept {
    if (this == &other)
        return *this;
    cell_ = other.cell_;
    inv_cell_ = other.inv_cell_;
    inv_lx_ = other.inv_lx_;
    inv_ly_ = other.inv_ly_;
    inv_lz_ = other.inv_lz_;
    face_heights_ = other.face_heights_;
    mic_q_ = other.mic_q_;
    mic_r_ = other.mic_r_;
    ortho_ = other.ortho_;
    bump_revision_();
    return *this;
}

Box& Box::operator=(Box&& other) noexcept {
    return *this = static_cast<const Box&>(other);
}

void Box::set_cell(const Mat3& cell) {
    bool zero = true, diagonal = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(cell(i, j)))
                throw std::invalid_argument("Box: cell entries must be finite");
            zero &= cell(i, j) == 0;
            if (i != j)
                diagonal &= cell(i, j) == 0;
        }
    Mat3 inverse;
    Vec3 heights;
    MicMatrix mic_q{}, mic_r{};
    bool positive_diagonal = diagonal;
    if (!zero) {
        if (diagonal) {
            for (int d = 0; d < 3; ++d) {
                if (cell(d, d) == 0)
                    throw std::invalid_argument(
                        "Box: cell must be nonsingular");
                positive_diagonal &= cell(d, d) > 0;
                inverse(d, d) = 1.0 / cell(d, d);
                if (!std::isfinite(inverse(d, d)))
                    throw std::invalid_argument(
                        "Box: cell inverse must be finite");
                heights[d] = std::abs(cell(d, d));
            }
        } else {
            // Reuse the canonical fixed-size matrix inversion. Compute every
            // derived value before publishing, so a failed mutation is atomic.
            inverse = cell.inverse();
            for (int d = 0; d < 3; ++d) {
                const double norm =
                    std::hypot(inverse(d, 0), inverse(d, 1), inverse(d, 2));
                if (!(norm > 0) || !std::isfinite(norm))
                    throw std::invalid_argument(
                        "Box: cell inverse must be finite and nonsingular");
                heights[d] = 1.0 / norm;
            }
        }
    }
    if (!zero && !positive_diagonal)
        mic_reduce(cell, mic_q, mic_r);
    cell_ = cell;
    inv_cell_ = inverse;
    face_heights_ = heights;
    mic_q_ = mic_q;
    mic_r_ = mic_r;
    ortho_ = positive_diagonal;
    inv_lx_ = positive_diagonal ? inverse(0, 0) : 0;
    inv_ly_ = positive_diagonal ? inverse(1, 1) : 0;
    inv_lz_ = positive_diagonal ? inverse(2, 2) : 0;
    bump_revision_();
}

void Box::scale_diagonal(double sx, double sy, double sz) {
    Mat3 scaled = cell_;
    scaled(0, 0) *= sx;
    scaled(1, 1) *= sy;
    scaled(2, 2) *= sz;
    set_cell(scaled);
}

void Box::scale_isotropic(double s) { set_cell(cell_ * s); }

void Box::scale_axis(int d, double s) {
    if (d < 0 || d >= 3)
        throw std::out_of_range("Box: axis must be in [0,3)");
    Mat3 scaled = cell_;
    scaled(d, d) *= s;
    set_cell(scaled);
}

std::pair<Vec3, Vec3> Box::cartesian_bounds(const Vec3& lo,
                                            const Vec3& hi) const noexcept {
    // LAMMPS src/domain.cpp Domain::bbox, a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
    // Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
    // the U.S. Government retains certain rights. GPL-2.0-only.
    // Each Cartesian coordinate is linear, so choosing each endpoint by its
    // coefficient sign gives the same extrema as enumerating all eight corners.
    Vec3 lower, upper;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double a = cell_(i, j) * lo[j], b = cell_(i, j) * hi[j];
            lower[i] += std::min(a, b);
            upper[i] += std::max(a, b);
        }
    return {lower, upper};
}

void Box::min_image_triclinic_(double& dx, double& dy,
                               double& dz) const noexcept {
    const long double input[3] = {dx, dy, dz};
    long double y[3]{}, nearest[3]{};
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d)
            y[i] += mic_q_[3 * d + i] * input[d];
    // Babai nearest-plane seed. Translate the target before enumeration:
    // offsets stay small even when the input is many lattice images away.
    for (int i = 2; i >= 0; --i) {
        long double t = y[i];
        for (int j = i + 1; j < 3; ++j)
            t -= mic_r_[3 * i + j] * nearest[j];
        nearest[i] = std::nearbyint(t / mic_r_[3 * i + i]);
    }
    long double best[3]{}, residual[3]{}, index[3]{};
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j)
            y[i] -= mic_r_[3 * i + j] * nearest[j];
        if (!std::isfinite(y[i]))
            return;
        best[i] = y[i];
    }
    auto search = [&](auto&& self, int level) -> void {
        long double t = y[level];
        for (int j = level + 1; j < 3; ++j)
            t -= mic_r_[3 * level + j] * index[j];
        const long double center = t / mic_r_[3 * level + level];
        const long double first = std::nearbyint(center);
        auto remaining = [&]() {
            // Difference term-by-term avoids subtracting two huge total
            // norms when a long lattice axis has the same residual in both.
            long double budget = 0;
            for (int j = 0; j <= level; ++j)
                budget += best[j] * best[j];
            for (int j = level + 1; j < 3; ++j)
                budget += (best[j] - residual[j]) * (best[j] + residual[j]);
            return budget;
        };
        auto visit = [&](long double n) {
            index[level] = n;
            residual[level] = t - mic_r_[3 * level + level] * n;
            const long double cost = residual[level] * residual[level];
            const long double budget = remaining();
            if (cost > std::nextafter(
                    budget, std::numeric_limits<long double>::infinity()))
                return false;
            if (level > 0)
                self(self, level - 1);
            else {
                long double improvement = 0;
                for (int j = 0; j < 3; ++j)
                    improvement +=
                        (best[j] - residual[j]) * (best[j] + residual[j]);
                if (improvement > 0)
                    for (int j = 0; j < 3; ++j)
                        best[j] = residual[j];
            }
            return true;
        };
        visit(first);
        // The final coordinate is independently minimized by rounding.
        if (level == 0)
            return;
        const long double direction = center >= first ? 1 : -1;
        for (long double offset = 1;; offset += 1) {
            const bool plus = visit(first + direction * offset);
            const bool minus = visit(first - direction * offset);
            if (!plus && !minus)
                break;
        }
    };
    search(search, 2);
    long double result[3]{};
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < 3; ++i)
            result[d] += mic_q_[3 * d + i] * best[i];
    dx = static_cast<double>(result[0]);
    dy = static_cast<double>(result[1]);
    dz = static_cast<double>(result[2]);
}

} // namespace ATV::cpu
