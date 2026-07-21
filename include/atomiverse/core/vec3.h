/// @file core/vec3.h
/// @brief Vec3 — 3D vector value type.
#pragma once

#include "atomiverse/core/precision.h"
#include <cmath>
#include <xtensor/containers/xfixed.hpp>

namespace ATV {

class Vec3 {
    xt::xtensor_fixed<AccReal, xt::xshape<3>> v_{};

  public:
    Vec3() = default;
    Vec3(AccReal x, AccReal y, AccReal z) { v_ = {x, y, z}; }

    AccReal& operator[](int i) { return v_(i); }
    const AccReal& operator[](int i) const { return v_(i); }

    AccReal& x() { return v_(0); }
    AccReal& y() { return v_(1); }
    AccReal& z() { return v_(2); }
    AccReal x() const { return v_(0); }
    AccReal y() const { return v_(1); }
    AccReal z() const { return v_(2); }

    Vec3 operator+(const Vec3& o) const {
        return {v_(0) + o.v_(0), v_(1) + o.v_(1), v_(2) + o.v_(2)};
    }
    Vec3 operator-(const Vec3& o) const {
        return {v_(0) - o.v_(0), v_(1) - o.v_(1), v_(2) - o.v_(2)};
    }
    Vec3 operator*(AccReal s) const {
        return {v_(0) * s, v_(1) * s, v_(2) * s};
    }
    Vec3 operator/(AccReal s) const {
        return {v_(0) / s, v_(1) / s, v_(2) / s};
    }

    AccReal dot(const Vec3& o) const {
        return v_(0) * o.v_(0) + v_(1) * o.v_(1) + v_(2) * o.v_(2);
    }
    Vec3 cross(const Vec3& o) const {
        return {v_(1) * o.v_(2) - v_(2) * o.v_(1),
                v_(2) * o.v_(0) - v_(0) * o.v_(2),
                v_(0) * o.v_(1) - v_(1) * o.v_(0)};
    }
    AccReal norm_squared() const { return dot(*this); }
    AccReal norm() const { return std::sqrt(norm_squared()); }
    Vec3 normalized() const {
        AccReal n = norm();
        return {v_(0) / n, v_(1) / n, v_(2) / n};
    }

    static Vec3 zeros() { return {0, 0, 0}; }
    static Vec3 unit_x() { return {1, 0, 0}; }
    static Vec3 unit_y() { return {0, 1, 0}; }
    static Vec3 unit_z() { return {0, 0, 1}; }
};

} // namespace ATV
