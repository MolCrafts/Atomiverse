/// @file core/mat3.h
/// @brief Mat3 — 3x3 matrix OOP class with xtensor_fixed storage.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include <cmath>
#include <stdexcept>
#include <xtensor/containers/xfixed.hpp>

namespace ATV {

class Mat3 {
    xt::xtensor_fixed<AccReal, xt::xshape<3, 3>> m_{};

  public:
    Mat3() = default;

    AccReal& operator()(int i, int j) { return m_(i, j); }
    const AccReal& operator()(int i, int j) const { return m_(i, j); }

    Mat3 operator+(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = m_(i, j) + o(i, j);
        return r;
    }
    Mat3 operator-(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = m_(i, j) - o(i, j);
        return r;
    }
    Mat3 operator*(AccReal s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = m_(i, j) * s;
        return r;
    }
    friend Mat3 operator*(AccReal s, const Mat3& m) { return m * s; }
    Mat3 operator/(AccReal s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = m_(i, j) / s;
        return r;
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r(i, j) = 0;
                for (int k = 0; k < 3; ++k)
                    r(i, j) += m_(i, k) * o(k, j);
            }
        return r;
    }
    Vec3 operator*(const Vec3& v) const {
        return {m_(0, 0) * v[0] + m_(0, 1) * v[1] + m_(0, 2) * v[2],
                m_(1, 0) * v[0] + m_(1, 1) * v[1] + m_(1, 2) * v[2],
                m_(2, 0) * v[0] + m_(2, 1) * v[1] + m_(2, 2) * v[2]};
    }

    Mat3 transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = m_(j, i);
        return r;
    }
    AccReal trace() const { return m_(0, 0) + m_(1, 1) + m_(2, 2); }
    AccReal norm() const {
        AccReal s = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                s += m_(i, j) * m_(i, j);
        return std::sqrt(s);
    }
    AccReal det() const {
        return m_(0, 0) * (m_(1, 1) * m_(2, 2) - m_(1, 2) * m_(2, 1))
               - m_(0, 1) * (m_(1, 0) * m_(2, 2) - m_(1, 2) * m_(2, 0))
               + m_(0, 2) * (m_(1, 0) * m_(2, 1) - m_(1, 1) * m_(2, 0));
    }
    Mat3 inverse() const {
        AccReal d = det();
        if (std::abs(d) < AccReal(1e-30))
            throw std::runtime_error("Mat3::inverse: singular matrix");
        AccReal inv = AccReal(1) / d;
        Mat3 r;
        r(0, 0) = (m_(1, 1) * m_(2, 2) - m_(1, 2) * m_(2, 1)) * inv;
        r(0, 1) = (m_(0, 2) * m_(2, 1) - m_(0, 1) * m_(2, 2)) * inv;
        r(0, 2) = (m_(0, 1) * m_(1, 2) - m_(0, 2) * m_(1, 1)) * inv;
        r(1, 0) = (m_(1, 2) * m_(2, 0) - m_(1, 0) * m_(2, 2)) * inv;
        r(1, 1) = (m_(0, 0) * m_(2, 2) - m_(0, 2) * m_(2, 0)) * inv;
        r(1, 2) = (m_(0, 2) * m_(1, 0) - m_(0, 0) * m_(1, 2)) * inv;
        r(2, 0) = (m_(1, 0) * m_(2, 1) - m_(1, 1) * m_(2, 0)) * inv;
        r(2, 1) = (m_(0, 1) * m_(2, 0) - m_(0, 0) * m_(2, 1)) * inv;
        r(2, 2) = (m_(0, 0) * m_(1, 1) - m_(0, 1) * m_(1, 0)) * inv;
        return r;
    }

    static Mat3 zeros() { return Mat3{}; }
    static Mat3 identity() {
        Mat3 m;
        m(0, 0) = m(1, 1) = m(2, 2) = 1;
        return m;
    }
    static Mat3 diagonal(AccReal a, AccReal b, AccReal c) {
        Mat3 m;
        m(0, 0) = a;
        m(1, 1) = b;
        m(2, 2) = c;
        return m;
    }
    static Mat3 cubic(AccReal l) { return diagonal(l, l, l); }
};

} // namespace ATV
