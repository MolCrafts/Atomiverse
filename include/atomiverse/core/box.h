/// @file core/box.h
/// @brief Box — simulation cell value type with periodic boundary operations.
#pragma once

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include <cmath>

namespace ATV {

struct Box {
    Mat3 cell;
    Vec3 origin;
    bool pbc[3] = {true, true, true};

    static Box cubic(AccReal a) {
        Box b;
        b.cell = Mat3::diagonal(a, a, a);
        b.origin = Vec3::zeros();
        return b;
    }

    AccReal volume() const { return std::abs(cell.det()); }

    Mat3 recip_cell() const {
        return cell.inverse().transpose() * AccReal(2.0 * units::pi);
    }

    Vec3 lengths() const {
        return {std::sqrt(cell(0, 0) * cell(0, 0) + cell(0, 1) * cell(0, 1)
                          + cell(0, 2) * cell(0, 2)),
                std::sqrt(cell(1, 0) * cell(1, 0) + cell(1, 1) * cell(1, 1)
                          + cell(1, 2) * cell(1, 2)),
                std::sqrt(cell(2, 0) * cell(2, 0) + cell(2, 1) * cell(2, 1)
                          + cell(2, 2) * cell(2, 2))};
    }

    Vec3 to_frac(const Vec3& cart) const {
        auto inv = cell.inverse();
        auto d =
            Vec3{cart[0] - origin[0], cart[1] - origin[1], cart[2] - origin[2]};
        return inv * d;
    }

    Vec3 to_cart(const Vec3& frac) const {
        auto cart = cell * frac;
        return {cart[0] + origin[0], cart[1] + origin[1], cart[2] + origin[2]};
    }

    Vec3 wrap(const Vec3& cart) const {
        auto f = to_frac(cart);
        for (int i = 0; i < 3; ++i) {
            if (pbc[i]) {
                f[i] -= std::floor(f[i]);
                if (f[i] < 0)
                    f[i] += AccReal(1);
            }
        }
        return to_cart(f);
    }

    void wrap_batch(AccReal* xyz, int n) const {
        for (int i = 0; i < n; ++i) {
            auto w = wrap(Vec3{xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]});
            xyz[i * 3] = w[0];
            xyz[i * 3 + 1] = w[1];
            xyz[i * 3 + 2] = w[2];
        }
    }

    Vec3 shortest_vector(const Vec3& r1, const Vec3& r2) const {
        auto f1 = to_frac(r1);
        auto f2 = to_frac(r2);
        Vec3 df;
        for (int i = 0; i < 3; ++i) {
            df[i] = f2[i] - f1[i];
            if (pbc[i]) {
                df[i] -= std::round(df[i]);
            }
        }
        return cell * df;
    }
};

} // namespace ATV
