/// @file src/cpu/bias/angle_cv.cpp
#include "atomiverse/cpu/bias/angle_cv.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace bias {

AngleCV::AngleCV(TagInt tag_i, TagInt tag_j, TagInt tag_k)
    : tag_i_(tag_i), tag_j_(tag_j), tag_k_(tag_k) {
    if (tag_i == tag_j || tag_j == tag_k || tag_i == tag_k)
        throw std::invalid_argument("AngleCV: tags must be distinct");
}

void AngleCV::build(const cpu::Frame& frame) {
    row_i_ = row_j_ = row_k_ = -1;
    const int n = frame.nlocal();
    const auto tags = frame.span(kAtom_tag);
    for (int a = 0; a < n; ++a) {
        if (tags[a] == tag_i_)
            row_i_ = a;
        if (tags[a] == tag_j_)
            row_j_ = a;
        if (tags[a] == tag_k_)
            row_k_ = a;
    }
    if (row_i_ < 0 || row_j_ < 0 || row_k_ < 0)
        throw std::runtime_error(
            "AngleCV: missing tag(s) in frame (need all three atoms local)");
}

double AngleCV::evaluate(const cpu::Frame& frame, double* grad) const {
    if (row_i_ < 0 || row_j_ < 0 || row_k_ < 0)
        throw std::runtime_error("AngleCV: build() not called");

    // Vectors from vertex j: r_ji, r_jk
    double xi = frame.span(kAtom_x)[row_i_] - frame.span(kAtom_x)[row_j_];
    double yi = frame.span(kAtom_y)[row_i_] - frame.span(kAtom_y)[row_j_];
    double zi = frame.span(kAtom_z)[row_i_] - frame.span(kAtom_z)[row_j_];
    double xk = frame.span(kAtom_x)[row_k_] - frame.span(kAtom_x)[row_j_];
    double yk = frame.span(kAtom_y)[row_k_] - frame.span(kAtom_y)[row_j_];
    double zk = frame.span(kAtom_z)[row_k_] - frame.span(kAtom_z)[row_j_];
    frame.box.min_image(xi, yi, zi);
    frame.box.min_image(xk, yk, zk);

    const double ri = std::sqrt(xi * xi + yi * yi + zi * zi);
    const double rk = std::sqrt(xk * xk + yk * yk + zk * zk);
    if (!(ri > 0.0) || !(rk > 0.0))
        throw std::runtime_error("AngleCV: zero bond length");

    const double cos_th = (xi * xk + yi * yk + zi * zk) / (ri * rk);
    const double c = std::max(-1.0, std::min(1.0, cos_th));
    const double theta = std::acos(c);

    if (grad) {
        const int n = frame.nlocal();
        std::fill(grad, grad + 3 * n, 0.0);
        const double sin_th = std::sin(theta);
        if (sin_th < 1e-12) {
            // Collinear: gradient ill-defined; leave zeros.
            return theta;
        }
        // d(cos)/d r_i = (r_k / (|ri||rk|) − cos r_i / |ri|²) etc. Wilson form:
        // ∇_i θ = (cos θ · û_i − û_k) / (|ri| sin θ)
        // ∇_k θ = (cos θ · û_k − û_i) / (|rk| sin θ)
        // ∇_j θ = −∇_i − ∇_k
        const double inv_ri = 1.0 / ri;
        const double inv_rk = 1.0 / rk;
        const double uix = xi * inv_ri, uiy = yi * inv_ri, uiz = zi * inv_ri;
        const double ukx = xk * inv_rk, uky = yk * inv_rk, ukz = zk * inv_rk;
        const double inv_sin = 1.0 / sin_th;
        const double gix = (c * uix - ukx) * inv_ri * inv_sin;
        const double giy = (c * uiy - uky) * inv_ri * inv_sin;
        const double giz = (c * uiz - ukz) * inv_ri * inv_sin;
        const double gkx = (c * ukx - uix) * inv_rk * inv_sin;
        const double gky = (c * uky - uiy) * inv_rk * inv_sin;
        const double gkz = (c * ukz - uiz) * inv_rk * inv_sin;
        grad[3 * row_i_ + 0] = gix;
        grad[3 * row_i_ + 1] = giy;
        grad[3 * row_i_ + 2] = giz;
        grad[3 * row_k_ + 0] = gkx;
        grad[3 * row_k_ + 1] = gky;
        grad[3 * row_k_ + 2] = gkz;
        grad[3 * row_j_ + 0] = -gix - gkx;
        grad[3 * row_j_ + 1] = -giy - gky;
        grad[3 * row_j_ + 2] = -giz - gkz;
    }
    return theta;
}

} // namespace bias
} // namespace cpu
} // namespace ATV
