/// @file src/cpu/bias/distance_cv.cpp
#include "atomiverse/cpu/bias/distance_cv.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace bias {

DistanceCV::DistanceCV(TagInt tag_i, TagInt tag_j)
    : tag_i_(tag_i), tag_j_(tag_j) {
    if (tag_i == tag_j)
        throw std::invalid_argument("DistanceCV: tags must differ");
}

void DistanceCV::build(const cpu::Frame& frame) {
    row_i_ = -1;
    row_j_ = -1;
    const int n = frame.nlocal();
    const auto tags = frame.span(kAtom_tag);
    for (int i = 0; i < n; ++i) {
        if (tags[i] == tag_i_)
            row_i_ = i;
        if (tags[i] == tag_j_)
            row_j_ = i;
    }
    if (row_i_ < 0 || row_j_ < 0)
        throw std::runtime_error(
            "DistanceCV: missing tag(s) in frame (need both atoms local for "
            "serial v1)");
}

double DistanceCV::evaluate(const cpu::Frame& frame, double* grad) const {
    if (row_i_ < 0 || row_j_ < 0)
        throw std::runtime_error("DistanceCV: build() not called");
    double dx = frame.span(kAtom_x)[row_i_] - frame.span(kAtom_x)[row_j_];
    double dy = frame.span(kAtom_y)[row_i_] - frame.span(kAtom_y)[row_j_];
    double dz = frame.span(kAtom_z)[row_i_] - frame.span(kAtom_z)[row_j_];
    frame.box.min_image(dx, dy, dz);
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 == 0.0)
        throw std::runtime_error("DistanceCV: zero separation");
    const double r = std::sqrt(r2);
    if (grad) {
        const int n = frame.nlocal();
        std::fill(grad, grad + 3 * n, 0.0);
        const double inv = 1.0 / r;
        grad[3 * row_i_ + 0] = dx * inv;
        grad[3 * row_i_ + 1] = dy * inv;
        grad[3 * row_i_ + 2] = dz * inv;
        grad[3 * row_j_ + 0] = -dx * inv;
        grad[3 * row_j_ + 1] = -dy * inv;
        grad[3 * row_j_ + 2] = -dz * inv;
    }
    return r;
}

} // namespace bias
} // namespace cpu
} // namespace ATV
