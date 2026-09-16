/// @file src/cpu/bias/dihedral_cv.cpp
#include "atomiverse/cpu/bias/dihedral_cv.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/potentials/dihedral_geom.h"

namespace ATV {
namespace cpu {
namespace bias {

DihedralCV::DihedralCV(TagInt tag_i, TagInt tag_j, TagInt tag_k, TagInt tag_l)
    : tag_i_(tag_i), tag_j_(tag_j), tag_k_(tag_k), tag_l_(tag_l) {
    if (tag_i == tag_j || tag_i == tag_k || tag_i == tag_l || tag_j == tag_k
        || tag_j == tag_l || tag_k == tag_l)
        throw std::invalid_argument("DihedralCV: tags must be distinct");
}

void DihedralCV::build(const cpu::Frame& frame) {
    row_i_ = row_j_ = row_k_ = row_l_ = -1;
    const int n = frame.nlocal();
    const auto tags = frame.span(kAtom_tag);
    for (int a = 0; a < n; ++a) {
        if (tags[a] == tag_i_)
            row_i_ = a;
        if (tags[a] == tag_j_)
            row_j_ = a;
        if (tags[a] == tag_k_)
            row_k_ = a;
        if (tags[a] == tag_l_)
            row_l_ = a;
    }
    if (row_i_ < 0 || row_j_ < 0 || row_k_ < 0 || row_l_ < 0)
        throw std::runtime_error(
            "DihedralCV: missing tag(s) in frame (need all four atoms local)");
}

double DihedralCV::evaluate(const cpu::Frame& frame, double* grad) const {
    if (row_i_ < 0 || row_j_ < 0 || row_k_ < 0 || row_l_ < 0)
        throw std::runtime_error("DihedralCV: build() not called");

    const double r1x = frame.span(kAtom_x)[row_i_];
    const double r1y = frame.span(kAtom_y)[row_i_];
    const double r1z = frame.span(kAtom_z)[row_i_];
    const double r2x = frame.span(kAtom_x)[row_j_];
    const double r2y = frame.span(kAtom_y)[row_j_];
    const double r2z = frame.span(kAtom_z)[row_j_];
    const double r3x = frame.span(kAtom_x)[row_k_];
    const double r3y = frame.span(kAtom_y)[row_k_];
    const double r3z = frame.span(kAtom_z)[row_k_];
    const double r4x = frame.span(kAtom_x)[row_l_];
    const double r4y = frame.span(kAtom_y)[row_l_];
    const double r4z = frame.span(kAtom_z)[row_l_];

    dihedral_geom::Geometry g;
    const bool ok = dihedral_geom::fill(
        r1x, r1y, r1z, r2x, r2y, r2z, r3x, r3y, r3z, r4x, r4y, r4z,
        [&](double& dx, double& dy, double& dz) {
            frame.box.min_image(dx, dy, dz);
        },
        g);
    if (!ok)
        throw std::runtime_error("DihedralCV: colinear dihedral (undefined φ)");

    const double phi = std::atan2(g.sin_phi, g.cos_phi);

    if (grad) {
        const int n = frame.nlocal();
        std::fill(grad, grad + 3 * n, 0.0);
        // dφ/dcos = d(atan2(sin,cos))/dcos with sin fixed is awkward;
        // use dV/dcos with V=φ ⇒ dφ/dcos = −1/sin (same as dE/dcos for E=φ).
        const double dphi_dcos =
            dihedral_geom::dV_dcos_from_dE_dphi(1.0, g.sin_phi, g.cos_phi, 0.0);
        // forces_from_dV_dcos returns F = −∇V; we need ∇φ, so flip sign.
        double f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;
        dihedral_geom::forces_from_dV_dcos(g, dphi_dcos, f1x, f1y, f1z, f2x,
                                           f2y, f2z, f3x, f3y, f3z, f4x, f4y,
                                           f4z);
        grad[3 * row_i_ + 0] = -f1x;
        grad[3 * row_i_ + 1] = -f1y;
        grad[3 * row_i_ + 2] = -f1z;
        grad[3 * row_j_ + 0] = -f2x;
        grad[3 * row_j_ + 1] = -f2y;
        grad[3 * row_j_ + 2] = -f2z;
        grad[3 * row_k_ + 0] = -f3x;
        grad[3 * row_k_ + 1] = -f3y;
        grad[3 * row_k_ + 2] = -f3z;
        grad[3 * row_l_ + 0] = -f4x;
        grad[3 * row_l_ + 1] = -f4y;
        grad[3 * row_l_ + 2] = -f4z;
    }
    return phi;
}

} // namespace bias
} // namespace cpu
} // namespace ATV
