/// @file src/cpu/dynamics/virtual_sites.cpp
#include "atomiverse/cpu/dynamics/virtual_sites.h"

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

namespace {

inline void cross(double ax, double ay, double az, double bx, double by,
                  double bz, double& cx, double& cy, double& cz) {
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

inline double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

} // namespace

void VirtualSites::add_weighted3(TagInt site_tag, TagInt p0, TagInt p1,
                                 TagInt p2, double alpha) {
    if (!(std::isfinite(alpha)))
        throw std::invalid_argument(
            "VirtualSites::add_weighted3: alpha must be finite");
    Site s;
    s.kind = Kind::Weighted3;
    s.site = site_tag;
    s.p0 = p0;
    s.p1 = p1;
    s.p2 = p2;
    s.param[0] = alpha;
    sites_.push_back(s);
    n_resolved_ = -1;
}

void VirtualSites::add_out_of_plane3(TagInt site_tag, TagInt p0, TagInt p1,
                                     TagInt p2, double a, double b, double c) {
    if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c)))
        throw std::invalid_argument(
            "VirtualSites::add_out_of_plane3: a,b,c must be finite");
    Site s;
    s.kind = Kind::OutOfPlane3;
    s.site = site_tag;
    s.p0 = p0;
    s.p1 = p1;
    s.p2 = p2;
    s.param[0] = a;
    s.param[1] = b;
    s.param[2] = c;
    sites_.push_back(s);
    n_resolved_ = -1;
}

double VirtualSites::tip4p_alpha(double qdist, double r_oh, double theta_hoh) {
    if (qdist < 0.0)
        throw std::invalid_argument("tip4p_alpha: qdist must be ≥ 0");
    if (!(r_oh > 0.0))
        throw std::invalid_argument("tip4p_alpha: r_oh must be > 0");
    if (!(theta_hoh > 0.0) || !(theta_hoh < ATV::units::pi))
        throw std::invalid_argument("tip4p_alpha: theta_hoh must be in (0, π)");
    const double c = std::cos(0.5 * theta_hoh);
    if (!(c > 0.0))
        throw std::invalid_argument("tip4p_alpha: cos(θ/2) must be > 0");
    return qdist / (c * r_oh);
}

void VirtualSites::tip5p_out_of_plane_coeffs(double r_ol, double theta_lol,
                                             double r_oh, double theta_hoh,
                                             int sign, double& a, double& b,
                                             double& c) {
    if (!(r_ol > 0.0) || !(r_oh > 0.0))
        throw std::invalid_argument(
            "tip5p_out_of_plane_coeffs: distances must be > 0");
    if (!(theta_lol > 0.0) || !(theta_lol < ATV::units::pi)
        || !(theta_hoh > 0.0) || !(theta_hoh < ATV::units::pi))
        throw std::invalid_argument(
            "tip5p_out_of_plane_coeffs: angles must be in (0, π)");
    if (sign != 1 && sign != -1)
        throw std::invalid_argument(
            "tip5p_out_of_plane_coeffs: sign must be ±1");

    // Equilibrium H positions in the xz plane (O at origin).
    const double half_h = 0.5 * theta_hoh;
    const double d1x = r_oh * std::sin(half_h);
    const double d1y = 0.0;
    const double d1z = r_oh * std::cos(half_h);
    const double d2x = -r_oh * std::sin(half_h);
    const double d2y = 0.0;
    const double d2z = r_oh * std::cos(half_h);

    // Bisector (toward H) and plane normal.
    double bx = d1x + d2x, by = d1y + d2y, bz = d1z + d2z;
    const double bn = norm3(bx, by, bz);
    if (!(bn > 0.0))
        throw std::invalid_argument(
            "tip5p_out_of_plane_coeffs: degenerate HOH bisector");
    bx /= bn;
    by /= bn;
    bz /= bn;

    double nx, ny, nz;
    cross(d1x, d1y, d1z, d2x, d2y, d2z, nx, ny, nz);
    const double nn = norm3(nx, ny, nz);
    if (!(nn > 0.0))
        throw std::invalid_argument("tip5p_out_of_plane_coeffs: collinear HOH");
    nx /= nn;
    ny /= nn;
    nz /= nn;

    // L direction: opposite the H-bisector, ± half of LOL about the plane normal.
    const double half_l = 0.5 * theta_lol;
    const double cl = std::cos(half_l);
    const double sl = std::sin(half_l);
    const double s = static_cast<double>(sign);
    double ux = -cl * bx + s * sl * nx;
    double uy = -cl * by + s * sl * ny;
    double uz = -cl * bz + s * sl * nz;
    const double un = norm3(ux, uy, uz);
    ux /= un;
    uy /= un;
    uz /= un;

    const double tx = r_ol * ux;
    const double ty = r_ol * uy;
    const double tz = r_ol * uz;

    // Basis columns: d1, d2, d1×d2 — solve M [a,b,c]^T = target.
    double cx, cy, cz;
    cross(d1x, d1y, d1z, d2x, d2y, d2z, cx, cy, cz);

    // 3×3 determinant
    const double det = d1x * (d2y * cz - d2z * cy) - d1y * (d2x * cz - d2z * cx)
                       + d1z * (d2x * cy - d2y * cx);
    if (!(std::fabs(det) > 1e-18))
        throw std::invalid_argument(
            "tip5p_out_of_plane_coeffs: singular parent basis");

    // Cramer's rule
    a = (tx * (d2y * cz - d2z * cy) - ty * (d2x * cz - d2z * cx)
         + tz * (d2x * cy - d2y * cx))
        / det;
    b = (d1x * (ty * cz - tz * cy) - d1y * (tx * cz - tz * cx)
         + d1z * (tx * cy - ty * cx))
        / det;
    c = (d1x * (d2y * tz - d2z * ty) - d1y * (d2x * tz - d2z * tx)
         + d1z * (d2x * ty - d2y * tx))
        / det;
}

void VirtualSites::resolve_(Frame& f) {
    const int n = f.n_total();
    if (n_resolved_ == n) {
        // Indices still valid if topology/tags fixed for the run.
        return;
    }
    auto& index_mut = f.meta.get(kTagIndex);
    if (index_mut.empty())
        f.rebuild_tag_index(0, f.nlocal());
    const auto& index = f.meta.get(kTagIndex);
    for (auto& s : sites_) {
        auto req = [&](TagInt t, const char* who) -> int {
            auto it = index.find(t);
            if (it == index.end())
                throw std::runtime_error(
                    std::string("VirtualSites: missing tag for ") + who);
            return it->second;
        };
        s.is = req(s.site, "site");
        s.i0 = req(s.p0, "parent0");
        s.i1 = req(s.p1, "parent1");
        s.i2 = req(s.p2, "parent2");
    }
    n_resolved_ = n;
}

void VirtualSites::place(Frame& f) {
    if (sites_.empty())
        return;
    resolve_(f);

    StorageReal* __restrict x = f.span(kAtom_x);
    StorageReal* __restrict y = f.span(kAtom_y);
    StorageReal* __restrict z = f.span(kAtom_z);
    const auto& box = f.box;

    for (const auto& s : sites_) {
        const int i0 = s.i0, i1 = s.i1, i2 = s.i2, is = s.is;
        // Closest-image parents relative to p0 (LAMMPS tip4p convention).
        double d1x = static_cast<double>(x[i1]) - static_cast<double>(x[i0]);
        double d1y = static_cast<double>(y[i1]) - static_cast<double>(y[i0]);
        double d1z = static_cast<double>(z[i1]) - static_cast<double>(z[i0]);
        box.min_image(d1x, d1y, d1z);
        double d2x = static_cast<double>(x[i2]) - static_cast<double>(x[i0]);
        double d2y = static_cast<double>(y[i2]) - static_cast<double>(y[i0]);
        double d2z = static_cast<double>(z[i2]) - static_cast<double>(z[i0]);
        box.min_image(d2x, d2y, d2z);

        double sx, sy, sz;
        if (s.kind == Kind::Weighted3) {
            const double alpha = s.param[0];
            // r_s = r0 + (α/2)*(d1+d2) = (1-α)r0 + (α/2)r1 + (α/2)r2
            sx = static_cast<double>(x[i0]) + 0.5 * alpha * (d1x + d2x);
            sy = static_cast<double>(y[i0]) + 0.5 * alpha * (d1y + d2y);
            sz = static_cast<double>(z[i0]) + 0.5 * alpha * (d1z + d2z);
        } else {
            const double a = s.param[0], b = s.param[1], c = s.param[2];
            double cx, cy, cz;
            cross(d1x, d1y, d1z, d2x, d2y, d2z, cx, cy, cz);
            sx = static_cast<double>(x[i0]) + a * d1x + b * d2x + c * cx;
            sy = static_cast<double>(y[i0]) + a * d1y + b * d2y + c * cy;
            sz = static_cast<double>(z[i0]) + a * d1z + b * d2z + c * cz;
        }
        x[is] = static_cast<StorageReal>(sx);
        y[is] = static_cast<StorageReal>(sy);
        z[is] = static_cast<StorageReal>(sz);
    }
}

void VirtualSites::redistribute(Frame& f) {
    if (sites_.empty())
        return;
    resolve_(f);

    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const auto& box = f.box;

    for (const auto& s : sites_) {
        const int i0 = s.i0, i1 = s.i1, i2 = s.i2, is = s.is;
        const double Fsx = static_cast<double>(fx[is]);
        const double Fsy = static_cast<double>(fy[is]);
        const double Fsz = static_cast<double>(fz[is]);

        if (s.kind == Kind::Weighted3) {
            const double alpha = s.param[0];
            const double w0 = 1.0 - alpha;
            const double wh = 0.5 * alpha;
            fx[i0] += static_cast<AccReal>(w0 * Fsx);
            fy[i0] += static_cast<AccReal>(w0 * Fsy);
            fz[i0] += static_cast<AccReal>(w0 * Fsz);
            fx[i1] += static_cast<AccReal>(wh * Fsx);
            fy[i1] += static_cast<AccReal>(wh * Fsy);
            fz[i1] += static_cast<AccReal>(wh * Fsz);
            fx[i2] += static_cast<AccReal>(wh * Fsx);
            fy[i2] += static_cast<AccReal>(wh * Fsy);
            fz[i2] += static_cast<AccReal>(wh * Fsz);
        } else {
            // Linear part + cross-term Jacobian (GROMACS vsite3OUT).
            const double a = s.param[0], b = s.param[1], cc = s.param[2];
            double d1x =
                static_cast<double>(x[i1]) - static_cast<double>(x[i0]);
            double d1y =
                static_cast<double>(y[i1]) - static_cast<double>(y[i0]);
            double d1z =
                static_cast<double>(z[i1]) - static_cast<double>(z[i0]);
            box.min_image(d1x, d1y, d1z);
            double d2x =
                static_cast<double>(x[i2]) - static_cast<double>(x[i0]);
            double d2y =
                static_cast<double>(y[i2]) - static_cast<double>(y[i0]);
            double d2z =
                static_cast<double>(z[i2]) - static_cast<double>(z[i0]);
            box.min_image(d2x, d2y, d2z);

            // F0 += (1-a-b) F; F1 += a F; F2 += b F
            const double w0 = 1.0 - a - b;
            fx[i0] += static_cast<AccReal>(w0 * Fsx);
            fy[i0] += static_cast<AccReal>(w0 * Fsy);
            fz[i0] += static_cast<AccReal>(w0 * Fsz);
            fx[i1] += static_cast<AccReal>(a * Fsx);
            fy[i1] += static_cast<AccReal>(a * Fsy);
            fz[i1] += static_cast<AccReal>(a * Fsz);
            fx[i2] += static_cast<AccReal>(b * Fsx);
            fy[i2] += static_cast<AccReal>(b * Fsy);
            fz[i2] += static_cast<AccReal>(b * Fsz);

            // Cross term: t1 = c (d2 × F), t2 = c (F × d1)
            double t1x, t1y, t1z, t2x, t2y, t2z;
            cross(d2x, d2y, d2z, Fsx, Fsy, Fsz, t1x, t1y, t1z);
            t1x *= cc;
            t1y *= cc;
            t1z *= cc;
            cross(Fsx, Fsy, Fsz, d1x, d1y, d1z, t2x, t2y, t2z);
            t2x *= cc;
            t2y *= cc;
            t2z *= cc;
            fx[i0] += static_cast<AccReal>(-t1x - t2x);
            fy[i0] += static_cast<AccReal>(-t1y - t2y);
            fz[i0] += static_cast<AccReal>(-t1z - t2z);
            fx[i1] += static_cast<AccReal>(t1x);
            fy[i1] += static_cast<AccReal>(t1y);
            fz[i1] += static_cast<AccReal>(t1z);
            fx[i2] += static_cast<AccReal>(t2x);
            fy[i2] += static_cast<AccReal>(t2y);
            fz[i2] += static_cast<AccReal>(t2z);
        }

        // Site is massless construction — clear residual force.
        fx[is] = static_cast<AccReal>(0);
        fy[is] = static_cast<AccReal>(0);
        fz[is] = static_cast<AccReal>(0);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
