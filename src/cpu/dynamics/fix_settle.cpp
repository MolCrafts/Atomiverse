/// @file src/cpu/dynamics/fix_settle.cpp
/// @brief Analytic SETTLE for 3-site rigid water (Miyamoto–Kollman 1992).
///
/// The published closed form: translate the reference (constraint-satisfying)
/// triangle to the centre of mass of the unconstrained one, build the body
/// frame from the reference triangle's normal and the unconstrained oxygen,
/// and recover the three rotation angles φ, ψ, θ analytically. That is the
/// exact SHAKE answer for a three-site rigid body — not a reshaping of the
/// unconstrained triangle, which would reproduce the bond lengths while losing
/// the orientation entirely.
///
/// Ref: S. Miyamoto and P. A. Kollman, "SETTLE: An analytical version of the
/// SHAKE and RATTLE algorithm for rigid water models", J. Comput. Chem. 13,
/// 952 (1992), Appendix.

#include "atomiverse/cpu/dynamics/fix_settle.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {

namespace {

/// Cross product, written out because the three components are read
/// individually everywhere below.
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 normalized(const Vec3& v) {
    const double n = std::sqrt(dot(v, v));
    if (!(n > 0.0))
        throw std::runtime_error(
            "FixSettle: degenerate water geometry — the three sites are "
            "collinear or coincident, so the body frame is undefined");
    return {v.x / n, v.y / n, v.z / n};
}

/// `sqrt` of a quantity that is non-negative in exact arithmetic; rounding can
/// push it a few ulps below zero at the extremes (|sinφ| → 1).
double safe_sqrt(double v) { return std::sqrt(v > 0.0 ? v : 0.0); }

} // namespace

FixSettle::FixSettle(double dt, double d_oh, double d_hh)
    : dt_(dt), d_oh_(d_oh), d_hh_(d_hh) {
    if (!(dt > 0.0))
        throw std::invalid_argument("FixSettle: dt must be > 0");
    if (!(d_oh > 0.0) || !(d_hh > 0.0))
        throw std::invalid_argument("FixSettle: distances must be > 0");
    if (d_hh >= 2.0 * d_oh)
        throw std::invalid_argument("FixSettle: triangle inequality violated");
}

void FixSettle::add_water(TagInt tag_O, TagInt tag_H1, TagInt tag_H2) {
    if (tag_O == tag_H1 || tag_O == tag_H2 || tag_H1 == tag_H2)
        throw std::invalid_argument("FixSettle: water tags must be distinct");
    waters_.push_back({tag_O, tag_H1, tag_H2, -1, -1, -1});
}

bool FixSettle::try_resolve_rows_(const Frame& f) {
    // Tags are the identity, rows are not: `atom_modify sort` spatially
    // reorders owned atoms at reneighbour, so a row cached at build() points
    // at a different atom a thousand steps later. Resolve through the Frame's
    // tag index — O(1) per water and refreshed by the sort itself.
    const TagIndex& tag_index = f.meta.get(kTagIndex);
    for (auto& w : waters_) {
        const auto io = tag_index.find(w.o);
        const auto ih1 = tag_index.find(w.h1);
        const auto ih2 = tag_index.find(w.h2);
        if (io == tag_index.end() || ih1 == tag_index.end()
            || ih2 == tag_index.end())
            return false;
        w.io = io->second;
        w.ih1 = ih1->second;
        w.ih2 = ih2->second;
    }
    return true;
}

void FixSettle::resolve_rows_(Frame& f) {
    if (try_resolve_rows_(f))
        return;
    // The index is published and maintained by the engine: this component
    // declares `requires_tag_index()`, so `DomainRuntime` keeps it alive for
    // the run (and rebuilds it for the owned rows after the initial partition),
    // and `md::notify_sorted` republishes it after every spatial reorder.
    // Reaching here therefore means a tag this fix was given is not on this
    // frame — a registration error, not a bookkeeping gap, and rebuilding the
    // map would only hide it.
    for (const auto& w : waters_) {
        const TagIndex& tag_index = f.meta.get(kTagIndex);
        if (tag_index.count(w.o) == 0 || tag_index.count(w.h1) == 0
            || tag_index.count(w.h2) == 0)
            throw std::runtime_error(
                "FixSettle: water ("
                + std::to_string(static_cast<long long>(w.o)) + ", "
                + std::to_string(static_cast<long long>(w.h1)) + ", "
                + std::to_string(static_cast<long long>(w.h2))
                + ") names a tag no row of this frame carries");
    }
    throw std::runtime_error(
        "FixSettle: the Frame tag index does not resolve a registered water. "
        "Every tag is present, so the index is stale — whatever renumbered the "
        "rows did not republish it (see md::notify_sorted).");
}

void FixSettle::build(const BuildContext& ctx) {
    if (waters_.empty())
        throw std::runtime_error("FixSettle: no waters registered");
    resolve_rows_(ctx.frame);

    // The closed form exists because the triangle is isosceles about the
    // oxygen: equal H masses are a precondition, not a convenience.
    const auto mass = ctx.frame.span(kAtom_mass);
    for (const auto& w : waters_) {
        const double ma = mass[w.ih1], mb = mass[w.ih2];
        if (!(ma > 0.0) || !(mass[w.io] > 0.0))
            throw std::runtime_error("FixSettle: water masses must be > 0");
        if (std::fabs(ma - mb) > 1e-12 * std::max(ma, mb))
            throw std::runtime_error(
                "FixSettle: the two hydrogens of a water must have equal mass "
                "— SETTLE's closed form assumes an isosceles triangle. Use "
                "FixRattle for an asymmetric rigid triple.");
    }
    ref_.assign(waters_.size() * 9, 0.0);
}

void FixSettle::snapshot_(const Frame& f) {
    // Only the registered waters, and only their own nine coordinates: the
    // reference is a per-molecule quantity, so an N-sized shadow of the whole
    // frame would be O(N) of pure waste every step.
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    if (ref_.size() != waters_.size() * 9)
        ref_.assign(waters_.size() * 9, 0.0);
    for (std::size_t k = 0; k < waters_.size(); ++k) {
        const Water& w = waters_[k];
        double* r = ref_.data() + 9 * k;
        // Unwrap about the oxygen: the integrator wraps coordinates into the
        // box, so a molecule straddling a face has its hydrogens an image away.
        double h1x = x[w.ih1] - x[w.io], h1y = y[w.ih1] - y[w.io],
               h1z = z[w.ih1] - z[w.io];
        double h2x = x[w.ih2] - x[w.io], h2y = y[w.ih2] - y[w.io],
               h2z = z[w.ih2] - z[w.io];
        f.box.min_image(h1x, h1y, h1z);
        f.box.min_image(h2x, h2y, h2z);
        r[0] = x[w.io];
        r[1] = y[w.io];
        r[2] = z[w.io];
        r[3] = r[0] + h1x;
        r[4] = r[1] + h1y;
        r[5] = r[2] + h1z;
        r[6] = r[0] + h2x;
        r[7] = r[1] + h2y;
        r[8] = r[2] + h2z;
    }
}

void FixSettle::settle_positions_(Frame& f) const {
    // Canonical triangle in its own centre-of-mass frame (Miyamoto–Kollman
    // eq. A1): O at (0, +ra, 0), H at (∓rc, −rb, 0).
    const double rc = 0.5 * d_hh_;
    const double height = std::sqrt(d_oh_ * d_oh_ - rc * rc);

    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    const auto mass = f.span(kAtom_mass);

    for (std::size_t k = 0; k < waters_.size(); ++k) {
        const Water& w = waters_[k];
        const int o = w.io, a = w.ih1, b = w.ih2;
        const double m_o = mass[o], m_h = mass[a];
        const double m_total = m_o + 2.0 * m_h;
        const double ra = 2.0 * m_h * height / m_total;
        const double rb = height - ra;

        // Unconstrained triangle, unwrapped about the oxygen for the same
        // reason the reference is.
        const Vec3 o1{x[o], y[o], z[o]};
        double b1x = x[a] - o1.x, b1y = y[a] - o1.y, b1z = z[a] - o1.z;
        double c1x = x[b] - o1.x, c1y = y[b] - o1.y, c1z = z[b] - o1.z;
        f.box.min_image(b1x, b1y, b1z);
        f.box.min_image(c1x, c1y, c1z);

        // Centre of mass of the unconstrained configuration — SETTLE keeps it
        // exactly where the unconstrained update put it.
        const Vec3 com{o1.x + m_h * (b1x + c1x) / m_total,
                       o1.y + m_h * (b1y + c1y) / m_total,
                       o1.z + m_h * (b1z + c1z) / m_total};
        const Vec3 a1{o1.x - com.x, o1.y - com.y, o1.z - com.z};
        const Vec3 b1{o1.x + b1x - com.x, o1.y + b1y - com.y,
                      o1.z + b1z - com.z};
        const Vec3 c1{o1.x + c1x - com.x, o1.y + c1y - com.y,
                      o1.z + c1z - com.z};

        // Reference bond vectors (constraint-satisfying, pre-move).
        const double* r = ref_.data() + 9 * k;
        const Vec3 b0{r[3] - r[0], r[4] - r[1], r[5] - r[2]};
        const Vec3 c0{r[6] - r[0], r[7] - r[1], r[8] - r[2]};

        // Body frame: z′ normal to the REFERENCE triangle, x′ ⊥ (oxygen, z′).
        const Vec3 ez = normalized(cross(b0, c0));
        const Vec3 ex = normalized(cross(a1, ez));
        const Vec3 ey = cross(ez, ex);

        // Project. By construction a1·x′ = 0 and the reference bonds have no
        // z′ component, so only the entries the algebra needs are formed.
        const double xb0d = dot(b0, ex), yb0d = dot(b0, ey);
        const double xc0d = dot(c0, ex), yc0d = dot(c0, ey);
        const double za1d = dot(a1, ez);
        const double xb1d = dot(b1, ex), yb1d = dot(b1, ey), zb1d = dot(b1, ez);
        const double xc1d = dot(c1, ex), yc1d = dot(c1, ey), zc1d = dot(c1, ez);

        // Rotations φ (about x′) and ψ (about y′) — eq. A6/A7.
        const double sinphi = za1d / ra;
        const double cosphi = safe_sqrt(1.0 - sinphi * sinphi);
        if (!(cosphi > 0.0))
            throw std::runtime_error(
                "FixSettle: the unconstrained oxygen left the plane by more "
                "than the molecule's own size — the timestep is far too large "
                "for this constraint");
        const double sinpsi = (zb1d - zc1d) / (2.0 * rc * cosphi);
        const double cospsi = safe_sqrt(1.0 - sinpsi * sinpsi);

        const double ya2d = ra * cosphi;
        const double xb2d = -rc * cospsi;
        const double t1 = -rb * cosphi;
        const double t2 = rc * sinpsi * sinphi;
        const double yb2d = t1 - t2;
        const double yc2d = t1 + t2;

        // Rotation θ about z′ — eq. A11: the one angle that is fixed by the
        // reference orientation, and the whole reason the reference matters.
        const double alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
        const double beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
        const double gamma =
            xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
        const double al2be2 = alpha * alpha + beta * beta;
        if (!(al2be2 > 0.0))
            throw std::runtime_error(
                "FixSettle: degenerate reference triangle (α² + β² = 0)");
        const double sinthe =
            (alpha * gamma - beta * safe_sqrt(al2be2 - gamma * gamma)) / al2be2;
        const double costhe = safe_sqrt(1.0 - sinthe * sinthe);

        // Constrained positions in the body frame — eq. A12.
        const double xa3d = -ya2d * sinthe;
        const double ya3d = ya2d * costhe;
        const double za3d = za1d;
        const double xb3d = xb2d * costhe - yb2d * sinthe;
        const double yb3d = xb2d * sinthe + yb2d * costhe;
        const double zb3d = zb1d;
        const double xc3d = -xb2d * costhe - yc2d * sinthe;
        const double yc3d = -xb2d * sinthe + yc2d * costhe;
        const double zc3d = zc1d;

        auto place = [&](double xd, double yd, double zd, int row) {
            x[row] = static_cast<StorageReal>(com.x + ex.x * xd + ey.x * yd
                                              + ez.x * zd);
            y[row] = static_cast<StorageReal>(com.y + ex.y * xd + ey.y * yd
                                              + ez.y * zd);
            z[row] = static_cast<StorageReal>(com.z + ex.z * xd + ey.z * yd
                                              + ez.z * zd);
        };
        place(xa3d, ya3d, za3d, o);
        place(xb3d, yb3d, zb3d, a);
        place(xc3d, yc3d, zc3d, b);
    }
}

void FixSettle::settle_velocities_(Frame& f) const {
    // Coupled 3-bond velocity constraints: iterate RATTLE-style projections.
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    const auto x = f.span(kAtom_x);
    const auto y = f.span(kAtom_y);
    const auto z = f.span(kAtom_z);
    const auto mass = f.span(kAtom_mass);
    for (int iter = 0; iter < 20; ++iter) {
        double max_rv = 0;
        for (const auto& w : waters_) {
            const int idxs[3][2] = {
                {w.io, w.ih1}, {w.io, w.ih2}, {w.ih1, w.ih2}};
            for (int c = 0; c < 3; ++c) {
                const int i = idxs[c][0], j = idxs[c][1];
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];
                f.box.min_image(dx, dy, dz);
                const double dvx = vx[i] - vx[j];
                const double dvy = vy[i] - vy[j];
                const double dvz = vz[i] - vz[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < 1e-30)
                    continue;
                const double rv = dx * dvx + dy * dvy + dz * dvz;
                max_rv = std::max(max_rv, std::fabs(rv));
                const double mi = mass[i], mj = mass[j];
                const double inv_mu = 1.0 / mi + 1.0 / mj;
                const double lam = -rv / (r2 * inv_mu);
                vx[i] += static_cast<StorageReal>(lam * dx / mi);
                vy[i] += static_cast<StorageReal>(lam * dy / mi);
                vz[i] += static_cast<StorageReal>(lam * dz / mi);
                vx[j] -= static_cast<StorageReal>(lam * dx / mj);
                vy[j] -= static_cast<StorageReal>(lam * dy / mj);
                vz[j] -= static_cast<StorageReal>(lam * dz / mj);
            }
        }
        if (max_rv < 1e-12)
            break;
    }
}

int FixSettle::pack_forward_comm(int n, const int* list, double* buf,
                                 int pbc_flag, const int* pbc) {
    if (!frame_ || n <= 0)
        return 0;
    int m = 0;
    if (comm_mode_ == CommMode::XSHAKE) {
        const auto x = frame_->span(kAtom_x);
        const auto y = frame_->span(kAtom_y);
        const auto z = frame_->span(kAtom_z);
        double dx = 0.0, dy = 0.0, dz = 0.0;
        if (pbc_flag != 0) {
            dx = static_cast<double>(pbc[0]) * frame_->box.length(0);
            dy = static_cast<double>(pbc[1]) * frame_->box.length(1);
            dz = static_cast<double>(pbc[2]) * frame_->box.length(2);
        }
        for (int i = 0; i < n; ++i) {
            const int j = list[i];
            buf[m++] = static_cast<double>(x[j]) + dx;
            buf[m++] = static_cast<double>(y[j]) + dy;
            buf[m++] = static_cast<double>(z[j]) + dz;
        }
        return m;
    }
    const auto vx = frame_->span(kAtom_vx);
    const auto vy = frame_->span(kAtom_vy);
    const auto vz = frame_->span(kAtom_vz);
    (void)pbc_flag;
    (void)pbc;
    for (int i = 0; i < n; ++i) {
        const int j = list[i];
        buf[m++] = static_cast<double>(vx[j]);
        buf[m++] = static_cast<double>(vy[j]);
        buf[m++] = static_cast<double>(vz[j]);
    }
    return m;
}

void FixSettle::unpack_forward_comm(int n, int first, const double* buf) {
    if (!frame_ || n <= 0)
        return;
    int m = 0;
    const int last = first + n;
    if (comm_mode_ == CommMode::XSHAKE) {
        auto x = frame_->span(kAtom_x);
        auto y = frame_->span(kAtom_y);
        auto z = frame_->span(kAtom_z);
        for (int i = first; i < last; ++i) {
            x[i] = static_cast<StorageReal>(buf[m++]);
            y[i] = static_cast<StorageReal>(buf[m++]);
            z[i] = static_cast<StorageReal>(buf[m++]);
        }
        return;
    }
    auto vx = frame_->span(kAtom_vx);
    auto vy = frame_->span(kAtom_vy);
    auto vz = frame_->span(kAtom_vz);
    for (int i = first; i < last; ++i) {
        vx[i] = static_cast<StorageReal>(buf[m++]);
        vy[i] = static_cast<StorageReal>(buf[m++]);
        vz[i] = static_cast<StorageReal>(buf[m++]);
    }
}

void FixSettle::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    frame_ = &f;
    if (ctx.at(ATV::md::Phase::initial_integrate)) {
        resolve_rows_(f);
        snapshot_(f);
        return;
    }
    if (ctx.at(ATV::md::Phase::post_integrate)) {
        comm_mode_ = CommMode::XSHAKE;
        if (cpu::Comm* comm = ctx.runtime.md.comm)
            comm->forward_comm(*this);
        settle_positions_(f);
        return;
    }
    if (ctx.at(ATV::md::Phase::final_integrate)) {
        comm_mode_ = CommMode::VP;
        if (cpu::Comm* comm = ctx.runtime.md.comm)
            comm->forward_comm(*this);
        settle_velocities_(f);
    }
}

} // namespace md
} // namespace cpu
} // namespace ATV
