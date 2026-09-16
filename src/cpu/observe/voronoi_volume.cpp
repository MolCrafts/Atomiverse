/// @file cpu/observe/voronoi_volume.cpp
/// @brief Periodic Voronoi cell volumes via convex bisector clipping.

#include "atomiverse/cpu/observe/voronoi_volume.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace cpu {

namespace {

using Vec3 = std::array<double, 3>;
/// Convex cell as independent face loops (no shared vertex indexing — faces
/// are clipped independently and the cap face is rebuilt per cut).
using Polygon = std::vector<Vec3>;
using Cell = std::vector<Polygon>;

constexpr double kOnPlaneEps = 1e-9; ///< [Å] point-on-plane classification.
constexpr double kMergeEps = 1e-7;   ///< [Å] cap-face duplicate merge.
constexpr double kCoincident = 1e-6; ///< [Å] degenerate site separation.

double dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

/// Axis-aligned cuboid [-h0,h0]x[-h1,h1]x[-h2,h2] centred at the origin.
Cell box_cuboid(double h0, double h1, double h2) {
    const Vec3 v000{-h0, -h1, -h2}, v100{h0, -h1, -h2}, v010{-h0, h1, -h2},
        v110{h0, h1, -h2}, v001{-h0, -h1, h2}, v101{h0, -h1, h2},
        v011{-h0, h1, h2}, v111{h0, h1, h2};
    return {
        {v000, v010, v110, v100}, // z = -h2
        {v001, v101, v111, v011}, // z = +h2
        {v000, v100, v101, v001}, // y = -h1
        {v010, v011, v111, v110}, // y = +h1
        {v000, v001, v011, v010}, // x = -h0
        {v100, v110, v111, v101}, // x = +h0
    };
}

double max_vertex_norm(const Cell& cell) {
    double r2 = 0.0;
    for (const Polygon& poly : cell)
        for (const Vec3& p : poly)
            r2 = std::max(r2, dot3(p, p));
    return std::sqrt(r2);
}

/// Clip `cell` by the half-space \(p\cdot n \le h\) (h > 0, origin inside).
/// @return True when the plane actually cut the cell.
bool clip_cell(Cell& cell, const Vec3& nrm, double h) {
    bool any_out = false;
    for (const Polygon& poly : cell) {
        for (const Vec3& p : poly) {
            if (dot3(p, nrm) - h > kOnPlaneEps) {
                any_out = true;
                break;
            }
        }
        if (any_out)
            break;
    }
    if (!any_out)
        return false;

    Cell out;
    out.reserve(cell.size() + 1);
    Polygon cap; // points on the cutting plane (unordered, with duplicates)
    for (const Polygon& poly : cell) {
        Polygon np;
        const std::size_t m = poly.size();
        for (std::size_t k = 0; k < m; ++k) {
            const Vec3& a = poly[k];
            const Vec3& b = poly[(k + 1) % m];
            const double da = dot3(a, nrm) - h;
            const double db = dot3(b, nrm) - h;
            const bool a_in = da <= kOnPlaneEps;
            const bool b_in = db <= kOnPlaneEps;
            if (a_in) {
                np.push_back(a);
                if (std::abs(da) <= kOnPlaneEps)
                    cap.push_back(a);
                if (!b_in && da < -kOnPlaneEps) {
                    const double t = da / (da - db);
                    const Vec3 p{a[0] + t * (b[0] - a[0]),
                                 a[1] + t * (b[1] - a[1]),
                                 a[2] + t * (b[2] - a[2])};
                    np.push_back(p);
                    cap.push_back(p);
                }
            } else if (b_in && db < -kOnPlaneEps) {
                const double t = da / (da - db);
                const Vec3 p{a[0] + t * (b[0] - a[0]),
                             a[1] + t * (b[1] - a[1]),
                             a[2] + t * (b[2] - a[2])};
                np.push_back(p);
                cap.push_back(p);
            }
        }
        if (np.size() >= 3)
            out.push_back(std::move(np));
    }

    // Deduplicate cap points and order them around the plane normal.
    Polygon uniq;
    for (const Vec3& p : cap) {
        bool dup = false;
        for (const Vec3& q : uniq) {
            const Vec3 d{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
            if (dot3(d, d) < kMergeEps * kMergeEps) {
                dup = true;
                break;
            }
        }
        if (!dup)
            uniq.push_back(p);
    }
    if (uniq.size() >= 3) {
        Vec3 c{0.0, 0.0, 0.0};
        for (const Vec3& p : uniq)
            for (int d = 0; d < 3; ++d)
                c[d] += p[d] / static_cast<double>(uniq.size());
        // In-plane basis (u, v) orthogonal to the cut normal.
        Vec3 ref{1.0, 0.0, 0.0};
        if (std::abs(nrm[0]) > 0.9)
            ref = {0.0, 1.0, 0.0};
        Vec3 u = cross3(nrm, ref);
        const double un = std::sqrt(dot3(u, u));
        for (int d = 0; d < 3; ++d)
            u[d] /= un;
        const Vec3 v = cross3(nrm, u);
        std::sort(uniq.begin(), uniq.end(), [&](const Vec3& a, const Vec3& b) {
            const Vec3 pa{a[0] - c[0], a[1] - c[1], a[2] - c[2]};
            const Vec3 pb{b[0] - c[0], b[1] - c[1], b[2] - c[2]};
            return std::atan2(dot3(pa, v), dot3(pa, u))
                   < std::atan2(dot3(pb, v), dot3(pb, u));
        });
        out.push_back(std::move(uniq));
    }
    cell = std::move(out);
    return true;
}

/// Cell volume by per-face pyramid decomposition about the origin (which is
/// strictly interior: every bisector plane sits at distance > 0).
double cell_volume(const Cell& cell) {
    double vol = 0.0;
    for (const Polygon& poly : cell) {
        double s = 0.0;
        for (std::size_t k = 1; k + 1 < poly.size(); ++k)
            s += dot3(poly[0], cross3(poly[k], poly[k + 1]));
        vol += std::abs(s) / 6.0;
    }
    return vol;
}

} // namespace

std::vector<double> VoronoiVolume::volumes(const Frame& frame) const {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("VoronoiVolume: empty frame");
    const Box& box = frame.box;
    if (!box.is_orthorhombic())
        throw std::invalid_argument(
            "VoronoiVolume: orthorhombic boxes only (triclinic unsupported)");
    const double L[3] = {box.length(0), box.length(1), box.length(2)};
    for (int d = 0; d < 3; ++d)
        if (!(L[d] > 0.0))
            throw std::invalid_argument(
                "VoronoiVolume: box edges must be > 0 [Å]");

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);

    const double diag =
        std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    // Neighbor images with |dr| >= diag cannot cut the initial cuboid
    // (circumradius diag/2). Base displacements are min-imaged, so
    // ceil(diag/L)+1 image shells per axis cover every reachable image.
    int nmax[3];
    for (int d = 0; d < 3; ++d)
        nmax[d] = static_cast<int>(std::ceil(diag / L[d])) + 1;

    struct Plane {
        double dist; ///< [Å] distance from the site to the bisector.
        Vec3 nrm;    ///< Unit normal (towards the neighbor image).
    };

    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    std::vector<Plane> planes;
    for (int i = 0; i < n; ++i) {
        planes.clear();
        for (int j = 0; j < n; ++j) {
            if (j == i)
                continue; // self-images are the initial cuboid's own faces
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            box.min_image(dx, dy, dz);
            for (int n0 = -nmax[0]; n0 <= nmax[0]; ++n0) {
                for (int n1 = -nmax[1]; n1 <= nmax[1]; ++n1) {
                    for (int n2 = -nmax[2]; n2 <= nmax[2]; ++n2) {
                        const Vec3 dr{dx + n0 * L[0], dy + n1 * L[1],
                                      dz + n2 * L[2]};
                        const double r = std::sqrt(dot3(dr, dr));
                        if (r >= diag)
                            continue;
                        if (r < kCoincident)
                            throw std::runtime_error(
                                "VoronoiVolume: coincident sites (rows "
                                + std::to_string(i) + ", " + std::to_string(j)
                                + ")");
                        planes.push_back(
                            {0.5 * r, {dr[0] / r, dr[1] / r, dr[2] / r}});
                    }
                }
            }
        }
        std::sort(planes.begin(), planes.end(),
                  [](const Plane& a, const Plane& b) {
                      return a.dist < b.dist;
                  });

        Cell cell = box_cuboid(0.5 * L[0], 0.5 * L[1], 0.5 * L[2]);
        double rmax = max_vertex_norm(cell);
        for (const Plane& p : planes) {
            if (p.dist >= rmax)
                break; // sorted ascending; no later plane can cut
            if (clip_cell(cell, p.nrm, p.dist))
                rmax = max_vertex_norm(cell);
        }
        out[static_cast<std::size_t>(i)] = cell_volume(cell);
    }
    return out;
}

} // namespace cpu
} // namespace ATV
