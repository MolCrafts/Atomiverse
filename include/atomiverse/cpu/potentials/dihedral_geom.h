/// @file cpu/potentials/dihedral_geom.h
/// @brief Shared 4-atom dihedral geometry + dV/dcos → force map.
///
/// Used by DihedralMultiHarmonic and DihedralCharmm (second caller justified
/// the extract). Not a Component — pure geometry helpers.
#pragma once

#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace dihedral_geom {

/// @brief Bond vectors (min-imaged), plane normals, and cos/sin of φ.
///
/// Normals are stored so `forces_from_dV_dcos` does not re-cross the bonds
/// (both multi/harmonic and CHARMM call fill then forces every tuple).
struct Geometry {
    double b1x = 0, b1y = 0, b1z = 0;
    double b2x = 0, b2y = 0, b2z = 0;
    double b3x = 0, b3y = 0, b3z = 0;
    double n1x = 0, n1y = 0, n1z = 0;
    double n2x = 0, n2y = 0, n2z = 0;
    double n1_inv = 0; ///< 1/|n1| cached for forces_from_dV_dcos
    double n2_inv = 0;
    double cos_phi = 0;
    double sin_phi = 0; ///< signed: sign((n1 × n2) · b2)
};

/// @brief Cross product a × b → (cx,cy,cz).
inline void cross(double ax, double ay, double az, double bx, double by,
                  double bz, double& cx, double& cy, double& cz) {
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

/// @brief Fill bond vectors and (cos φ, sin φ) for atoms r1–r4.
/// @param min_image Callable `(double&,double&,double&) const` applying PBC.
/// @return false if either plane normal vanishes (colinear bonds).
template <typename MinImage>
inline bool fill(double r1x, double r1y, double r1z, double r2x, double r2y,
                 double r2z, double r3x, double r3y, double r3z, double r4x,
                 double r4y, double r4z, MinImage&& min_image, Geometry& g) {
    g.b1x = r2x - r1x;
    g.b1y = r2y - r1y;
    g.b1z = r2z - r1z;
    g.b2x = r3x - r2x;
    g.b2y = r3y - r2y;
    g.b2z = r3z - r2z;
    g.b3x = r4x - r3x;
    g.b3y = r4y - r3y;
    g.b3z = r4z - r3z;
    min_image(g.b1x, g.b1y, g.b1z);
    min_image(g.b2x, g.b2y, g.b2z);
    min_image(g.b3x, g.b3y, g.b3z);

    cross(g.b1x, g.b1y, g.b1z, g.b2x, g.b2y, g.b2z, g.n1x, g.n1y, g.n1z);
    cross(g.b2x, g.b2y, g.b2z, g.b3x, g.b3y, g.b3z, g.n2x, g.n2y, g.n2z);

    const double n1_sq = g.n1x * g.n1x + g.n1y * g.n1y + g.n1z * g.n1z;
    const double n2_sq = g.n2x * g.n2x + g.n2y * g.n2y + g.n2z * g.n2z;
    if (n1_sq < 1e-20 || n2_sq < 1e-20)
        return false;

    g.n1_inv = 1.0 / std::sqrt(n1_sq);
    g.n2_inv = 1.0 / std::sqrt(n2_sq);
    const double b2_sq = g.b2x * g.b2x + g.b2y * g.b2y + g.b2z * g.b2z;
    if (b2_sq < 1e-20)
        return false;
    const double b2_inv = 1.0 / std::sqrt(b2_sq);

    double cos_phi =
        (g.n1x * g.n2x + g.n1y * g.n2y + g.n1z * g.n2z) * g.n1_inv * g.n2_inv;
    cos_phi = std::max(-1.0, std::min(1.0, cos_phi));

    // Signed sin via (n1 × n2) · û_b2 / (|n1||n2|).
    double mx, my, mz;
    cross(g.n1x, g.n1y, g.n1z, g.n2x, g.n2y, g.n2z, mx, my, mz);
    double sin_phi =
        (mx * g.b2x + my * g.b2y + mz * g.b2z) * g.n1_inv * g.n2_inv * b2_inv;
    sin_phi = std::max(-1.0, std::min(1.0, sin_phi));

    g.cos_phi = cos_phi;
    g.sin_phi = sin_phi;
    return true;
}

/// @brief Map dV/dcos(φ) to Cartesian forces on the four dihedral atoms.
///
/// Same chain rule as DihedralMultiHarmonic (see test_pe_fd_force.cpp).
/// Output forces are F = −∇V (ready to accumulate into frame force arrays).
inline void forces_from_dV_dcos(const Geometry& g, double dV_dcos, double& f1x,
                                double& f1y, double& f1z, double& f2x,
                                double& f2y, double& f2z, double& f3x,
                                double& f3y, double& f3z, double& f4x,
                                double& f4y, double& f4z) {
    // Reuse inverses from fill() — saves two sqrts per dihedral.
    const double n1_inv = g.n1_inv;
    const double n2_inv = g.n2_inv;
    const double cos_phi = g.cos_phi;

    const double n1h_x = g.n1x * n1_inv, n1h_y = g.n1y * n1_inv,
                 n1h_z = g.n1z * n1_inv;
    const double n2h_x = g.n2x * n2_inv, n2h_y = g.n2y * n2_inv,
                 n2h_z = g.n2z * n2_inv;

    const double fh1_x = (n2h_x - cos_phi * n1h_x) * n1_inv;
    const double fh1_y = (n2h_y - cos_phi * n1h_y) * n1_inv;
    const double fh1_z = (n2h_z - cos_phi * n1h_z) * n1_inv;

    const double fh2_x = (n1h_x - cos_phi * n2h_x) * n2_inv;
    const double fh2_y = (n1h_y - cos_phi * n2h_y) * n2_inv;
    const double fh2_z = (n1h_z - cos_phi * n2h_z) * n2_inv;

    const double db1_x = g.b2y * fh1_z - g.b2z * fh1_y;
    const double db1_y = g.b2z * fh1_x - g.b2x * fh1_z;
    const double db1_z = g.b2x * fh1_y - g.b2y * fh1_x;

    const double db3_x = fh2_y * g.b2z - fh2_z * g.b2y;
    const double db3_y = fh2_z * g.b2x - fh2_x * g.b2z;
    const double db3_z = fh2_x * g.b2y - fh2_y * g.b2x;

    const double db2_x =
        (fh1_y * g.b1z - fh1_z * g.b1y) + (g.b3y * fh2_z - g.b3z * fh2_y);
    const double db2_y =
        (fh1_z * g.b1x - fh1_x * g.b1z) + (g.b3z * fh2_x - g.b3x * fh2_z);
    const double db2_z =
        (fh1_x * g.b1y - fh1_y * g.b1x) + (g.b3x * fh2_y - g.b3y * fh2_x);

    f1x = dV_dcos * db1_x;
    f1y = dV_dcos * db1_y;
    f1z = dV_dcos * db1_z;

    f4x = -dV_dcos * db3_x;
    f4y = -dV_dcos * db3_y;
    f4z = -dV_dcos * db3_z;

    const double gx = dV_dcos * db2_x;
    const double gy = dV_dcos * db2_y;
    const double gz = dV_dcos * db2_z;

    f2x = -f1x + gx;
    f2y = -f1y + gy;
    f2z = -f1z + gz;

    f3x = -gx - f4x;
    f3y = -gy - f4y;
    f3z = -gz - f4z;
}

/// @brief Convert dE/dφ → dE/dcos(φ) with a stable limit at sin φ → 0.
///
/// d(cos)/dφ = −sin φ ⇒ dE/dcos = (dE/dφ) / (−sin φ).
inline double dV_dcos_from_dE_dphi(double dE_dphi, double sin_phi,
                                   double cos_phi, double d2E_dphi2) {
    constexpr double kSinEps = 1e-10;
    if (std::abs(sin_phi) > kSinEps)
        return -dE_dphi / sin_phi;
    // L'Hôpital: d/dφ [dE/dφ] / d/dφ [−sin φ] = (d²E/dφ²) / (−cos φ)
    // when sin φ = 0 and cos φ = ±1.
    if (std::abs(cos_phi) > kSinEps)
        return -d2E_dphi2 / cos_phi;
    return 0.0;
}

} // namespace dihedral_geom
} // namespace cpu
} // namespace ATV
