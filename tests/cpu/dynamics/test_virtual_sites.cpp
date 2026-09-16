/// @file tests/cpu/dynamics/test_virtual_sites.cpp
/// @brief Unit: multi-topology virtual sites (Weighted3 + OutOfPlane3).
///
/// No full MD/SCF e2e — geometry, force partition, tip4p α, tip5p coeffs only.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/virtual_sites.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kROH = 0.9572;
constexpr double kThetaDeg = 104.52;
constexpr double kQDist = 0.15;
constexpr double kROL = 0.70;
constexpr double kThetaLOLDeg = 109.47;

double theta_hoh() { return kThetaDeg * units::kDegreeInRad; }
double theta_lol() { return kThetaLOLDeg * units::kDegreeInRad; }

/// One rigid water: O,H1,H2 at equilibrium; optional 1–2 virtual rows at origin.
cpu::Frame water_parents(int n_extra = 0) {
    const double half = 0.5 * theta_hoh();
    const double s = std::sin(half), c = std::cos(half);
    const int n = 3 + n_extra;
    cpu::Frame f(n);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = kROH * s;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = kROH * c;
    f.span(kAtom_x)[2] = -kROH * s;
    f.span(kAtom_y)[2] = 0;
    f.span(kAtom_z)[2] = kROH * c;
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_type)[i] = (i == 0) ? 0 : (i <= 2 ? 1 : 2);
        f.span(kAtom_mass)[i] = (i == 0) ? 15.9994 : (i <= 2 ? 1.008 : 0.0);
        f.span(kAtom_fx)[i] = 0;
        f.span(kAtom_fy)[i] = 0;
        f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

} // namespace

TEST(VirtualSites, Tip4pAlpha) {
    const double a =
        cpu::md::VirtualSites::tip4p_alpha(kQDist, kROH, theta_hoh());
    const double ref = kQDist / (std::cos(0.5 * theta_hoh()) * kROH);
    EXPECT_NEAR(a, ref, 1e-15);
}

TEST(VirtualSites, Weighted3PlaceOmEqualsQdist) {
    cpu::Frame f = water_parents(1); // site tag 4
    const double alpha =
        cpu::md::VirtualSites::tip4p_alpha(kQDist, kROH, theta_hoh());
    cpu::md::VirtualSites vs;
    vs.add_weighted3(/*M=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, alpha);
    // Garbage initial M
    f.span(kAtom_x)[3] = 9.0;
    f.span(kAtom_y)[3] = 9.0;
    f.span(kAtom_z)[3] = 9.0;
    vs.place(f);
    const double mx = f.span(kAtom_x)[3];
    const double my = f.span(kAtom_y)[3];
    const double mz = f.span(kAtom_z)[3];
    EXPECT_NEAR(mx, 0.0, 1e-12);
    EXPECT_NEAR(my, 0.0, 1e-12);
    EXPECT_NEAR(mz, kQDist, 1e-10);
    const double om = std::sqrt(mx * mx + my * my + mz * mz);
    EXPECT_NEAR(om, kQDist, 1e-10);
}

TEST(VirtualSites, Weighted3ForcePartitionSum) {
    cpu::Frame f = water_parents(1);
    const double alpha =
        cpu::md::VirtualSites::tip4p_alpha(kQDist, kROH, theta_hoh());
    cpu::md::VirtualSites vs;
    vs.add_weighted3(4, 1, 2, 3, alpha);
    vs.place(f);
    // Put a force only on the site
    f.span(kAtom_fx)[3] = 1.2;
    f.span(kAtom_fy)[3] = -0.4;
    f.span(kAtom_fz)[3] = 0.7;
    const double Fs[3] = {1.2, -0.4, 0.7};
    vs.redistribute(f);
    // Site force zeroed
    EXPECT_NEAR(f.span(kAtom_fx)[3], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fy)[3], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[3], 0.0, 1e-15);
    // Sum of parent forces = original site force
    const double sx =
        f.span(kAtom_fx)[0] + f.span(kAtom_fx)[1] + f.span(kAtom_fx)[2];
    const double sy =
        f.span(kAtom_fy)[0] + f.span(kAtom_fy)[1] + f.span(kAtom_fy)[2];
    const double sz =
        f.span(kAtom_fz)[0] + f.span(kAtom_fz)[1] + f.span(kAtom_fz)[2];
    EXPECT_NEAR(sx, Fs[0], 1e-12);
    EXPECT_NEAR(sy, Fs[1], 1e-12);
    EXPECT_NEAR(sz, Fs[2], 1e-12);
    // Feenstra weights
    const double w0 = 1.0 - alpha;
    const double wh = 0.5 * alpha;
    EXPECT_NEAR(f.span(kAtom_fx)[0], w0 * Fs[0], 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], wh * Fs[0], 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[2], wh * Fs[0], 1e-12);
}

TEST(VirtualSites, Tip5pGeometryAndPartition) {
    cpu::Frame f = water_parents(2); // L1 tag4, L2 tag5
    double a1, b1, c1, a2, b2, c2;
    cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
        kROL, theta_lol(), kROH, theta_hoh(), +1, a1, b1, c1);
    cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
        kROL, theta_lol(), kROH, theta_hoh(), -1, a2, b2, c2);
    cpu::md::VirtualSites vs;
    vs.add_out_of_plane3(4, 1, 2, 3, a1, b1, c1);
    vs.add_out_of_plane3(5, 1, 2, 3, a2, b2, c2);
    vs.place(f);

    const double ox = f.span(kAtom_x)[0], oy = f.span(kAtom_y)[0],
                 oz = f.span(kAtom_z)[0];
    auto dist = [&](int i) {
        const double dx = f.span(kAtom_x)[i] - ox;
        const double dy = f.span(kAtom_y)[i] - oy;
        const double dz = f.span(kAtom_z)[i] - oz;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_NEAR(dist(3), kROL, 1e-8);
    EXPECT_NEAR(dist(4), kROL, 1e-8);

    // θ_LOL via cos law
    double v1x = f.span(kAtom_x)[3] - ox, v1y = f.span(kAtom_y)[3] - oy,
           v1z = f.span(kAtom_z)[3] - oz;
    double v2x = f.span(kAtom_x)[4] - ox, v2y = f.span(kAtom_y)[4] - oy,
           v2z = f.span(kAtom_z)[4] - oz;
    const double n1 = dist(3), n2 = dist(4);
    const double cosang = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2);
    EXPECT_NEAR(std::acos(cosang), theta_lol(), 1e-8);

    // Force partition identity on L1
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_fx)[i] = 0;
        f.span(kAtom_fy)[i] = 0;
        f.span(kAtom_fz)[i] = 0;
    }
    f.span(kAtom_fx)[3] = 0.3;
    f.span(kAtom_fy)[3] = -0.5;
    f.span(kAtom_fz)[3] = 0.8;
    // Only redistribute L1 (remove L2 from table for clean sum) — re-register
    cpu::md::VirtualSites vs1;
    vs1.add_out_of_plane3(4, 1, 2, 3, a1, b1, c1);
    vs1.place(f);
    f.span(kAtom_fx)[3] = 0.3;
    f.span(kAtom_fy)[3] = -0.5;
    f.span(kAtom_fz)[3] = 0.8;
    vs1.redistribute(f);
    EXPECT_NEAR(f.span(kAtom_fx)[3], 0.0, 1e-15);
    const double sx =
        f.span(kAtom_fx)[0] + f.span(kAtom_fx)[1] + f.span(kAtom_fx)[2];
    const double sy =
        f.span(kAtom_fy)[0] + f.span(kAtom_fy)[1] + f.span(kAtom_fy)[2];
    const double sz =
        f.span(kAtom_fz)[0] + f.span(kAtom_fz)[1] + f.span(kAtom_fz)[2];
    EXPECT_NEAR(sx, 0.3, 1e-12);
    EXPECT_NEAR(sy, -0.5, 1e-12);
    EXPECT_NEAR(sz, 0.8, 1e-12);
}
