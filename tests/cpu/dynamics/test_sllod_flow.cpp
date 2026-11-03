/// @file tests/cpu/dynamics/test_sllod_flow.cpp
/// @brief LAMMPS ERATE / SLLOD / PBC / NHC / box-flip / skin-trigger locks.
///
/// These numbers are transcribed from `fix_deform`, `fix_nvt_sllod`,
/// `compute_temp_deform`, `fix_nh`, `domain` and `neighbor` (LAMMPS
/// e8c4c278a8). They do not call the integrator, so a login-node compile
/// of the headers can run them.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/neighbor_check.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/units.h"

#include "cpu/dynamics/lammps_nhc_reference.h"

using ATV::md::kFlowX;
using ATV::md::kFlowXY;
using ATV::md::kFlowYZ;
using ATV::md::skin_trigger_sq;
using ATV::md::sllod_advance_box;
using ATV::md::sllod_apply_bias;
using ATV::md::sllod_apply_flip;
using ATV::md::sllod_box_at;
using ATV::md::sllod_chain_masses;
using ATV::md::sllod_deform_bias;
using ATV::md::sllod_edge_shift;
using ATV::md::sllod_erate_rates;
using ATV::md::sllod_flip_pending;
using ATV::md::sllod_image_flip;
using ATV::md::sllod_invert;
using ATV::md::sllod_multiply_shape_shape;
using ATV::md::sllod_nhc_half;
using ATV::md::sllod_nve_x;
using ATV::md::sllod_pbc;
using ATV::md::sllod_refresh;
using ATV::md::SllodFlow;

TEST(SllodFlow, MultiplyShapeShape) {
    const double one[6] = {0.1, 0.2, 0.3, 0.01, 0.02, 0.04};
    const double two[6] = {0.5, 0.25, 0.2, -0.1, 0.05, -0.2};
    double ans[6]{};
    sllod_multiply_shape_shape(one, two, ans);
    const double want[6] = {0.05, 0.05, 0.06, -0.018, 0.005, -0.01};
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(ans[i], want[i], 1e-15) << i;
}

TEST(SllodFlow, UniaxialRateBoxAndBias) {
    double h0[6] = {10, 8, 6, 0, 0, 0};
    double h_rate[6]{};
    double h_ratelo[6]{};
    sllod_erate_rates(h0, 0.01, kFlowX, h_rate, h_ratelo);
    EXPECT_NEAR(h_rate[0], 0.1, 0);
    EXPECT_NEAR(h_ratelo[0], -0.05, 0);
    for (int i = 1; i < 6; ++i) {
        EXPECT_EQ(h_rate[i], 0.0);
        EXPECT_EQ(h_ratelo[i], 0.0);
    }

    double h[6]{};
    sllod_box_at(h0, 0.01, 0.1, kFlowX, 1, h);
    EXPECT_NEAR(h[0], 10.01, 1e-12);
    EXPECT_NEAR(sllod_edge_shift(h0[0], h[0], kFlowX), 0.005, 1e-15);

    SllodFlow flow{};
    for (int i = 0; i < 6; ++i)
        flow.h[i] = h0[i];
    for (int i = 0; i < 6; ++i) {
        flow.h_rate[i] = h_rate[i];
        flow.h_ratelo[i] = h_ratelo[i];
    }
    ASSERT_TRUE(sllod_refresh(flow));
    EXPECT_NEAR(flow.grad_u[0], 0.01, 1e-15);

    double vx = 0, vy = 0, vz = 0;
    sllod_apply_bias(vx, vy, vz, 10, 0, 0, flow.h, flow.grad_u, 0);
    EXPECT_NEAR(vx, 0.05, 1e-15);
    double bx, by, bz;
    sllod_deform_bias(10, 0, 0, flow.h_inv, flow.h_rate, flow.h_ratelo, bx, by,
                      bz);
    EXPECT_NEAR(bx, 0.05, 1e-15);
    EXPECT_NEAR(by, 0.0, 1e-15);
    EXPECT_NEAR(bz, 0.0, 1e-15);
}

TEST(SllodFlow, ShearXyRate) {
    double h0[6] = {10, 8, 6, 0, 0, 0};
    double h_rate[6]{};
    double h_ratelo[6]{};
    sllod_erate_rates(h0, 0.02, kFlowXY, h_rate, h_ratelo);
    EXPECT_NEAR(h_rate[5], 0.16, 1e-15);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(h_ratelo[i], 0.0);

    SllodFlow flow{};
    for (int i = 0; i < 6; ++i)
        flow.h[i] = h0[i];
    for (int i = 0; i < 6; ++i)
        flow.h_rate[i] = h_rate[i];
    ASSERT_TRUE(sllod_refresh(flow));
    EXPECT_NEAR(flow.grad_u[5], 0.02, 1e-15);

    double vx = 0, vy = 0, vz = 0;
    sllod_apply_bias(vx, vy, vz, 0, 0, 0, flow.h, flow.grad_u, 0);
    EXPECT_NEAR(vx, 0.0, 1e-15);
    vx = vy = vz = 0;
    sllod_apply_bias(vx, vy, vz, 0, 8, 0, flow.h, flow.grad_u, 0);
    EXPECT_NEAR(vx, 0.16, 1e-15);

    double bx, by, bz;
    sllod_deform_bias(0, 8, 0, flow.h_inv, flow.h_rate, flow.h_ratelo, bx, by,
                      bz);
    EXPECT_NEAR(bx, 0.16, 1e-15);

    double h[6]{};
    sllod_box_at(h0, 0.02, 0.1, kFlowXY, 1, h);
    EXPECT_NEAR(h[5], 0.016, 1e-15);
    EXPECT_EQ(sllod_edge_shift(h0[5], h[5], kFlowXY), 0.0);
}

TEST(SllodFlow, TiltedInverse) {
    const double h[6] = {10, 8, 6, 0.3, -0.2, 1.5};
    double h_inv[6]{};
    ASSERT_TRUE(sllod_invert(h, h_inv));
    EXPECT_NEAR(h_inv[0], 0.1, 1e-15);
    EXPECT_NEAR(h_inv[1], 0.125, 1e-15);
    EXPECT_NEAR(h_inv[2], 1.0 / 6.0, 1e-15);
    EXPECT_NEAR(h_inv[3], -0.3 / 48.0, 1e-15);
    EXPECT_NEAR(h_inv[4], 2.05 / 480.0, 1e-15);
    EXPECT_NEAR(h_inv[5], -1.5 / 80.0, 1e-15);
    const double bad[6] = {0, 8, 6, 0, 0, 0};
    EXPECT_FALSE(sllod_invert(bad, h_inv));
}

TEST(SllodFlow, CenterAtomStays) {
    SllodFlow flow{};
    flow.h[0] = flow.h[1] = flow.h[2] = 10;
    sllod_erate_rates(flow.h, 0.01, kFlowX, flow.h_rate, flow.h_ratelo);
    ASSERT_TRUE(sllod_refresh(flow));
    double x = 5, y = 1, z = 2, vx = 0, vy = 0, vz = 0;
    sllod_nve_x(x, y, z, vx, vy, vz, flow.h, flow.grad_u, 0.1);
    EXPECT_NEAR(x, 5.0, 1e-12);
    EXPECT_NEAR(y, 1.0, 1e-12);
    EXPECT_NEAR(z, 2.0, 1e-12);
    EXPECT_NEAR(vx, 0.0, 1e-15);
}

TEST(SllodFlow, UniaxialNveXOneStep) {
    SllodFlow flow{};
    flow.h[0] = flow.h[1] = flow.h[2] = 10;
    sllod_erate_rates(flow.h, 0.01, kFlowX, flow.h_rate, flow.h_ratelo);
    ASSERT_TRUE(sllod_refresh(flow));
    double x = 7, y = 1, z = 2, vx = 0.2, vy = 0.1, vz = -0.05;
    sllod_nve_x(x, y, z, vx, vy, vz, flow.h, flow.grad_u, 0.1);
    const double xfac = std::exp(0.0005);
    const double vfac = std::exp(-0.0005);
    const double x_after_drift = 5.0 + 2.0 * xfac + 0.1 * 0.2 * vfac;
    EXPECT_NEAR(x, 5.0 + (x_after_drift - 5.0) * xfac, 1e-12);
    EXPECT_NEAR(y, 1.01, 1e-12);
    EXPECT_NEAR(z, 1.995, 1e-12);
    EXPECT_NEAR(vx, 0.2 * vfac * vfac, 1e-12);
    EXPECT_NEAR(vy, 0.1, 1e-12);
    EXPECT_NEAR(vz, -0.05, 1e-12);
}

TEST(SllodFlow, PbcVremap) {
    const double h[6] = {10, 10, 10, 0, 0, 0};
    double h_inv[6]{};
    ASSERT_TRUE(sllod_invert(h, h_inv));
    const double h_rate[6] = {0.1, 0, 0, 0, 0, 0};
    double x = -0.1, y = 1, z = 2, vx = 0.3, vy = 0, vz = 0;
    sllod_pbc(x, y, z, vx, vy, vz, h, h_inv, h_rate);
    EXPECT_NEAR(x, 9.9, 1e-12);
    EXPECT_NEAR(vx, 0.4, 1e-12);

    x = -10.1;
    vx = 0;
    sllod_pbc(x, y, z, vx, vy, vz, h, h_inv, h_rate);
    EXPECT_NEAR(x, 9.9, 1e-12);
    EXPECT_NEAR(vx, 0.2, 1e-12);

    const double tilted[6] = {10, 10, 10, 0, 0, 2};
    double inv[6]{};
    ASSERT_TRUE(sllod_invert(tilted, inv));
    const double rate[6] = {0.1, 0, 0, 0, 0, 0.16};
    x = -0.1;
    y = 0;
    z = 0;
    vx = 0;
    vy = 0;
    vz = 0;
    sllod_pbc(x, y, z, vx, vy, vz, tilted, inv, rate);
    EXPECT_NEAR(x, 9.9, 1e-12);
    EXPECT_NEAR(vx, 0.1, 1e-12);
}

TEST(SllodFlow, PbcReportsTheLatticeTranslationForImageFlags) {
    // Tilted xy and yz cell; the wrapped row plus H·shift is the input row.
    const double h[6] = {10, 9, 8, 1.5, -1.0, 2.5};
    double h_inv[6]{};
    ASSERT_TRUE(sllod_invert(h, h_inv));
    const double h_rate[6] = {0, 0, 0, 0, 0, 0};
    const double x0 = 23.0, y0 = -2.0, z0 = 17.5;
    double x = x0, y = y0, z = z0, vx = 0, vy = 0, vz = 0;
    int shift[3] = {9, 9, 9};
    sllod_pbc(x, y, z, vx, vy, vz, h, h_inv, h_rate, shift);
    EXPECT_EQ(shift[2], 2);
    // Voigt h: lx ly lz yz xz xy; lattice a = (lx,0,0), b = (xy,ly,0),
    // c = (xz,yz,lz).
    EXPECT_NEAR(x + h[0] * shift[0] + h[5] * shift[1] + h[4] * shift[2], x0,
                1e-12);
    EXPECT_NEAR(y + h[1] * shift[1] + h[3] * shift[2], y0, 1e-12);
    EXPECT_NEAR(z + h[2] * shift[2], z0, 1e-12);
    // In the cell: fractional coordinates in [0, 1).
    const double lam2 = h_inv[2] * z;
    const double lam1 = h_inv[1] * y + h_inv[3] * z;
    const double lam0 = h_inv[0] * x + h_inv[5] * y + h_inv[4] * z;
    for (double lam : {lam0, lam1, lam2}) {
        EXPECT_GE(lam, 0.0);
        EXPECT_LT(lam, 1.0);
    }
}

TEST(SllodFlow, BiasRoundTrip) {
    SllodFlow flow{};
    flow.h[0] = 12;
    flow.h[1] = 8;
    flow.h[2] = 6;
    const double h0[6] = {10, 8, 6, 0, 0, 0};
    sllod_erate_rates(h0, 0.01, kFlowX, flow.h_rate, flow.h_ratelo);
    ASSERT_TRUE(sllod_refresh(flow));
    const double x = 3.5, y = 1.25, z = 2.5;
    double vx = 0.4, vy = -0.2, vz = 0.1;
    const double vx0 = vx, vy0 = vy, vz0 = vz;
    double bx, by, bz;
    sllod_deform_bias(x, y, z, flow.h_inv, flow.h_rate, flow.h_ratelo, bx, by,
                      bz);
    vx -= bx;
    vy -= by;
    vz -= bz;
    sllod_apply_bias(vx, vy, vz, x, y, z, flow.h, flow.grad_u, 0);
    EXPECT_NEAR(vx, vx0, 1e-12);
    EXPECT_NEAR(vy, vy0, 1e-12);
    EXPECT_NEAR(vz, vz0, 1e-12);
}

TEST(SllodFlow, NoseHooverHalf) {
    const double kB = ATV::units::md_boltz;
    const double ndof = 3;
    // lj units (md_boltz = 1): T is the metal k_B·300 K in eV, so the pins
    // below keep their pre-lj magnitudes.
    const double T = 0.025851999786;
    const double tau = 100;
    const double dt = 1;
    const double ke_target = ndof * kB * T;
    EXPECT_NEAR(ke_target, 0.077555999358, 1e-12);

    double eta = 0, eta_dot = 0, mass = 0;
    sllod_chain_masses(&mass, 1, ndof, kB * T, tau);
    EXPECT_NEAR(mass, ke_target * tau * tau, 1e-15);
    double ke = ke_target;
    const double scale =
        sllod_nhc_half(&eta, &eta_dot, &mass, 1, ke, ke_target, kB * T, dt);
    EXPECT_NEAR(scale, std::exp(-1.25e-5), 1e-15);
    EXPECT_NEAR(-2.0 * std::log(scale), 2.5e-5, 1e-15);
    // LAMMPS carries t_current *= factor_eta^2 into the next half-step.
    EXPECT_NEAR(ke, ke_target * scale * scale, 1e-16);

    double eq_eta = 0, eq_dot = 0;
    double eq_ke = 0.5 * ke_target;
    const double eq_scale = sllod_nhc_half(&eq_eta, &eq_dot, &mass, 1, eq_ke,
                                           ke_target, kB * T, dt);
    EXPECT_NEAR(eq_scale, 1.0, 1e-15);
    EXPECT_NEAR(eq_dot, 0.0, 1e-18);
}

namespace {

SllodFlow make_flow(const double* h0, double rate, double dt, int dir) {
    SllodFlow flow{};
    for (int i = 0; i < 6; ++i)
        flow.h0[i] = flow.h[i] = h0[i];
    flow.rate = rate;
    flow.dt = dt;
    flow.dir = dir;
    sllod_erate_rates(h0, rate, dir, flow.h_rate, flow.h_ratelo);
    EXPECT_TRUE(sllod_refresh(flow));
    return flow;
}

} // namespace

TEST(SllodFlow, ChainMatchesLammpsNhcTempIntegrate) {
    // Chains of several lengths over half-steps whose measured KE jumps
    // between calls (as final_integrate's fresh measurement does).
    const double kT = ATV::units::md_boltz * 300.0;
    const double ndof = 45, tau = 20, dt = 2;
    for (int m : {1, 2, 3, 5}) {
        atv_test::LammpsNhc ref(m, ndof, kT, tau, dt);
        std::vector<double> eta(m, 0.0), dot(m, 0.0), mass(m, 0.0);
        sllod_chain_masses(mass.data(), m, ndof, kT, tau);
        for (int k = 0; k < m; ++k)
            EXPECT_DOUBLE_EQ(mass[k], ref.eta_mass[k]) << "m=" << m;
        double ke_ref = 1.7 * 0.5 * ndof * kT, ke = ke_ref;
        for (int call = 0; call < 40; ++call) {
            if (call % 2 == 1) {
                ke_ref *= 1.0 + 0.01 * std::sin(0.7 * call);
                ke = ke_ref;
            }
            const double want = ref.integrate(ke_ref);
            const double got = sllod_nhc_half(
                eta.data(), dot.data(), mass.data(), m, ke, ndof * kT, kT, dt);
            ASSERT_NEAR(got, want, 1e-14) << "m=" << m << " call " << call;
            ASSERT_NEAR(ke, ke_ref, 1e-14 * ke_ref);
            for (int k = 0; k < m; ++k) {
                ASSERT_NEAR(eta[k], ref.eta[k], 1e-13 * (1 + std::abs(eta[k])))
                    << "m=" << m << " link " << k << " call " << call;
                ASSERT_NEAR(dot[k], ref.eta_dot[k],
                            1e-13 * (1e-3 + std::abs(dot[k])))
                    << "m=" << m << " link " << k << " call " << call;
            }
        }
        if (m > 1)
            EXPECT_NE(dot[1], 0.0) << "outer links must move";
    }
}

TEST(SllodFlow, XyShearFlipsPastHalfAndContinuesTheLinearPath) {
    // xy grows by rate * ly * dt = 0.09 per step from 4.8 (lx = 10).
    const double h0[6] = {10, 9, 8, 0, 0, 4.8};
    SllodFlow flow = make_flow(h0, 0.01, 1.0, kFlowXY);
    int flipped_at = -1;
    for (long long step = 1; step <= 6; ++step) {
        sllod_advance_box(flow, step);
        ASSERT_TRUE(sllod_refresh(flow));
        EXPECT_NEAR(flow.h[kFlowXY], 4.8 + 0.09 * step, 1e-12);
        if (sllod_flip_pending(flow)) {
            flipped_at = static_cast<int>(step);
            break;
        }
    }
    // 4.8 + 3 * 0.09 = 5.07 > 5: decided at the third end_of_step.
    ASSERT_EQ(flipped_at, 3);
    EXPECT_EQ(flow.flip[0], -1);
    EXPECT_EQ(flow.flip[1], 0);
    EXPECT_EQ(flow.flip[2], 0);
    EXPECT_NEAR(flow.h_flip[kFlowXY], 5.07 - 10.0, 1e-12);
    // The cell stays unflipped until pre_exchange.
    EXPECT_NEAR(flow.h[kFlowXY], 5.07, 1e-12);
    const double grad_before = flow.grad_u[kFlowXY];
    sllod_apply_flip(flow);
    ASSERT_TRUE(sllod_refresh(flow));
    flow.flip[0] = 0;
    EXPECT_NEAR(flow.h[kFlowXY], -4.93, 1e-12);
    EXPECT_DOUBLE_EQ(flow.grad_u[kFlowXY], grad_before);
    // update_domain: the next target is the linear tilt shifted by -lx.
    sllod_advance_box(flow, 4);
    EXPECT_NEAR(flow.h[kFlowXY], 4.8 + 0.36 - 10.0, 1e-12);
    EXPECT_FALSE(sllod_flip_pending(flow));
    // Negative shear flips the other way.
    const double hn[6] = {10, 9, 8, 0, 0, -4.95};
    SllodFlow neg = make_flow(hn, -0.01, 1.0, kFlowXY);
    sllod_advance_box(neg, 1);
    EXPECT_EQ(neg.flip[0], 1);
    EXPECT_NEAR(neg.h_flip[kFlowXY], -5.04 + 10.0, 1e-12);
}

TEST(SllodFlow, YzFlipMovesXzByXy) {
    // fix_deform.cpp:1206: a yz flip shifts c by b, so xz moves by xy.
    const double h0[6] = {10, 9, 8, 4.45, 0.3, 1.0};
    SllodFlow flow = make_flow(h0, 0.01, 1.0, kFlowYZ);
    sllod_advance_box(flow, 1); // yz = 4.45 + 0.08 = 4.53 > 4.5
    ASSERT_TRUE(sllod_flip_pending(flow));
    EXPECT_EQ(flow.flip[2], -1);
    EXPECT_EQ(flow.flip[1], 0);
    EXPECT_EQ(flow.flip[0], 0);
    EXPECT_NEAR(flow.h_flip[kFlowYZ], 4.53 - 9.0, 1e-12);
    EXPECT_NEAR(flow.h_flip[4], 0.3 - 1.0, 1e-12);
    EXPECT_EQ(flow.h_flip[kFlowXY], 1.0);
    sllod_apply_flip(flow);
    ASSERT_TRUE(sllod_refresh(flow));
    flow.flip[2] = 0;
    // Later steps keep the flipped xz (no second shift) and continue yz.
    sllod_advance_box(flow, 2);
    EXPECT_NEAR(flow.h[kFlowYZ], 4.45 + 0.16 - 9.0, 1e-12);
    EXPECT_NEAR(flow.h[4], -0.7, 1e-12);
}

TEST(SllodFlow, ImageFlipKeepsUnwrappedPositions) {
    // Domain::image_flip + remap: r + H image is invariant across the flip.
    const double before[6] = {10, 9, 8, 4.6, 0.4, 5.2};
    SllodFlow flow{};
    for (int i = 0; i < 6; ++i)
        flow.h[i] = flow.h_flip[i] = before[i];
    ASSERT_TRUE(sllod_refresh(flow));
    flow.h_flip[kFlowYZ] = 4.6 - 9.0;
    flow.h_flip[4] = 0.4 - 5.2;      // yz flip: xz -= xy
    flow.h_flip[kFlowXY] = 5.2 - 10; // xy flip
    flow.flip[0] = -1;
    flow.flip[1] = 0;
    flow.flip[2] = -1;
    SllodFlow after = flow;
    sllod_apply_flip(after);
    ASSERT_TRUE(sllod_refresh(after));
    const double rows[4][3] = {
        {1.0, 2.0, 3.0}, {9.5, 8.7, 7.9}, {-0.4, 4.0, 0.2}, {12.0, 3.0, 6.5}};
    const int images[4][3] = {{0, 0, 0}, {1, -2, 3}, {-1, 1, -1}, {2, 0, -3}};
    auto unwrap = [](const double* h, const double* r, const int* n,
                     double* u) {
        u[0] = r[0] + h[0] * n[0] + h[5] * n[1] + h[4] * n[2];
        u[1] = r[1] + h[1] * n[1] + h[3] * n[2];
        u[2] = r[2] + h[2] * n[2];
    };
    int moved = 0;
    for (int a = 0; a < 4; ++a) {
        double u0[3], u1[3];
        unwrap(flow.h, rows[a], images[a], u0);
        int n[3] = {images[a][0], images[a][1], images[a][2]};
        sllod_image_flip(n[0], n[1], n[2], flow.flip);
        double x = rows[a][0], y = rows[a][1], z = rows[a][2];
        double vx = 0, vy = 0, vz = 0;
        int shift[3];
        sllod_pbc(x, y, z, vx, vy, vz, after.h, after.h_inv, after.h_rate,
                  shift);
        const double r[3] = {x, y, z};
        for (int d = 0; d < 3; ++d) {
            n[d] += shift[d];
            moved += n[d] != images[a][d];
        }
        unwrap(after.h, r, n, u1);
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(u1[d], u0[d], 1e-12) << "row " << a << " axis " << d;
    }
    EXPECT_GT(moved, 0);
}

TEST(SllodFlow, SkinTriggerShrinksWithBoxCorners) {
    const double skin = 1.0;
    const double ortho[9] = {10, 0, 0, 0, 9, 0, 0, 0, 8};
    EXPECT_DOUBLE_EQ(skin_trigger_sq(ortho, ortho, skin), 0.25);
    // Orthogonal: lo corner fixed, hi corner moves by |(0.3, 0, 0.4)| = 0.5.
    const double grown[9] = {10.3, 0, 0, 0, 9, 0, 0, 0, 8.4};
    EXPECT_NEAR(skin_trigger_sq(ortho, grown, skin), 0.25 * 0.25, 1e-15);
    // A tilt makes the cell triclinic: the four corners with lambda1 = 1
    // move by the xy change 0.2; LAMMPS keeps the first maximum and the
    // next value that is not above it, so delta1 = delta2 = 0.2.
    const double tilted[9] = {10, 0.2, 0, 0, 9, 0, 0, 0, 8};
    const double delta = 0.5 * (skin - (0.2 + 0.2));
    EXPECT_NEAR(skin_trigger_sq(ortho, tilted, skin), delta * delta, 1e-15);
    // Box motion that consumes the skin rebuilds outright.
    const double sheared[9] = {10, 0.6, 0, 0, 9, 0, 0, 0, 8};
    EXPECT_LT(skin_trigger_sq(ortho, sheared, skin), 0.0);
    // A flip moves corners by a box length.
    const double before[9] = {10, 5.1, 0, 0, 9, 0, 0, 0, 8};
    const double flipped[9] = {10, -4.9, 0, 0, 9, 0, 0, 0, 8};
    EXPECT_LT(skin_trigger_sq(before, flipped, skin), 0.0);
    // Zero skin with an unchanged cell: any motion triggers.
    EXPECT_EQ(skin_trigger_sq(ortho, ortho, 0.0), 0.0);
}
