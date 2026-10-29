/// @file tests/cpu/dynamics/test_sllod_flow.cpp
/// @brief LAMMPS ERATE / SLLOD / PBC / single-chain NHC locks.
///
/// These numbers are transcribed from `fix_deform`, `fix_nvt_sllod`,
/// `compute_temp_deform`, and `fix_nh` (mtchain = 1). They do not call
/// the integrator, so a login-node compile of the header can run them.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/units.h"

using ATV::md::kFlowX;
using ATV::md::kFlowXY;
using ATV::md::sllod_apply_bias;
using ATV::md::sllod_box_at;
using ATV::md::sllod_deform_bias;
using ATV::md::sllod_edge_shift;
using ATV::md::sllod_erate_rates;
using ATV::md::sllod_invert;
using ATV::md::sllod_multiply_shape_shape;
using ATV::md::sllod_nh_half;
using ATV::md::sllod_nve_x;
using ATV::md::sllod_pbc;
using ATV::md::sllod_refresh;
using ATV::md::SllodChain;
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
    const double kB = ATV::units::kB_ev_per_k;
    const double ndof = 3;
    const double T = 300;
    const double tau = 100;
    const double dt = 1;
    const double ke_target = ndof * kB * T;
    EXPECT_NEAR(ke_target, 0.077555999358, 1e-12);

    SllodChain hot;
    hot.mass = ke_target * tau * tau;
    const double scale = sllod_nh_half(hot, ke_target, ke_target, kB * T, dt);
    EXPECT_NEAR(scale, std::exp(-1.25e-5), 1e-15);
    EXPECT_NEAR(-2.0 * std::log(scale), 2.5e-5, 1e-15);

    SllodChain eq;
    eq.mass = hot.mass;
    const double eq_scale =
        sllod_nh_half(eq, 0.5 * ke_target, ke_target, kB * T, dt);
    EXPECT_NEAR(eq_scale, 1.0, 1e-15);
    EXPECT_NEAR(eq.eta_dot, 0.0, 1e-18);
}
