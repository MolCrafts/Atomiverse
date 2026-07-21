/// @file tests/cpu/test_box.cpp
/// @brief Coverage for `cpu::Box`: API, ortho min-image parity, triclinic
///        roundtrip identity, wrap-axis parity, mutation cache coherence,
///        xsimd batch parity.
///
/// Reference forms are written inline in the tests, independent of
/// `Box::*` implementations, so the assertions catch implementation
/// drift.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/box_simd.h"

#include <xsimd/xsimd.hpp>

using ATV::Mat3;
using ATV::cpu::Box;

namespace {

/// @brief Reference: branchy single-period min-image (matches the form
///        copied across pre-refactor bonded/ and lj.cpp scalar tail).
///        Valid for `|d_*| <= 1.5 * L_*`.
void min_image_reference(double& dx, double& dy, double& dz, const Mat3& cell) {
    const double lx = cell(0, 0), ly = cell(1, 1), lz = cell(2, 2);
    if (dx > 0.5 * lx)
        dx -= lx;
    if (dx < -0.5 * lx)
        dx += lx;
    if (dy > 0.5 * ly)
        dy -= ly;
    if (dy < -0.5 * ly)
        dy += ly;
    if (dz > 0.5 * lz)
        dz -= lz;
    if (dz < -0.5 * lz)
        dz += lz;
}

/// @brief Reference: floor-based axis wrap (matches the pre-refactor
///        `wrap_to_box` static-inline copies).
double wrap_axis_reference(double q, double L) {
    if (L <= 0.0)
        return q;
    double w = q - std::floor(q / L) * L;
    if (w < 0.0)
        w += L;
    if (w >= L)
        w -= L;
    return w;
}

/// @brief Reference: triclinic min-image via inv_cell roundtrip.
///        Independent of `Box::min_image`'s implementation.
void min_image_triclinic_reference(double& dx, double& dy, double& dz,
                                   const Mat3& cell, const Mat3& inv) {
    const double fx = inv(0, 0) * dx + inv(0, 1) * dy + inv(0, 2) * dz;
    const double fy = inv(1, 0) * dx + inv(1, 1) * dy + inv(1, 2) * dz;
    const double fz = inv(2, 0) * dx + inv(2, 1) * dy + inv(2, 2) * dz;
    const double rfx = fx - std::nearbyint(fx);
    const double rfy = fy - std::nearbyint(fy);
    const double rfz = fz - std::nearbyint(fz);
    dx = cell(0, 0) * rfx + cell(0, 1) * rfy + cell(0, 2) * rfz;
    dy = cell(1, 0) * rfx + cell(1, 1) * rfy + cell(1, 2) * rfz;
    dz = cell(2, 0) * rfx + cell(2, 1) * rfy + cell(2, 2) * rfz;
}

double frob_norm(const Mat3& m) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            s += m(i, j) * m(i, j);
    return std::sqrt(s);
}

Mat3 mat_mul(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
                s += a(i, k) * b(k, j);
            c(i, j) = s;
        }
    return c;
}

Mat3 mat_sub(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            c(i, j) = a(i, j) - b(i, j);
    return c;
}

Mat3 identity_3x3() { return Mat3::cubic(1.0); }

} // namespace

// ── ac-001: API surface ────────────────────────────────────────────────

TEST(BoxApi, FactoriesAndQueries) {
    auto bx = Box::orthorhombic(10.0, 12.0, 8.0);
    EXPECT_DOUBLE_EQ(bx.length(0), 10.0);
    EXPECT_DOUBLE_EQ(bx.length(1), 12.0);
    EXPECT_DOUBLE_EQ(bx.length(2), 8.0);
    EXPECT_TRUE(bx.is_orthorhombic());
    EXPECT_DOUBLE_EQ(bx.volume(), 10.0 * 12.0 * 8.0);

    auto cube = Box::cubic(7.5);
    EXPECT_DOUBLE_EQ(cube.length(0), 7.5);
    EXPECT_DOUBLE_EQ(cube.length(1), 7.5);
    EXPECT_DOUBLE_EQ(cube.length(2), 7.5);
    EXPECT_DOUBLE_EQ(cube.volume(), 7.5 * 7.5 * 7.5);
}

TEST(BoxApi, DefaultConstructIsZero) {
    Box bx;
    EXPECT_DOUBLE_EQ(bx.length(0), 0.0);
    EXPECT_DOUBLE_EQ(bx.volume(), 0.0);
    EXPECT_TRUE(bx.is_orthorhombic());
}

TEST(BoxApi, ConstructFromMat3Triclinic) {
    Mat3 cell = Mat3::diagonal(10.0, 10.0, 10.0);
    cell(0, 1) = 0.5; // small shear
    Box bx(cell);
    EXPECT_FALSE(bx.is_orthorhombic());
    EXPECT_NEAR(bx.volume(), 10.0 * 10.0 * 10.0, 1e-12);
    // cell · inv_cell ≈ I
    Mat3 prod = mat_mul(bx.cell(), bx.inv_cell());
    EXPECT_LT(frob_norm(mat_sub(prod, identity_3x3())), 1e-12);
}

TEST(BoxApi, ExplicitSetCell) {
    Box bx;
    bx.set_cell(Mat3::cubic(5.0));
    EXPECT_DOUBLE_EQ(bx.length(0), 5.0);
    EXPECT_DOUBLE_EQ(bx.cell()(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(bx.cell()(1, 2), 0.0);
}

// ── ac-006: ortho min-image numerical parity ───────────────────────────

TEST(BoxMinImage, OrthoParity10kRandom) {
    std::mt19937 rng(0xA70125u);
    std::uniform_real_distribution<double> Ldist(5.0, 50.0);
    std::uniform_real_distribution<double> dxdist(-2.0, 2.0); // multiplied by L

    for (int trial = 0; trial < 10000; ++trial) {
        const double Lx = Ldist(rng), Ly = Ldist(rng), Lz = Ldist(rng);
        const auto bx = Box::orthorhombic(Lx, Ly, Lz);

        double dx0 = dxdist(rng) * Lx;
        double dy0 = dxdist(rng) * Ly;
        double dz0 = dxdist(rng) * Lz;
        // Reference is single-period valid for |d| <= 1.5L; clamp.
        if (std::abs(dx0) > 1.5 * Lx)
            dx0 = std::copysign(1.5 * Lx - 1e-9, dx0);
        if (std::abs(dy0) > 1.5 * Ly)
            dy0 = std::copysign(1.5 * Ly - 1e-9, dy0);
        if (std::abs(dz0) > 1.5 * Lz)
            dz0 = std::copysign(1.5 * Lz - 1e-9, dz0);

        double dx = dx0, dy = dy0, dz = dz0;
        bx.min_image(dx, dy, dz);

        double rdx = dx0, rdy = dy0, rdz = dz0;
        min_image_reference(rdx, rdy, rdz, bx.cell());

        EXPECT_NEAR(dx, rdx, 1e-12) << "trial=" << trial << " Lx=" << Lx;
        EXPECT_NEAR(dy, rdy, 1e-12);
        EXPECT_NEAR(dz, rdz, 1e-12);

        // Post-condition: |d| <= L/2 + eps.
        EXPECT_LE(std::abs(dx), 0.5 * Lx + 1e-12);
        EXPECT_LE(std::abs(dy), 0.5 * Ly + 1e-12);
        EXPECT_LE(std::abs(dz), 0.5 * Lz + 1e-12);
    }
}

// ── ac-007: triclinic min-image roundtrip identity ─────────────────────

TEST(BoxMinImage, TriclinicSingleShear) {
    Mat3 cell = Mat3::diagonal(10.0, 10.0, 10.0);
    cell(0, 1) = 1.5; // xy shear
    Box bx(cell);

    std::mt19937 rng(0x7E1C0u);
    std::uniform_real_distribution<double> dist(-15.0, 15.0);

    for (int trial = 0; trial < 1000; ++trial) {
        const double dx0 = dist(rng), dy0 = dist(rng), dz0 = dist(rng);

        double dx = dx0, dy = dy0, dz = dz0;
        bx.min_image(dx, dy, dz);

        double rdx = dx0, rdy = dy0, rdz = dz0;
        min_image_triclinic_reference(rdx, rdy, rdz, bx.cell(), bx.inv_cell());

        EXPECT_NEAR(dx, rdx, 1e-10);
        EXPECT_NEAR(dy, rdy, 1e-10);
        EXPECT_NEAR(dz, rdz, 1e-10);

        // Fractional components within [-0.5, 0.5].
        const Mat3& iv = bx.inv_cell();
        const double fx = iv(0, 0) * dx + iv(0, 1) * dy + iv(0, 2) * dz;
        const double fy = iv(1, 0) * dx + iv(1, 1) * dy + iv(1, 2) * dz;
        const double fz = iv(2, 0) * dx + iv(2, 1) * dy + iv(2, 2) * dz;
        EXPECT_LE(std::abs(fx), 0.5 + 1e-12);
        EXPECT_LE(std::abs(fy), 0.5 + 1e-12);
        EXPECT_LE(std::abs(fz), 0.5 + 1e-12);
    }

    // Cache identity: cell · inv_cell - I.
    Mat3 prod = mat_mul(bx.cell(), bx.inv_cell());
    EXPECT_LT(frob_norm(mat_sub(prod, identity_3x3())), 1e-12);
}

TEST(BoxMinImage, TriclinicTwoShears) {
    Mat3 cell = Mat3::diagonal(10.0, 12.0, 8.0);
    cell(0, 1) = 1.0;
    cell(0, 2) = 0.5;
    Box bx(cell);
    EXPECT_FALSE(bx.is_orthorhombic());

    Mat3 prod = mat_mul(bx.cell(), bx.inv_cell());
    EXPECT_LT(frob_norm(mat_sub(prod, identity_3x3())), 1e-12);

    double dx = 9.0, dy = 11.0, dz = 7.0;
    bx.min_image(dx, dy, dz);
    const Mat3& iv = bx.inv_cell();
    const double fx = iv(0, 0) * dx + iv(0, 1) * dy + iv(0, 2) * dz;
    const double fy = iv(1, 0) * dx + iv(1, 1) * dy + iv(1, 2) * dz;
    const double fz = iv(2, 0) * dx + iv(2, 1) * dy + iv(2, 2) * dz;
    EXPECT_LE(std::abs(fx), 0.5 + 1e-12);
    EXPECT_LE(std::abs(fy), 0.5 + 1e-12);
    EXPECT_LE(std::abs(fz), 0.5 + 1e-12);
}

// ── wrap_axis parity ───────────────────────────────────────────────────

TEST(BoxWrap, WrapAxisParity10kRandom) {
    std::mt19937 rng(0xBEAD0u);
    std::uniform_real_distribution<double> Ldist(5.0, 50.0);
    std::uniform_real_distribution<double> qdist(-3.0, 3.0);

    for (int trial = 0; trial < 10000; ++trial) {
        const double Lx = Ldist(rng), Ly = Ldist(rng), Lz = Ldist(rng);
        const auto bx = Box::orthorhombic(Lx, Ly, Lz);

        const double qx = qdist(rng) * Lx;
        const double qy = qdist(rng) * Ly;
        const double qz = qdist(rng) * Lz;

        const double wx = bx.wrap_axis(qx, 0);
        const double wy = bx.wrap_axis(qy, 1);
        const double wz = bx.wrap_axis(qz, 2);

        const double rwx = wrap_axis_reference(qx, Lx);
        const double rwy = wrap_axis_reference(qy, Ly);
        const double rwz = wrap_axis_reference(qz, Lz);

        EXPECT_NEAR(wx, rwx, 1e-12);
        EXPECT_NEAR(wy, rwy, 1e-12);
        EXPECT_NEAR(wz, rwz, 1e-12);

        EXPECT_GE(wx, 0.0);
        EXPECT_LT(wx, Lx);
        EXPECT_GE(wy, 0.0);
        EXPECT_LT(wy, Ly);
        EXPECT_GE(wz, 0.0);
        EXPECT_LT(wz, Lz);
    }
}

TEST(BoxWrap, WrapIntoCallsAllAxes) {
    auto bx = Box::orthorhombic(10.0, 12.0, 8.0);
    double x = 25.0, y = -5.0, z = 19.5;
    bx.wrap_into(x, y, z);
    EXPECT_DOUBLE_EQ(x, 5.0);
    EXPECT_DOUBLE_EQ(y, 7.0);
    EXPECT_DOUBLE_EQ(z, 3.5);
}

// ── ac-009: cache coherence after mutation ─────────────────────────────

TEST(BoxMutation, ScaleIsotropicRefreshesCache) {
    auto bx = Box::orthorhombic(10.0, 10.0, 10.0);
    bx.scale_isotropic(1.5);
    EXPECT_DOUBLE_EQ(bx.length(0), 15.0);
    EXPECT_NEAR(bx.cell()(0, 0) * bx.inv_cell()(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(bx.cell()(1, 1) * bx.inv_cell()(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(bx.cell()(2, 2) * bx.inv_cell()(2, 2), 1.0, 1e-12);
}

TEST(BoxMutation, ScaleIsotropicShiftsMinImageBoundary) {
    auto bx = Box::orthorhombic(10.0, 10.0, 10.0);
    bx.scale_isotropic(1.5); // L = 15 → half_L = 7.5
    double dx = 9.0;
    double dy = 0.0;
    double dz = 0.0;
    bx.min_image(dx, dy, dz);
    EXPECT_NEAR(dx, 9.0 - 15.0, 1e-12); // -6.0
}

TEST(BoxMutation, ScaleDiagonalAniso) {
    auto bx = Box::orthorhombic(10.0, 10.0, 10.0);
    bx.scale_diagonal(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(bx.length(0), 10.0);
    EXPECT_DOUBLE_EQ(bx.length(1), 20.0);
    EXPECT_DOUBLE_EQ(bx.length(2), 30.0);
    EXPECT_NEAR(bx.cell()(1, 1) * bx.inv_cell()(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(bx.cell()(2, 2) * bx.inv_cell()(2, 2), 1.0, 1e-12);
}

TEST(BoxMutation, ScaleAxis) {
    auto bx = Box::orthorhombic(10.0, 10.0, 10.0);
    bx.scale_axis(1, 0.5);
    EXPECT_DOUBLE_EQ(bx.length(0), 10.0);
    EXPECT_DOUBLE_EQ(bx.length(1), 5.0);
    EXPECT_DOUBLE_EQ(bx.length(2), 10.0);
    EXPECT_NEAR(bx.cell()(1, 1) * bx.inv_cell()(1, 1), 1.0, 1e-12);
}

TEST(BoxMutation, SetCellRefreshes) {
    auto bx = Box::orthorhombic(10.0, 10.0, 10.0);
    bx.set_cell(Mat3::cubic(20.0));
    EXPECT_DOUBLE_EQ(bx.length(0), 20.0);
    EXPECT_NEAR(bx.cell()(0, 0) * bx.inv_cell()(0, 0), 1.0, 1e-12);

    Mat3 tri = Mat3::diagonal(10.0, 10.0, 10.0);
    tri(0, 1) = 1.0;
    bx.set_cell(tri);
    EXPECT_FALSE(bx.is_orthorhombic());
    Mat3 prod = mat_mul(bx.cell(), bx.inv_cell());
    EXPECT_LT(frob_norm(mat_sub(prod, identity_3x3())), 1e-12);
}

TEST(BoxMutation, RevisionAdvancesForEveryMutationAndAssignment) {
    auto box = Box::cubic(10.0);
    const auto initial = box.revision();
    box.scale_isotropic(1.0);
    EXPECT_GT(box.revision(), initial);

    const auto after_scale = box.revision();
    box.set_cell(Mat3::cubic(10.0));
    EXPECT_GT(box.revision(), after_scale);

    const auto before_assignment = box.revision();
    const auto replacement = Box::orthorhombic(8.0, 9.0, 10.0);
    box = replacement;
    EXPECT_GT(box.revision(), before_assignment);
    EXPECT_DOUBLE_EQ(box.length(0), 8.0);
    EXPECT_DOUBLE_EQ(box.length(1), 9.0);
}

// ── ac-008: xsimd batch parity ─────────────────────────────────────────

TEST(BoxBatch, OrthoLaneParity) {
    using batch_t = xsimd::batch<double>;
    constexpr std::size_t W = batch_t::size;

    auto bx = Box::orthorhombic(10.0, 12.0, 8.0);

    std::mt19937 rng(0xBA7C0u);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);

    for (int trial = 0; trial < 1000; ++trial) {
        alignas(64) double xb[16] = {}, yb[16] = {}, zb[16] = {};
        for (std::size_t lane = 0; lane < W; ++lane) {
            xb[lane] = dist(rng) * bx.length(0);
            yb[lane] = dist(rng) * bx.length(1);
            zb[lane] = dist(rng) * bx.length(2);
        }

        batch_t bdx = batch_t::load_aligned(xb);
        batch_t bdy = batch_t::load_aligned(yb);
        batch_t bdz = batch_t::load_aligned(zb);
        bx.min_image_batch(bdx, bdy, bdz);

        alignas(64) double bxr[16], byr[16], bzr[16];
        bdx.store_aligned(bxr);
        bdy.store_aligned(byr);
        bdz.store_aligned(bzr);

        for (std::size_t lane = 0; lane < W; ++lane) {
            double sx = xb[lane], sy = yb[lane], sz = zb[lane];
            bx.min_image(sx, sy, sz);
            // Bit-exact: both code paths reduce to the same IEEE
            // operations (compare + select vs compare + branch on the
            // same mask).
            EXPECT_EQ(bxr[lane], sx) << "trial=" << trial << " lane=" << lane;
            EXPECT_EQ(byr[lane], sy);
            EXPECT_EQ(bzr[lane], sz);
        }
    }
}
