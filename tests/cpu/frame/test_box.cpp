/// @file tests/cpu/frame/test_box.cpp
/// @brief Coverage for `cpu::Box`: API, ortho min-image parity, triclinic
///        roundtrip identity, wrap-axis parity, mutation cache coherence,
///        xsimd batch parity.
///
/// Layout mirrors `include/atomiverse/cpu/frame/box.h` (+ `box_simd.h`).
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

// Independent lattice enumeration oracle for these bounded test fixtures.
void min_image_triclinic_reference(double& dx, double& dy, double& dz,
                                   const Mat3& cell, const Mat3&) {
    const ATV::Vec3 original{dx, dy, dz};
    double best2 = original.norm_squared();
    for (int i = -4; i <= 4; ++i)
        for (int j = -4; j <= 4; ++j)
            for (int k = -4; k <= 4; ++k) {
                const auto r = original - cell * ATV::Vec3(i, j, k);
                if (r.norm_squared() < best2) {
                    best2 = r.norm_squared();
                    dx = r[0];
                    dy = r[1];
                    dz = r[2];
                }
            }
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

/// @brief `recip = 2π (cell^{-1})^T` ⇒ `recip^T · cell = 2π I`.
///        Hard-coded analytical identity — covers the cold path retained
///        when `core/box.h` was deleted in favour of `cpu::Box`.
TEST(BoxApi, RecipCellTransposeTimesCellIs2PiI) {
    using ATV::units::pi;
    const double two_pi = 2.0 * pi;

    auto check = [&](const Box& bx, double tol, const char* tag) {
        // recip^T = 2π inv_cell  (by definition of recip_cell)
        const Mat3 recip = bx.recip_cell();
        Mat3 recip_T{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                recip_T(i, j) = recip(j, i);
        const Mat3 prod = mat_mul(recip_T, bx.cell());
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const double expected = (i == j) ? two_pi : 0.0;
                EXPECT_NEAR(prod(i, j), expected, tol)
                    << tag << " recip^T*cell != 2πI at (" << i << "," << j
                    << ")";
            }
    };

    check(Box::orthorhombic(5.0, 12.0, 9.0), 1e-12, "ortho");

    Mat3 cell = Mat3::diagonal(10.0, 11.0, 12.0);
    cell(0, 1) = 1.5;
    cell(0, 2) = 0.4;
    check(Box{cell}, 1e-10, "triclinic");
}

/// @brief `lengths()` = Euclidean norms of cell columns (not only diagonal).
TEST(BoxApi, LengthsColumnNorms) {
    auto ortho = Box::cubic(10.0);
    const auto lo = ortho.lengths();
    EXPECT_NEAR(lo[0], 10.0, 1e-12);
    EXPECT_NEAR(lo[1], 10.0, 1e-12);
    EXPECT_NEAR(lo[2], 10.0, 1e-12);

    Mat3 cell = Mat3::diagonal(3.0, 4.0, 5.0);
    cell(0, 1) = 0.0;
    cell(0, 2) = 0.0;
    // Columns are (6,0,0), (8,4,0), (0,0,5).
    cell(0, 0) = 6.0;
    cell(0, 1) = 8.0;
    cell(0, 2) = 0.0;
    Box tri(cell);
    const auto lt = tri.lengths();
    EXPECT_NEAR(lt[0], 6.0, 1e-12);
    EXPECT_NEAR(lt[1], std::sqrt(80.0), 1e-12);
    EXPECT_NEAR(lt[2], 5.0, 1e-12);
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
    double rx = 9, ry = 11, rz = 7;
    min_image_triclinic_reference(rx, ry, rz, bx.cell(), bx.inv_cell());
    EXPECT_NEAR(dx, rx, 1e-12);
    EXPECT_NEAR(dy, ry, 1e-12);
    EXPECT_NEAR(dz, rz, 1e-12);
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
    bx.wrap_into(x, y, z, nullptr);
    EXPECT_DOUBLE_EQ(x, 5.0);
    EXPECT_DOUBLE_EQ(y, 7.0);
    EXPECT_DOUBLE_EQ(z, 3.5);
}

TEST(BoxWrap, WrapIntoCountsOrthorhombicTranslationsIntoImage) {
    auto bx = Box::orthorhombic(10.0, 12.0, 8.0);
    // Far beyond an int8 counter: 1000 cells along x, -300 along y.
    double x = 10000.0 + 5.0, y = -300.0 * 12.0 + 7.0, z = 19.5;
    ATV::ImageInt image[3] = {127, -128, 5};
    bx.wrap_into(x, y, z, image);
    EXPECT_NEAR(x, 5.0, 1e-9);
    EXPECT_NEAR(y, 7.0, 1e-9);
    EXPECT_DOUBLE_EQ(z, 3.5);
    EXPECT_EQ(image[0], 127 + 1000);
    EXPECT_EQ(image[1], -128 - 300);
    EXPECT_EQ(image[2], 5 + 2);
}

TEST(BoxWrap, WrapIntoLeavesImageOfAnInsidePointAlone) {
    auto bx = Box::orthorhombic(10.0, 12.0, 8.0);
    double x = 0.0, y = 11.999, z = 4.0;
    ATV::ImageInt image[3] = {3, -4, 0};
    bx.wrap_into(x, y, z, image);
    EXPECT_EQ(image[0], 3);
    EXPECT_EQ(image[1], -4);
    EXPECT_EQ(image[2], 0);
}

TEST(BoxGeometry, GeneralCellTransformsWrapAndBounds) {
    Mat3 h = Mat3::diagonal(4, 5, 6);
    h(0, 1) = 2;
    h(1, 0) = -1;
    h(1, 2) = 1;
    h(2, 0) = .5;
    const Box box(h);
    const ATV::Vec3 fractional{-1.25, 2.5, .75};
    const auto r = box.to_cartesian(fractional);
    const auto restored = box.to_fractional(r);
    for (int d = 0; d < 3; ++d)
        EXPECT_NEAR(restored[d], fractional[d], 1e-14);
    double x = r[0], y = r[1], z = r[2];
    ATV::ImageInt image[3] = {0, 0, 0};
    box.wrap_into(x, y, z, image);
    const auto wrapped = box.to_fractional({x, y, z});
    EXPECT_NEAR(wrapped[0], .75, 1e-14);
    EXPECT_NEAR(wrapped[1], .5, 1e-14);
    EXPECT_NEAR(wrapped[2], .75, 1e-14);
    // s = (-1.25, 2.5, .75) → (.75, .5, .75): n = (-2, 2, 0).
    EXPECT_EQ(image[0], -2);
    EXPECT_EQ(image[1], 2);
    EXPECT_EQ(image[2], 0);
    const auto [lo, hi] = box.cartesian_bounds({0, 0, 0}, {1, 1, 1});
    EXPECT_DOUBLE_EQ(lo[0], 0);
    EXPECT_DOUBLE_EQ(hi[0], 6);
    EXPECT_DOUBLE_EQ(lo[1], -1);
    EXPECT_DOUBLE_EQ(hi[1], 6);
    EXPECT_DOUBLE_EQ(lo[2], 0);
    EXPECT_DOUBLE_EQ(hi[2], 6.5);
    const auto a = h * ATV::Vec3{1, 0, 0};
    const auto b = h * ATV::Vec3{0, 1, 0};
    const auto c = h * ATV::Vec3{0, 0, 1};
    EXPECT_NEAR(box.face_height(0), box.volume() / b.cross(c).norm(), 1e-13);
    EXPECT_NEAR(box.face_height(1), box.volume() / c.cross(a).norm(), 1e-13);
    EXPECT_NEAR(box.face_height(2), box.volume() / a.cross(b).norm(), 1e-13);
}

TEST(BoxMinImage, FractionalRoundingIsNotTheNearestImage) {
    Mat3 h = Mat3::diagonal(10, 2, 8);
    h(0, 1) = 9;
    const Box box(h);
    const auto r = box.to_cartesian({.49, .49, 0});
    double x = r[0], y = r[1], z = r[2];
    box.min_image(x, y, z);
    double rx = r[0], ry = r[1], rz = r[2];
    min_image_triclinic_reference(rx, ry, rz, h, box.inv_cell());
    EXPECT_NEAR(x, rx, 1e-12);
    EXPECT_NEAR(y, ry, 1e-12);
    EXPECT_NEAR(z, rz, 1e-12);
    EXPECT_LT(x * x + y * y + z * z, r.norm_squared());
}

TEST(BoxMutation, InvalidCellLeavesGeometryAndRevisionUnchanged) {
    Box box = Box::cubic(5);
    const auto revision = box.revision();
    Mat3 singular = Mat3::identity();
    singular(0, 1) = 1;
    singular(1, 0) = 1;
    EXPECT_THROW(box.set_cell(singular), std::runtime_error);
    Mat3 bad = Mat3::identity();
    bad(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(box.set_cell(bad), std::invalid_argument);
    EXPECT_EQ(box.revision(), revision);
    EXPECT_DOUBLE_EQ(box.volume(), 125);
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
    std::uniform_real_distribution<double> dist(-8.0, 8.0);

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
            // Both paths use the same division and nearest-integer reduction.
            EXPECT_EQ(bxr[lane], sx) << "trial=" << trial << " lane=" << lane;
            EXPECT_EQ(byr[lane], sy);
            EXPECT_EQ(bzr[lane], sz);
        }
    }
}

TEST(BoxMinImage,
     ExtremeIntegerShearDoesNotEnumerateUnrepresentableImageRange) {
    // The same Z^3 lattice in a catastrophically skew basis. The old
    // reciprocal-box loop started beyond 2^53, where double += 1 stalls.
    Mat3 h = Mat3::identity();
    h(0, 1) = 1e16;
    h(1, 2) = 1e16;
    const Box box(h);
    double x = .25, y = .49, z = -.2;
    box.min_image(x, y, z);
    EXPECT_NEAR(x, .25, 1e-13);
    EXPECT_NEAR(y, .49, 1e-13);
    EXPECT_NEAR(z, -.2, 1e-13);
    // Public cell and coordinate system remain untouched by private reduction.
    EXPECT_DOUBLE_EQ(box.cell()(0, 1), 1e16);
    Box assigned = Box::cubic(2);
    assigned = box;
    x = 1.25;
    y = -2.51;
    z = 3.8;
    assigned.min_image(x, y, z);
    EXPECT_NEAR(x, .25, 1e-13);
    EXPECT_NEAR(y, .49, 1e-13);
    EXPECT_NEAR(z, -.2, 1e-13);
}

TEST(BoxMinImage, RotatedAsymmetricCellMatchesIndependentLatticeOracle) {
    Mat3 h = Mat3::diagonal(4, 3, 5);
    h(0, 1) = 2.7;
    h(1, 2) = -.8;
    h(2, 0) = .3;
    Mat3 rotation = Mat3::identity();
    rotation(0, 0) = std::cos(.47);
    rotation(0, 1) = -std::sin(.47);
    rotation(1, 0) = std::sin(.47);
    rotation(1, 1) = std::cos(.47);
    h = rotation * h;
    const Box box(h);
    std::mt19937 rng(147u);
    std::uniform_real_distribution<double> fraction(-1.5, 1.5);
    for (int trial = 0; trial < 80; ++trial) {
        const auto r =
            h * ATV::Vec3{fraction(rng), fraction(rng), fraction(rng)};
        double x = r[0], y = r[1], z = r[2];
        double rx = x, ry = y, rz = z;
        min_image_triclinic_reference(rx, ry, rz, h, box.inv_cell());
        box.min_image(x, y, z);
        EXPECT_NEAR(x, rx, 2e-12);
        EXPECT_NEAR(y, ry, 2e-12);
        EXPECT_NEAR(z, rz, 2e-12);
    }
}

TEST(BoxGeometry, NegativeDiagonalGeneralCellKeepsSignedCoordinateTransform) {
    const Box box(Mat3::diagonal(-4, -5, 6));
    EXPECT_FALSE(box.is_orthorhombic());
    EXPECT_DOUBLE_EQ(box.volume(), 120);
    EXPECT_DOUBLE_EQ(box.face_height(0), 4);
    const auto r = box.to_cartesian({.25, .3, .4});
    EXPECT_DOUBLE_EQ(r[0], -1);
    EXPECT_DOUBLE_EQ(r[1], -1.5);
    double x = 3.2, y = -4.1, z = 5.3;
    box.min_image(x, y, z);
    EXPECT_NEAR(x, -.8, 1e-13);
    EXPECT_NEAR(y, .9, 1e-13);
    EXPECT_NEAR(z, -.7, 1e-13);
    EXPECT_THROW(Box::orthorhombic(-4, -5, 6), std::invalid_argument);
    EXPECT_THROW(Box::cubic(0), std::invalid_argument);
}

TEST(BoxWrap, TinyNegativeFractionDoesNotRoundToUpperFace) {
    Mat3 h = Mat3::diagonal(4, 5, 6);
    h(0, 1) = 1;
    const Box box(h);
    double x = -1e-300, y = 0, z = 0;
    // The fraction rounds back to 0, not to the upper face: no lattice
    // vector was removed, so the image must not move either.
    ATV::ImageInt image[3] = {0, 0, 0};
    box.wrap_into(x, y, z, image);
    EXPECT_EQ(image[0], 0);
    EXPECT_EQ(image[1], 0);
    EXPECT_EQ(image[2], 0);
    const auto f = box.to_fractional({x, y, z});
    for (int d = 0; d < 3; ++d) {
        EXPECT_GE(f[d], 0);
        EXPECT_LT(f[d], 1);
    }
    EXPECT_DOUBLE_EQ(x, 0);
}
