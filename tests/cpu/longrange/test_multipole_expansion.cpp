/// @file tests/longrange/test_multipole_expansion.cpp
/// @brief Unit tests for MultipoleExpansion / LocalExpansion (P2M, M2P, L2P).

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_operators.h"
#include "atomiverse/cpu/longrange/multipole_expansion.h"

using namespace ATV;
using ATV::longrange::irregular_solid;
using ATV::longrange::LocalExpansion;
using ATV::longrange::M2L;
using ATV::longrange::MultipoleExpansion;
using ATV::longrange::regular_solid;

TEST(MultipoleExpansion, RejectsNegativeOrder) {
    EXPECT_THROW(MultipoleExpansion(-1, Vec3::zeros()), std::invalid_argument);
    EXPECT_THROW(LocalExpansion(-1, Vec3::zeros()), std::invalid_argument);
}

TEST(MultipoleExpansion, MonopoleExact) {
    // Single charge at the expansion centre → M_00 = q, Φ = q/r.
    MultipoleExpansion M(4, Vec3{0, 0, 0});
    M.accumulate(2.5, Vec3{0, 0, 0});
    const Vec3 p{3.0, 4.0, 0.0}; // |p| = 5
    EXPECT_NEAR(M.evaluate(p), 2.5 / 5.0, 1e-12);
}

TEST(MultipoleExpansion, P2M_M2P_GeometricConvergence) {
    // Source at (0.1, -0.2, 0.15); evaluate at (3, 0, 0). Separation ratio
    // c ≈ 3 / 0.27 ≈ 11 — error should drop geometrically with p.
    const Vec3 src{0.1, -0.2, 0.15};
    const Vec3 eval{3.0, 0.0, 0.0};
    const AccReal q = 1.0;
    const AccReal exact = q / (eval - src).norm();

    AccReal prev_err = 1e300;
    for (int p = 0; p <= 10; ++p) {
        MultipoleExpansion M(p, Vec3::zeros());
        M.accumulate(q, src);
        const AccReal approx = M.evaluate(eval);
        const AccReal err = std::abs(approx - exact);
        if (p >= 2)
            EXPECT_LT(err, prev_err) << "p=" << p;
        // High-p absolute accuracy.
        if (p >= 8)
            EXPECT_LT(err, 1e-10) << "p=" << p;
        prev_err = err;
    }
}

TEST(MultipoleExpansion, RegularIrregularProductIsOneOverR) {
    // R_00 I_00 = 1/r.
    const Vec3 r{1.2, -0.7, 0.4};
    const auto R = regular_solid(0, 0, r);
    const auto I = irregular_solid(0, 0, r);
    EXPECT_NEAR((R * I).real(), 1.0 / r.norm(), 1e-14);
    EXPECT_NEAR((R * I).imag(), 0.0, 1e-14);
}

TEST(MultipoleExpansion, L2P_ConvergesViaM2L) {
    // Far multipole of one charge → local expansion about another centre → L2P.
    const Vec3 src_pos{0.05, 0.02, -0.03};
    const Vec3 src_center{0, 0, 0};
    const Vec3 loc_center{4.0, 0.0, 0.0};
    const Vec3 eval{4.1, 0.05, -0.02};
    const AccReal q = -1.5;
    const AccReal exact = q / (eval - src_pos).norm();

    AccReal prev_err = 1e300;
    for (int p = 2; p <= 12; ++p) {
        MultipoleExpansion M(p, src_center);
        M.accumulate(q, src_pos);
        LocalExpansion L(p, loc_center);
        M2L m2l(p);
        m2l.apply(M, L);
        const AccReal approx = L.evaluate(eval);
        const AccReal err = std::abs(approx - exact);
        // Strict decrease until floor; allow equality once under roundoff.
        if (prev_err > 1e-14)
            EXPECT_LT(err, prev_err) << "p=" << p;
        if (p >= 10)
            EXPECT_LT(err, 1e-10) << "p=" << p;
        prev_err = err;
    }
}
