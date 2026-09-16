/// @file tests/longrange/test_fmm_operators.cpp
/// @brief Unit tests for M2M / M2L / L2L translation operators.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_operators.h"
#include "atomiverse/cpu/longrange/multipole_expansion.h"

using namespace ATV;
using ATV::longrange::L2L;
using ATV::longrange::LocalExpansion;
using ATV::longrange::M2L;
using ATV::longrange::M2M;
using ATV::longrange::MultipoleExpansion;

namespace {

/// Direct P2M of charges about `center`.
MultipoleExpansion p2m_direct(int p, const Vec3& center,
                              const std::vector<Vec3>& pos,
                              const std::vector<AccReal>& q) {
    MultipoleExpansion M(p, center);
    for (std::size_t i = 0; i < pos.size(); ++i)
        M.accumulate(q[i], pos[i]);
    return M;
}

} // namespace

TEST(FmmOperators, RejectsNegativeOrder) {
    EXPECT_THROW(M2M(-1), std::invalid_argument);
    EXPECT_THROW(M2L(-1), std::invalid_argument);
    EXPECT_THROW(L2L(-1), std::invalid_argument);
}

TEST(FmmOperators, M2MExactWithinOrder) {
    // Cluster of 4 charges near child centre; compare M2M-to-parent vs
    // direct P2M about parent — coefficients and M2P evaluation.
    const int p = 6;
    const Vec3 child{0.5, -0.3, 0.2};
    const Vec3 parent{0.0, 0.0, 0.0};
    const std::vector<Vec3> pos = {
        child + Vec3{0.05, 0.02, -0.01},
        child + Vec3{-0.04, 0.03, 0.02},
        child + Vec3{0.01, -0.05, 0.03},
        child + Vec3{0.02, 0.01, -0.04},
    };
    const std::vector<AccReal> q = {1.0, -0.5, 0.7, -1.2};

    MultipoleExpansion Mchild = p2m_direct(p, child, pos, q);
    MultipoleExpansion Mparent_op(p, parent);
    M2M m2m(p);
    m2m.apply(Mchild, Mparent_op);

    MultipoleExpansion Mparent_ref = p2m_direct(p, parent, pos, q);

    // Coefficient match (exact within p).
    for (int i = 0; i < Mparent_op.ncoef(); ++i) {
        EXPECT_NEAR(Mparent_op.coeffs()[static_cast<std::size_t>(i)].real(),
                    Mparent_ref.coeffs()[static_cast<std::size_t>(i)].real(),
                    1e-12)
            << "coef " << i << " real";
        EXPECT_NEAR(Mparent_op.coeffs()[static_cast<std::size_t>(i)].imag(),
                    Mparent_ref.coeffs()[static_cast<std::size_t>(i)].imag(),
                    1e-12)
            << "coef " << i << " imag";
    }

    // Far-field evaluation match.
    const Vec3 far{5.0, 1.0, -2.0};
    EXPECT_NEAR(Mparent_op.evaluate(far), Mparent_ref.evaluate(far), 1e-12);
}

TEST(FmmOperators, L2LExactWithinOrder) {
    // L2L is exact as a polynomial shift: L2P after L2L must match L2P of the
    // parent expansion at the same point (to 1e-12). Comparing against a
    // re-M2L about the child is invalid — M2L truncation depends on centre.
    const int p = 6;
    const Vec3 parent{0, 0, 0};
    const Vec3 child{0.3, -0.2, 0.1};
    const Vec3 src_center{6.0, 0.0, 0.0};
    const Vec3 src_pos = src_center + Vec3{0.1, -0.05, 0.02};
    const AccReal q = 1.3;

    MultipoleExpansion M(p, src_center);
    M.accumulate(q, src_pos);

    LocalExpansion Lparent(p, parent);
    M2L m2l(p);
    m2l.apply(M, Lparent);

    LocalExpansion Lchild(p, child);
    L2L l2l(p);
    l2l.apply(Lparent, Lchild);

    const Vec3 eval = child + Vec3{0.05, 0.01, -0.02};
    EXPECT_NEAR(Lchild.evaluate(eval), Lparent.evaluate(eval), 1e-12);

    // Also near the true potential (M2L truncation at this separation).
    const AccReal exact = q / (eval - src_pos).norm();
    EXPECT_NEAR(Lchild.evaluate(eval), exact, 1e-8);
}

TEST(FmmOperators, M2LGeometricDecay) {
    // Fixed separation ratio c: source ball radius r≈0.2, |t|=4 ⇒ c≈20.
    // Error of M2L+L2P vs direct 1/r should decay ~ (1/c)^p.
    const Vec3 src_c{0, 0, 0};
    const Vec3 loc_c{4, 0, 0};
    const Vec3 src_pos{0.1, -0.15, 0.05};
    const Vec3 eval = loc_c + Vec3{0.08, 0.04, -0.03};
    const AccReal q = 1.0;
    const AccReal exact = q / (eval - src_pos).norm();
    // c ≈ |t| / r_source ≈ 4 / 0.2 = 20
    const AccReal c_ratio = 4.0 / 0.2;

    AccReal prev_err = 1e300;
    for (int p = 1; p <= 10; ++p) {
        MultipoleExpansion M(p, src_c);
        M.accumulate(q, src_pos);
        LocalExpansion L(p, loc_c);
        M2L m2l(p);
        m2l.apply(M, L);
        const AccReal err = std::abs(L.evaluate(eval) - exact);
        EXPECT_LT(err, prev_err) << "p=" << p;
        // Loose check: error is not worse than O((1/c)^{p/2}) (factor-of-safety
        // against angular prefactors); main pin is geometric decay.
        const AccReal bound =
            10.0 / (c_ratio - 1.0) * std::pow(1.0 / c_ratio, p * 0.5);
        EXPECT_LT(err, bound) << "p=" << p;
        prev_err = err;
    }
}
