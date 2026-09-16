/// @file tests/cpu/scf/test_cart2sph.cpp
/// @brief Unit: CartToSph coefficient matrices vs pinned PySCF
///        `gto.mole.cart2sph(l, normalized='sp')` values + the
///        double-factorial normalization pitfall (unit-normalized d-shell
///        self-overlap sandwich == identity).
///
/// Pinned oracle: PySCF 2.14.0, run externally per the oracle protocol
/// (docs/tutorial/appendix-pyscf-oracle.md):
///   from pyscf.gto.mole import cart2sph
///   cart2sph(l, normalized='sp')   # l = 0..3

#include <gtest/gtest.h>

#include <stdexcept>

#include "atomiverse/cpu/scf/internal/cart2sph.h"

using ATV::cpu::scf::internal::CartToSph;

namespace {

/// PySCF 2.14.0 cart2sph(2, normalized='sp') — rows: xx,xy,xz,yy,yz,zz;
/// cols m = -2..+2.
const double kD[6][5] = {
    {0.0, 0.0, -0.31539156525252, 0.0, 0.5462742152960396},
    {1.0925484305920792, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0925484305920792, 0.0},
    {0.0, 0.0, -0.31539156525252, 0.0, -0.5462742152960396},
    {0.0, 1.0925484305920792, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.63078313050504, 0.0, 0.0},
};

/// PySCF 2.14.0 cart2sph(3, normalized='sp') — rows: xxx,xxy,xxz,xyy,xyz,
/// xzz,yyy,yyz,yzz,zzz; cols m = -3..+3.
const double kF[10][7] = {
    {0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 0.5900435899266435},
    {1.7701307697799304, 0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 1.4453057213202771, 0.0},
    {0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, -1.7701307697799304},
    {0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.8281831978578629, 0.0, 0.0},
    {-0.5900435899266435, 0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, -1.1195289977703462, 0.0, -1.4453057213202771, 0.0},
    {0.0, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.7463526651802308, 0.0, 0.0, 0.0},
};

} // namespace

/// Shapes: cartesian (l+1)(l+2)/2 rows, spherical 2l+1 columns.
TEST(CartToSph, Shapes) {
    const CartToSph t;
    EXPECT_EQ(t.matrix(0).shape(0), 1u);
    EXPECT_EQ(t.matrix(0).shape(1), 1u);
    EXPECT_EQ(t.matrix(1).shape(0), 3u);
    EXPECT_EQ(t.matrix(1).shape(1), 3u);
    EXPECT_EQ(t.matrix(2).shape(0), 6u);
    EXPECT_EQ(t.matrix(2).shape(1), 5u);
    EXPECT_EQ(t.matrix(3).shape(0), 10u);
    EXPECT_EQ(t.matrix(3).shape(1), 7u);
}

/// l = 0 and l = 1 are identities in the PySCF 'sp'-normalized convention.
TEST(CartToSph, SAndPAreIdentity) {
    const CartToSph t;
    for (int l = 0; l <= 1; ++l) {
        const auto& m = t.matrix(l);
        for (std::size_t i = 0; i < m.shape(0); ++i)
            for (std::size_t j = 0; j < m.shape(1); ++j)
                EXPECT_NEAR(m(i, j), i == j ? 1.0 : 0.0, 1e-15);
    }
}

/// Every coefficient equals the pinned PySCF cart2sph value to 1e-12.
TEST(CartToSph, MatchesPinnedPyscfCoefficients) {
    const CartToSph t;
    const auto& d = t.matrix(2);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 5; ++j)
            EXPECT_NEAR(d(i, j), kD[i][j], 1e-12)
                << "d(" << i << "," << j << ")";
    const auto& f = t.matrix(3);
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 7; ++j)
            EXPECT_NEAR(f(i, j), kF[i][j], 1e-12)
                << "f(" << i << "," << j << ")";
}

/// THE double-factorial pitfall: in the engine's per-component
/// unit-normalized Cartesian convention, the single-primitive d-shell
/// self-overlap is S = I + (1/3)(xx,yy),(xx,zz),(yy,zz); the own-convention
/// transform block(2) must sandwich it to the 5x5 identity to 1e-12.
TEST(CartToSph, UnitNormalizedDShellSandwichIsIdentity) {
    const CartToSph t;
    const auto& b = t.block(2); // 6x5, double-factorial scaling absorbed
    double s[6][6] = {};
    for (int i = 0; i < 6; ++i)
        s[i][i] = 1.0;
    // cart order xx,xy,xz,yy,yz,zz: diagonal-pair overlaps are 1/3.
    s[0][3] = s[3][0] = 1.0 / 3.0; // xx-yy
    s[0][5] = s[5][0] = 1.0 / 3.0; // xx-zz
    s[3][5] = s[5][3] = 1.0 / 3.0; // yy-zz
    for (int m = 0; m < 5; ++m) {
        for (int n = 0; n < 5; ++n) {
            double acc = 0.0;
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    acc += b(i, m) * s[i][j] * b(j, n);
            EXPECT_NEAR(acc, m == n ? 1.0 : 0.0, 1e-12)
                << "sandwich(" << m << "," << n << ")";
        }
    }
}

/// Unsupported l throws — no fallback. l = 4 (g) became supported with
/// the density-fitting link (def2-JKFIT carries g shells); the first
/// unsupported l is now 5 (h).
TEST(CartToSph, UnsupportedLThrows) {
    const CartToSph t;
    EXPECT_THROW(t.matrix(5), std::out_of_range);
    EXPECT_THROW(t.block(5), std::out_of_range);
    EXPECT_THROW(t.matrix(-1), std::out_of_range);
    // l = 4 is served (15 × 9).
    EXPECT_EQ(t.matrix(4).shape()[0], 15u);
    EXPECT_EQ(t.matrix(4).shape()[1], 9u);
}
