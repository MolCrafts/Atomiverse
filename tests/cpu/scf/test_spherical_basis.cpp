/// @file tests/cpu/scf/test_spherical_basis.cpp
/// @brief Unit: spherical AO dimension switch for spherical-convention bases.
///
/// def2 sets are DEFINED with real spherical harmonics (5d/7f); Pople sets
/// stay Cartesian (6d by definition). The IntegralPack sandwiches its
/// matrices to the spherical dimension via the engine-convention c2s blocks;
/// the unit-diagonal of the spherical overlap is the end-to-end proof that
/// the double-factorial normalization was absorbed correctly.
///
/// Pinned oracle values: PySCF 2.14.0, H2O geometry
/// O(0,0,0.1173)/H(0,±0.7572,−0.4692) Å, run externally per
/// docs/tutorial/appendix-pyscf-oracle.md.

#include <gtest/gtest.h>

#include <stdexcept>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::IntegralPack;
using cpu::scf::angular_convention;
using cpu::scf::basis_data::AngularConvention;

namespace {

cpu::Frame water() {
    cpu::Frame f(3);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto t = f.span(kAtom_type);
    t[0] = 8;
    x[0] = 0.0;
    y[0] = 0.0;
    z[0] = 0.1173;
    t[1] = 1;
    x[1] = 0.0;
    y[1] = 0.7572;
    z[1] = -0.4692;
    t[2] = 1;
    x[2] = 0.0;
    y[2] = -0.7572;
    z[2] = -0.4692;
    f.simbox = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

/// The per-set convention gate.
TEST(SphericalBasis, ConventionGate) {
    EXPECT_EQ(angular_convention(BasisSet::STO3G),
              AngularConvention::Cartesian);
    EXPECT_EQ(angular_convention(BasisSet::Pople631G),
              AngularConvention::Cartesian);
    EXPECT_EQ(angular_convention(BasisSet::Pople631Gs),
              AngularConvention::Cartesian);
    EXPECT_EQ(angular_convention(BasisSet::Def2SVP),
              AngularConvention::Spherical);
    EXPECT_EQ(angular_convention(BasisSet::Def2TZVP),
              AngularConvention::Spherical);
}

/// H2O/def2-SVP: 25 Cartesian AOs collapse to 24 spherical AOs.
TEST(SphericalBasis, Def2SvpDimensionIs24) {
    const auto f = water();
    const IntegralPack p{f, BasisSet::Def2SVP};
    EXPECT_TRUE(p.spherical());
    EXPECT_EQ(p.n_basis, 24);
    EXPECT_EQ(p.n_basis_cart, 25);
    EXPECT_EQ(p.n_ao_cart(), 25);
    EXPECT_EQ(static_cast<int>(p.S.shape(0)), 24);
    EXPECT_EQ(static_cast<int>(p.Hcore.shape(1)), 24);
}

/// 6-31G* stays Cartesian by definition: dimension unchanged (19 for H2O).
TEST(SphericalBasis, Pople631GsStaysCartesian) {
    const auto f = water();
    const IntegralPack p{f, BasisSet::Pople631Gs};
    EXPECT_FALSE(p.spherical());
    EXPECT_EQ(p.n_basis, 19);
    EXPECT_EQ(p.n_ao_cart(), 19);
}

/// The spherical overlap has an exactly unit diagonal — the sandwich with the
/// double-factorial-absorbed c2s blocks lands on normalized spherical AOs.
/// (The explicit 5x5 sandwich identity at 1e-12 lives in test_cart2sph.cpp.)
TEST(SphericalBasis, SphericalOverlapDiagonalIsUnit) {
    const auto f = water();
    const IntegralPack p{f, BasisSet::Def2SVP};
    for (int i = 0; i < p.n_basis; ++i)
        EXPECT_NEAR(p.S(i, i), 1.0, 1e-12) << "S diag " << i;
}

/// Spot-check off-diagonal spherical overlap vs pinned PySCF values
/// (PySCF 2.14.0, cart=False; tolerance loosened to 1e-8 for the Å→Bohr
/// constant difference between codes).
TEST(SphericalBasis, SphericalOverlapMatchesPinnedPyscf) {
    const auto f = water();
    const IntegralPack p{f, BasisSet::Def2SVP};
    EXPECT_NEAR(p.S(0, 1), -0.3440165538027238, 1e-8);
    EXPECT_NEAR(p.S(0, 14), -0.04787596383062501, 1e-8);
    EXPECT_NEAR(p.S(5, 20), -0.06485770181065395, 1e-8);
    EXPECT_NEAR(p.S(9, 10), 0.0, 1e-12); // dxy ⊥ dyz on the same center
}

/// Analytic gradients refuse spherical bases loudly — never silently wrong.
TEST(SphericalBasis, AnalyticGradientThrowsForSpherical) {
    const auto f = water();
    const IntegralPack p{f, BasisSet::Def2SVP};
    // Minimal valid restricted determinant; the guard throws before any use.
    cpu::linalg::RealMatrix C = xt::zeros<double>({std::size_t(1), std::size_t(1)});
    C(0, 0) = 1.0;
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(1)});
    cpu::linalg::RealVector occ = xt::ones<double>({std::size_t(1)});
    const cpu::scf::SlaterDeterminant det{C, eps, occ, 1};
    const cpu::scf::internal::ScfGradient grad{
        cpu::scf::MeanFieldType::HF, cpu::scf::SpinType::Restricted,
        cpu::scf::XCType::LSDA,      cpu::scf::GridLevel::SG0,
        0,                           0,
        false,                       false};
    EXPECT_THROW(grad.compute(f, p, det, 0.0), std::runtime_error);
}
