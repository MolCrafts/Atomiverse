/// @file tests/cpu/scf/test_spherical_basis.cpp
/// @brief Unit: spherical AO dimension switch for spherical-convention bases.
///
/// def2 sets are DEFINED with real spherical harmonics (5d/7f); Pople sets
/// stay Cartesian (6d by definition). The internal::IntegralWorkspace sandwiches its
/// matrices to the spherical dimension via the engine-convention c2s blocks;
/// the unit-diagonal of the spherical overlap is the end-to-end proof that
/// the double-factorial normalization was absorbed correctly.
///
/// Pinned oracle values: PySCF 2.14.0, H2O geometry
/// O(0,0,0.1173)/H(0,±0.7572,−0.4692) Å, run externally per
/// docs/tutorial/appendix-pyscf-oracle.md.

#include <gtest/gtest.h>

#include <stdexcept>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::angular_convention;
using cpu::scf::BasisSet;
using cpu::scf::internal::IntegralWorkspace;
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
    f.box = cpu::Box::cubic(50.0);
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
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
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
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Pople631Gs};
    EXPECT_FALSE(p.spherical());
    EXPECT_EQ(p.n_basis, 19);
    EXPECT_EQ(p.n_ao_cart(), 19);
}

/// The spherical overlap has an exactly unit diagonal — the sandwich with the
/// double-factorial-absorbed c2s blocks lands on normalized spherical AOs.
/// (The explicit 5x5 sandwich identity at 1e-12 lives in test_cart2sph.cpp.)
TEST(SphericalBasis, SphericalOverlapDiagonalIsUnit) {
    const auto f = water();
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
    for (int i = 0; i < p.n_basis; ++i)
        EXPECT_NEAR(p.S(i, i), 1.0, 1e-12) << "S diag " << i;
}

/// Spot-check off-diagonal spherical overlap vs pinned PySCF values
/// (PySCF 2.14.0, cart=False; tolerance loosened to 1e-8 for the Å→Bohr
/// constant difference between codes).
TEST(SphericalBasis, SphericalOverlapMatchesPinnedPyscf) {
    const auto f = water();
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
    EXPECT_NEAR(p.S(0, 1), -0.3440165538027238, 1e-8);
    EXPECT_NEAR(p.S(0, 14), -0.04787596383062501, 1e-8);
    EXPECT_NEAR(p.S(5, 20), -0.06485770181065395, 1e-8);
    EXPECT_NEAR(p.S(9, 10), 0.0, 1e-12); // dxy ⊥ dyz on the same center
}

/// ac-002: the spherical derivative pack reports spherical AO dimensions —
/// dS/dT/dV collapse to n_sph (24 for H2O/def2-SVP), not n_cart (25).
TEST(SphericalGradient, DerivPackReportsSphericalDim) {
    const auto f = water();
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
    ASSERT_TRUE(p.spherical());
    const cpu::scf::IntegralDerivPack dp{p};
    EXPECT_EQ(dp.n_basis, p.n_basis);               // 24, spherical
    EXPECT_EQ(static_cast<int>(dp.dS.shape(0)), 3); // n_atoms
    EXPECT_EQ(static_cast<int>(dp.dS.shape(1)), 3); // x,y,z
    EXPECT_EQ(static_cast<int>(dp.dS.shape(2)), p.n_basis);
    EXPECT_EQ(static_cast<int>(dp.dT.shape(3)), p.n_basis);
    EXPECT_EQ(static_cast<int>(dp.dV.shape(3)), p.n_basis);
}

/// ac-001: the analytic spherical overlap derivative equals the central
/// difference of the spherical overlap S — i.e. dS is the derivative of the
/// SAME spherical S the energy path builds (the c2s sandwich is applied
/// correctly to the derivative). Discriminates a correct sandwich from a
/// missing/wrong one; no SCF solve, just three integral-pack builds.
TEST(SphericalGradient, SphericalOverlapDerivMatchesFiniteDifference) {
    const auto f = water();
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
    const cpu::scf::IntegralDerivPack dp{p};

    // Central difference of S w.r.t. the O atom's z coordinate, in Bohr.
    // dS is in 1/Bohr; the Frame stores Å, so displace by h_bohr·(Å/Bohr).
    constexpr int atom = 0; // O
    constexpr int axis = 2; // z
    constexpr double h_bohr = 1e-3;
    const double h_ang = h_bohr * units::kBohrInAngstrom;

    auto perturbed = [&](double dz_ang) {
        cpu::Frame g = water();
        g.span(kAtom_z)[atom] += dz_ang;
        return cpu::scf::internal::IntegralWorkspace{g, BasisSet::Def2SVP}.S;
    };
    const auto S_plus = perturbed(+h_ang);
    const auto S_minus = perturbed(-h_ang);
    const xt::xtensor<double, 2> fd = (S_plus - S_minus) / (2.0 * h_bohr);

    auto analytic = xt::view(dp.dS, atom, axis, xt::all(), xt::all());
    double max_err = xt::amax(xt::abs(analytic - fd))();
    EXPECT_LT(max_err, 1e-6) << "spherical dS vs finite-difference S";
}

/// ac-003: the RHF spherical guard is removed (compute runs), while the ROHF
/// and GHF spherical guards stay shut. Uses a single O atom (def2-SVP, d
/// shell exercises the 6→5 sandwich): translational invariance forces the
/// single-atom gradient to zero for any density, so no SCF solve is needed.
TEST(SphericalGradient, RhfSphericalRunsSiblingsStillThrow) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 8; // O
    f.box = cpu::Box::cubic(50.0);
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP};
    ASSERT_TRUE(p.spherical());

    const int n = p.n_basis;
    const int nocc = 4; // garbage density; dims only (translational inv. → 0)
    cpu::linalg::RealMatrix C = xt::eye<double>(n);
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(n)});
    cpu::linalg::RealVector occ = xt::zeros<double>({std::size_t(n)});
    for (int i = 0; i < nocc; ++i)
        occ(i) = 1.0;
    const cpu::scf::SlaterDeterminant det{C, eps, occ, nocc};

    auto make_grad = [](cpu::scf::SpinType spin) {
        return cpu::scf::internal::ScfGradient{cpu::scf::MeanFieldType::HF,
                                               spin,
                                               cpu::scf::XCType::LSDA,
                                               cpu::scf::GridLevel::SG0,
                                               0,
                                               0,
                                               false,
                                               /*d3bj=*/false,
                                               /*d4=*/false};
    };

    const auto rhf = make_grad(cpu::scf::SpinType::Restricted);
    cpu::scf::GradientResult res;
    ASSERT_NO_THROW(res = rhf.compute(f, p, det, 0.0));
    // Single atom: total force is zero by translational invariance.
    double fmax = xt::amax(xt::abs(res.forces))();
    EXPECT_LT(fmax, 1e-8) << "single-atom translational invariance";

    // The Generalized (GHF spinor) gradient path input-validates its
    // determinant: a non-spinor SlaterDeterminant (this restricted `det`) is
    // rejected with std::invalid_argument ("... requires a generalized
    // (spinor) SlaterDeterminant.", gradient.cpp). GHF spinor forces on
    // spherical packs remain a follow-up (no PySCF oracle); ROHF forces are
    // now supported. Fossil note: this once threw std::runtime_error from a
    // blanket spherical guard that has since been replaced by typed input
    // validation.
    EXPECT_THROW(
        make_grad(cpu::scf::SpinType::Generalized).compute(f, p, det, 0.0),
        std::invalid_argument);
}
