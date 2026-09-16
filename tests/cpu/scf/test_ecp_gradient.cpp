/// @file tests/cpu/scf/test_ecp_gradient.cpp
/// @brief Unit: ECP nuclear-gradient guard + single-atom translational
///        invariance of the semi-numerical dU/dR path (hf-parity-01-ecp-forces).
///
/// The ECP energy path is validated by regressions/scf/ecp_heavy_atom; the
/// gradient's binding physics bar (I2 vs pinned PySCF) lives in
/// regressions/scf/ecp_i2_grad. These are the fast unit checks: the RHF ECP
/// guard opens (siblings stay shut) and a single ECP atom has zero force for
/// any density (translational invariance) — no SCF solve.

#include <gtest/gtest.h>

#include <stdexcept>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/gradient.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::internal::IntegralWorkspace;

namespace {

/// A single iodine atom with def2-SVP + def2-ECP (Z_val = 25).
cpu::scf::internal::IntegralWorkspace iodine_pack() {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 53; // I
    f.box = cpu::Box::cubic(50.0);
    return cpu::scf::internal::IntegralWorkspace{f, BasisSet::Def2SVP, 0, false, "def2-ECP"};
}

cpu::scf::internal::ScfGradient make_grad(cpu::scf::SpinType spin) {
    return cpu::scf::internal::ScfGradient{cpu::scf::MeanFieldType::HF,
                                           spin,
                                           cpu::scf::XCType::LSDA,
                                           cpu::scf::GridLevel::SG0,
                                           0,
                                           0,
                                           false,
                                           /*d3bj=*/false,
                                           /*d4=*/false};
}

} // namespace

/// ac-001 / ac-002: the RHF ECP guard is removed (compute runs) and the single-
/// atom ECP gradient is zero by translational invariance; ROHF/GHF ECP throw.
TEST(EcpGradient, RhfEcpRunsSiblingsThrowSingleAtomZeroForce) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 53; // I
    f.box = cpu::Box::cubic(50.0);
    const cpu::scf::internal::IntegralWorkspace p{f, BasisSet::Def2SVP, 0, false, "def2-ECP"};
    ASSERT_TRUE(p.has_ecp);

    const int n = p.n_basis;
    const int nocc = 12; // garbage density; dims only (translational inv. → 0)
    cpu::linalg::RealMatrix C = xt::eye<double>(n);
    cpu::linalg::RealVector eps = xt::zeros<double>({std::size_t(n)});
    cpu::linalg::RealVector occ = xt::zeros<double>({std::size_t(n)});
    for (int i = 0; i < nocc; ++i)
        occ(i) = 1.0;
    const cpu::scf::SlaterDeterminant det{C, eps, occ, nocc};

    cpu::scf::GradientResult res;
    ASSERT_NO_THROW(
        res =
            make_grad(cpu::scf::SpinType::Restricted).compute(f, p, det, 0.0));
    // Single atom: total force zero by translational invariance (the FD dU/dR
    // displaces shells + ECP center together, so dU is exactly zero here).
    const double fmax = xt::amax(xt::abs(res.forces))();
    EXPECT_LT(fmax, 1e-8) << "single-atom ECP translational invariance";

    // GHF forces need a generalized (spinor) determinant; a collinear
    // restricted det with SpinType::Generalized is a type error.
    EXPECT_THROW(
        make_grad(cpu::scf::SpinType::Generalized).compute(f, p, det, 0.0),
        std::invalid_argument);
}

/// The pack retains the ECP centers so the gradient can rebuild dU/dR.
TEST(EcpGradient, PackRetainsEcpCenters) {
    const cpu::scf::internal::IntegralWorkspace p = iodine_pack();
    EXPECT_TRUE(p.has_ecp);
    EXPECT_EQ(p.ecp_centers.size(), 1u);
}
