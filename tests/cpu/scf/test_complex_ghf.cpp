/// @file tests/cpu/scf/test_complex_ghf.cpp
/// @brief Unit: sigma_dot, int1e_spnucsp scalar vs spin-free pVp, and the
///        X2C-1e spin-orbital hcore (no SCF).
///
/// H2O hcore pins are PySCF 2.14.0 `GHF.x2c1e` with `xuncontract=False`
/// (`SpinOrbitalX2CHelper.get_hcore` / `_x2c1e_get_hcore`).

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/complex_ghf.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/complex_ghf.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/x2c.h"

using namespace ATV;

namespace {

cpu::Frame h2() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.box = cpu::Box::cubic(30.0);
    return f;
}

cpu::Frame h2o() {
    constexpr double kPi = 3.14159265358979323846;
    const double r_oh = 0.95;
    const double half = 0.5 * 104.5 * kPi / 180.0;
    const double hx = r_oh * std::sin(half);
    const double hz = r_oh * std::cos(half);
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = hx;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = hz;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -hx;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[2] = hz;
    f.box = cpu::Box::cubic(30.0);
    return f;
}

cpu::scf::internal::ComplexMatrix hcore_x2c(const cpu::Frame& frame) {
    using namespace cpu::scf;
    internal::IntegralWorkspace pack{frame,
                                     BasisSet::STO3G,
                                     0,
                                     /*compute_eri=*/false,
                                     {},
                                     {},
                                     /*compute_pvp=*/true};
    AtomList atoms{frame, 0};
    ShellList shells{atoms, BasisSet::STO3G};
    const internal::Spnucsp q = internal::int1e_spnucsp(shells, atoms);
    const internal::ComplexMatrix t = internal::block_diag_real(pack.T);
    const internal::ComplexMatrix v = internal::block_diag_real(pack.V);
    const internal::ComplexMatrix s = internal::block_diag_real(pack.S);
    const internal::ComplexMatrix w = internal::sigma_dot(q.x, q.y, q.z, q.s);
    return internal::x2c1e_get_hcore(t, v, w, s, internal::kLightSpeedAU);
}

} // namespace

TEST(ComplexGhf, SigmaDotPauliBlocks) {
    using cpu::scf::internal::sigma_dot;
    // Antisymmetric vector part: a symmetric 1x1 is not a cross product
    // and is removed by the Hermitian projection.
    xt::xtensor<double, 2> z = xt::zeros<double>({2, 2});
    xt::xtensor<double, 2> s = xt::zeros<double>({2, 2});
    s(0, 0) = 1.0;
    s(1, 1) = 2.0;
    xt::xtensor<double, 2> a = xt::zeros<double>({2, 2});
    a(0, 1) = 1.0;
    a(1, 0) = -1.0;
    const auto Ws = sigma_dot(z, z, z, s);
    EXPECT_NEAR(Ws(0, 0).real(), 1.0, 0.0);
    EXPECT_NEAR(Ws(3, 3).real(), 2.0, 0.0);
    EXPECT_NEAR(std::abs(Ws(0, 2)), 0.0, 0.0);
    const auto Wz = sigma_dot(z, z, a, z);
    // α0,α1 = i * a_01 = i ; α1,α0 = conj.
    EXPECT_NEAR(Wz(0, 1).imag(), 1.0, 1e-15);
    EXPECT_NEAR(Wz(1, 0).imag(), -1.0, 1e-15);
    EXPECT_NEAR(Wz(0, 0).imag(), 0.0, 1e-15);
    const auto Wx = sigma_dot(a, z, z, z);
    // α0,β1 = i * a_01 = i ; βα partner is the conjugate.
    EXPECT_NEAR(Wx(0, 3).imag(), 1.0, 1e-15);
    EXPECT_NEAR(Wx(3, 0).imag(), -1.0, 1e-15);
    const auto Wy = sigma_dot(z, a, z, z);
    EXPECT_NEAR(Wy(0, 3).real(), 1.0, 1e-15);
    EXPECT_NEAR(Wy(3, 0).real(), 1.0, 1e-15);
}

TEST(ComplexGhf, SpnucspScalarMatchesPvp) {
    const cpu::Frame frame = h2();
    cpu::scf::internal::IntegralWorkspace pack{
        frame, cpu::scf::BasisSet::STO3G, 0, false, {}, {}, true};
    cpu::scf::AtomList atoms{frame, 0};
    cpu::scf::ShellList shells{atoms, cpu::scf::BasisSet::STO3G};
    const auto q = cpu::scf::internal::int1e_spnucsp(shells, atoms);
    ASSERT_EQ(q.s.shape(0), pack.W.shape(0));
    double max = 0.0;
    for (int i = 0; i < static_cast<int>(q.s.shape(0)); ++i)
        for (int j = 0; j < static_cast<int>(q.s.shape(1)); ++j)
            max = std::max(max, std::abs(q.s(i, j) - pack.W(i, j)));
    EXPECT_LT(max, 1e-10);
}

TEST(ComplexGhf, H2oX2cHcoreMatchesPyscf) {
    // PySCF 2.14.0 GHF(mol).x2c1e(); with_x2c.xuncontract = False;
    // H2O STO-3G, r(OH)=0.95 Å, angle 104.5°.
    const auto h = hcore_x2c(h2o());
    constexpr double kH00 = -32.74758842944706;
    constexpr double kImag = 0.0003786885826683635;
    constexpr double kFrob = 56.605333666417344;
    EXPECT_NEAR(h(0, 0).real(), kH00, 2e-6);
    EXPECT_NEAR(h(0, 0).imag(), 0.0, 1e-12);
    double imag = 0.0;
    double frob2 = 0.0;
    const int n = static_cast<int>(h.shape(0));
    EXPECT_EQ(n, 14);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            imag = std::max(imag, std::abs(h(i, j).imag()));
            frob2 += std::norm(h(i, j));
        }
    }
    EXPECT_NEAR(imag, kImag, 1e-8);
    EXPECT_NEAR(std::sqrt(frob2), kFrob, 2e-6);
}

TEST(ComplexGhf, ForcesThrowNamedKernel) {
    cpu::Frame frame = h2();
    cpu::scf::ComplexGHF ghf{cpu::scf::BasisSet::STO3G};
    try {
        ghf.forces(frame);
        FAIL() << "forces returned";
    } catch (const std::runtime_error& err) {
        const std::string msg = err.what();
        EXPECT_NE(msg.find("complex_ghf_nuclear_forces"), std::string::npos);
    }
}

// Driver shape only. The SCF solve stays in regressions/scf/ghf.
TEST(ComplexGhfDriver, PresetIsScfDriver) {
    static_assert(
        std::is_base_of_v<cpu::Driver<ATV::scf::Phase>, cpu::scf::ComplexGHF>,
        "cpu::scf::ComplexGHF must be cpu::Driver<scf::Phase>");
    static_assert(std::is_base_of_v<cpu::DriverNode, cpu::scf::ComplexGHF>,
                  "cpu::scf::ComplexGHF must be a nestable DriverNode");
    static_assert(std::is_base_of_v<cpu::HF, cpu::scf::ComplexGHF>,
                  "cpu::scf::ComplexGHF must carry the HF tag");
    static_assert(!std::is_base_of_v<cpu::Component, cpu::scf::ComplexGHF>,
                  "cpu::scf::ComplexGHF must not be a Component");

    cpu::scf::ComplexGHF ghf{cpu::scf::BasisSet::STO3G, 0, 1e-10, 80, true};
    EXPECT_TRUE(ghf.provides_energy());
    EXPECT_FALSE(ghf.provides_force());
    EXPECT_TRUE(ghf.requires_full_frame());
    EXPECT_EQ(ghf.default_budget(), ghf.max_iter);
    EXPECT_EQ(ghf.charge, 0);
    EXPECT_DOUBLE_EQ(ghf.tol, 1e-10);
    EXPECT_TRUE(ghf.x2c);

    const auto& plan = ghf.plan();
    EXPECT_EQ(plan.kind, PhaseKind::SCF);
    EXPECT_TRUE(plan.requires_full_frame);
    ASSERT_EQ(plan.slots.size(),
              static_cast<std::size_t>(ATV::slot_count<ATV::scf::Phase>()));
    const char* names[] = {"complex_ghf_fock", "complex_ghf_mix",
                           "complex_ghf_diagonalize", "complex_ghf_density",
                           "complex_ghf_check"};
    const ATV::scf::Phase phases[] = {
        ATV::scf::Phase::fock, ATV::scf::Phase::mix,
        ATV::scf::Phase::diagonalize, ATV::scf::Phase::density,
        ATV::scf::Phase::check};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(plan.slots[static_cast<std::size_t>(i)].phase,
                  ATV::slot_of(phases[i]));
        ASSERT_EQ(plan.slots[static_cast<std::size_t>(i)].ops.size(), 1u);
        const auto* op =
            plan.slots[static_cast<std::size_t>(i)].ops[0].component;
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(op->name(), names[i]);
    }
    ASSERT_EQ(plan.lifecycle_only.size(), 1u);
    const auto* crit =
        dynamic_cast<const cpu::Criterion*>(plan.lifecycle_only[0]);
    ASSERT_NE(crit, nullptr);
    EXPECT_EQ(crit->name(), "complex_ghf_energy_delta");
    EXPECT_TRUE(crit->contributes_to_convergence());
}
