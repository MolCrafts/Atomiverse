/// @file tests/cpu/dynamics/test_const_pot_bo.cpp
/// @brief Unit tests for ConstPotBO electrode charge solve (single methods).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Madelung-like 4-atom geometry, primary cell only (no 2D images).
cpu::Frame madelung_primary() {
    cpu::Frame f(4);
    // bot electrode, top electrode, bot elyt, top elyt
    // Centered in Lz=20 so full 3D PBC cannot wrap the slab (see regression).
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.97 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return f;
}

void run_once(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ConstPotBO, NameAndNoConfigFactory) {
    cpu::md::ConstPotBO conp(2.0);
    EXPECT_EQ(conp.name(), "const_pot_bo");
    conp.add_electrode(1, 0.0).add_electrode(2, 1.0);
    EXPECT_DOUBLE_EQ(conp.delta_psi(), 1.0);
}

TEST(ConstPotBO, NeutralityPrimaryCell) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = 0;
    run_once(conp, f);
    EXPECT_LE(conp.neutrality_residual(), 1e-10);
    EXPECT_NEAR(conp.total_charge(0) + conp.total_charge(1), 0.0, 1e-10);
}

TEST(ConstPotBO, SignFollowsDeltaPsi) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.xy_image_range = 0;
    run_once(conp, f);
    // Positive ΔΨ (top − bot) ⇒ top electrode draws opposite electrolyte
    // polarization; with +0.5 near bot and −0.5 near top, Q_top > 0.
    EXPECT_GT(conp.total_charge(1), 0.0);
    EXPECT_LT(conp.total_charge(0), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[0]),
                     conp.total_charge(0));
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[1]),
                     conp.total_charge(1));
}

TEST(ConstPotBO, PrimaryCellMatchesAnalytic) {
    // Independent analytic: no images, eta=2, Ψ=(0,1), q_elyt=(0.5,-0.5)
    // Q from BO + neutrality (see agent pin script).
    constexpr double kQTop = 0.27175356;
    auto f = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.xy_image_range = 0;
    run_once(conp, f);
    EXPECT_NEAR(conp.total_charge(1), kQTop, 1e-6);
    EXPECT_NEAR(conp.total_charge(0), -kQTop, 1e-6);
}

TEST(ConstPotBO, ElectrolyteChargesUntouched) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0).add_electrode(2, 1.0);
    run_once(conp, f);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[2]), 0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[3]), -0.5);
}

TEST(ConstPotBO, MdBindingMatchesStandalone) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO standalone(2.0);
    standalone.add_electrode(1, 0.0);
    standalone.add_electrode(2, 1.0);
    standalone.xy_image_range = 50;
    run_once(standalone, f);
    const double q_ref = standalone.total_charge(1);

    auto f2 = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.xy_image_range = 50;

    // Direct Context launch after MD-style build should match (no MD).
    run_once(conp, f2);
    EXPECT_NEAR(conp.total_charge(1), q_ref, 1e-9);
}

TEST(ConstPotBO, ZeroLtfUnchangedPrimaryPin) {
    // A6: l_tf = 0 must match the perfect-metal path bit-for-bit even if
    // site_density is set (hardness only applies when l_tf > 0).
    auto f = madelung_primary();
    cpu::md::ConstPotBO perfect(2.0);
    perfect.add_electrode(1, 0.0);
    perfect.add_electrode(2, 1.0);
    perfect.xy_image_range = 0;
    run_once(perfect, f);
    const double q_ref = perfect.total_charge(1);

    auto f2 = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.xy_image_range = 0;
    conp.l_tf = 0.0;
    conp.site_density = 0.0593; // ignored when l_tf == 0
    run_once(conp, f2);
    EXPECT_NEAR(conp.total_charge(1), q_ref, 1e-12);
    EXPECT_NEAR(conp.total_charge(1), 0.27175356, 1e-6);
}

TEST(ConstPotBO, ThomasFermiReducesCharge) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO perfect(2.0);
    perfect.add_electrode(1, 0.0);
    perfect.add_electrode(2, 1.0);
    perfect.xy_image_range = 0;
    run_once(perfect, f);
    const double q0 = perfect.total_charge(1);

    auto f2 = madelung_primary();
    cpu::md::ConstPotBO tf(2.0);
    tf.add_electrode(1, 0.0);
    tf.add_electrode(2, 1.0);
    tf.xy_image_range = 0;
    tf.l_tf = 2.0;
    tf.site_density = 0.0593;
    run_once(tf, f2);
    const double q_tf = tf.total_charge(1);
    EXPECT_GT(q0, 0.0);
    EXPECT_GT(q_tf, 0.0);
    EXPECT_LT(q_tf, q0);
    EXPECT_LT(tf.capacitance(), perfect.capacitance());
    EXPECT_LE(tf.neutrality_residual(), 1e-10);
}

TEST(ConstPotBO, ThomasFermiRequiresDensity) {
    auto f = madelung_primary();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.l_tf = 1.0;
    conp.site_density = 0.0;
    cpu::Context ctx;
    EXPECT_THROW(conp.build({ctx, f}), std::invalid_argument);
}

TEST(ConstPotBO, ThomasFermiHardnessAnalyticTwoSite) {
    // Two electrode atoms, vacuum, no images, neutrality.
    // A = [[a0+h, a01], [a01, a0+h]], psi = (0, 1), b = 0.
    // With neutrality Q1 = -Q0 and solve for |Q|.
    constexpr double kEta = 2.0;
    constexpr double kLtf = 1.0;
    constexpr double kD = 0.1;
    constexpr double kSep = 5.0; // Å along z
    const double ke = units::coulomb;
    const double a0 = ke * kEta * std::sqrt(2.0 / 3.14159265358979323846);
    const double a01 = ke * std::erf(kEta / std::sqrt(2.0) * kSep) / kSep;
    const double h = 4.0 * 3.14159265358979323846 * ke * kLtf * kLtf * kD;
    // A Q = psi - lambda * e  with e^T Q = 0.
    // For 2 sites: Q = (q, -q). From (a0+h)q - a01 q = 0 - lambda
    // and -a01 q + (a0+h)(-q wait): second eq with psi=1:
    // actually neutrality projector form:
    // A_eff = [[a0+h, a01],[a01, a0+h]]
    // inv applied to (0,1) then project.
    const double det = (a0 + h) * (a0 + h) - a01 * a01;
    // inv = (1/det) [[a0+h, -a01], [-a01, a0+h]]
    // v = A^{-1} psi = (1/det) * (-a01, a0+h)
    // u = A^{-1} e = (1/det) * (a0+h-a01, a0+h-a01)
    // e·v = (1/det)*(-a01 + a0+h), e·u = 2(a0+h-a01)/det
    // scale = e·v / e·u
    // Q = v - scale * u
    const double v0 = -a01 / det;
    const double v1 = (a0 + h) / det;
    const double u0 = (a0 + h - a01) / det;
    const double u1 = u0;
    const double scale = (v0 + v1) / (u0 + u1);
    const double q1_analytic = v1 - scale * u1;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = kSep;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_charge)[0] = f.span(kAtom_charge)[1] = 0.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    // Large L so min-image is primary (R=0 path).
    f.box = cpu::Box::orthorhombic(100.0, 100.0, 100.0);

    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.xy_image_range = 0;
    conp.l_tf = kLtf;
    conp.site_density = kD;
    run_once(conp, f);
    EXPECT_NEAR(conp.total_charge(1), q1_analytic, 1e-10);
    EXPECT_NEAR(conp.total_charge(0), -q1_analytic, 1e-10);
}
