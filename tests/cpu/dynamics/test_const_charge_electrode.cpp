/// @file tests/cpu/dynamics/test_const_charge_electrode.cpp
/// @brief Unit tests for the constant-charge (conq) mode of ConstPotBO.

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

/// Two one-atom electrodes in vacuum (no electrolyte, big box, R=0 path).
cpu::Frame two_site_vacuum(double sep) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = i * sep;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = i + 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(100.0, 100.0, 100.0);
    return f;
}

/// 2x2 sites per plate + asymmetric electrolyte pair (10 atoms total).
cpu::Frame multi_site_plates() {
    cpu::Frame f(10);
    int n = 0;
    auto put = [&](double x, double y, double z, int type, double q) {
        f.span(kAtom_x)[n] = x;
        f.span(kAtom_y)[n] = y;
        f.span(kAtom_z)[n] = z;
        f.span(kAtom_charge)[n] = q;
        f.span(kAtom_type)[n] = type;
        f.span(kAtom_mass)[n] = 1.0;
        f.span(kAtom_tag)[n] = n + 1;
        ++n;
    };
    for (double x : {0.0, 1.0})
        for (double y : {0.0, 1.0})
            put(x, y, 8.0, 1, 0.0); // bottom plate
    for (double x : {0.0, 1.0})
        for (double y : {0.0, 1.0})
            put(x, y, 12.0, 2, 0.0); // top plate
    put(0.3, 0.2, 9.5, 3, 0.4);      // asymmetric electrolyte +
    put(0.7, 1.1, 10.5, 3, -0.4);    // asymmetric electrolyte −
    f.box = cpu::Box::orthorhombic(2.0, 2.0, 20.0);
    return f;
}

void run_once(cpu::md::ConstPotBO& c, cpu::Frame& f) {
    cpu::Context ctx;
    c.build({ctx, f});
    c.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ConstChargeElectrode, RegistrationChainsAndPotentialFallback) {
    cpu::md::ConstPotBO conq = cpu::md::ConstPotBO(2.0)
                                   .add_electrode_charge(1, -0.5)
                                   .add_electrode_charge(2, 0.5);
    EXPECT_EQ(conq.name(), "const_pot_bo");
    // Before any solve, const-Q groups report a zero (unsolved) potential.
    EXPECT_DOUBLE_EQ(conq.electrode_potential(0), 0.0);
    EXPECT_DOUBLE_EQ(conq.electrode_potential(1), 0.0);
    EXPECT_DOUBLE_EQ(conq.delta_psi(), 0.0);
    EXPECT_THROW(conq.electrode_potential(2), std::out_of_range);
}

TEST(ConstChargeElectrode, GroupTotalsMatchTargetsExactly) {
    auto f = two_site_vacuum(5.0);
    cpu::md::ConstPotBO conq(2.0);
    conq.add_electrode_charge(1, -0.25);
    conq.add_electrode_charge(2, 0.25);
    run_once(conq, f);
    EXPECT_NEAR(conq.total_charge(0), -0.25, 1e-13);
    EXPECT_NEAR(conq.total_charge(1), 0.25, 1e-13);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]), -0.25, 1e-13);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[1]), 0.25, 1e-13);
    EXPECT_LE(conq.solve_residual(), 1e-10);
}

TEST(ConstChargeElectrode, SolvedPotentialMatchesAnalyticTwoSite) {
    // Vacuum, one atom per electrode: ψ = A Q with
    // A_ii = k η √(2/π), A_01 = k erf(η/√2 · r)/r (equal η product width).
    constexpr double kEta = 2.0;
    constexpr double kSep = 5.0;
    constexpr double kQ = 0.25;
    const double ke = units::coulomb;
    const double a_self = ke * kEta * std::sqrt(2.0 / 3.14159265358979323846);
    const double a01 = ke * std::erf(kEta / std::sqrt(2.0) * kSep) / kSep;
    const double psi_bot = a_self * (-kQ) + a01 * kQ;
    const double psi_top = a01 * (-kQ) + a_self * kQ;

    auto f = two_site_vacuum(kSep);
    cpu::md::ConstPotBO conq(kEta);
    conq.add_electrode_charge(1, -kQ);
    conq.add_electrode_charge(2, kQ);
    run_once(conq, f);
    EXPECT_NEAR(conq.electrode_potential(0), psi_bot, 1e-12);
    EXPECT_NEAR(conq.electrode_potential(1), psi_top, 1e-12);
    EXPECT_NEAR(conq.delta_psi(), psi_top - psi_bot, 1e-12);
    EXPECT_GT(conq.delta_psi(), 0.0); // +Q on top ⇒ higher potential
}

TEST(ConstChargeElectrode, VResponseLinearInChargeVacuum) {
    // No electrolyte ⇒ ΔΨ(Q) = Q/C exactly linear in Q.
    auto f1 = two_site_vacuum(5.0);
    cpu::md::ConstPotBO c1(2.0);
    c1.add_electrode_charge(1, -0.1).add_electrode_charge(2, 0.1);
    run_once(c1, f1);
    auto f2 = two_site_vacuum(5.0);
    cpu::md::ConstPotBO c2(2.0);
    c2.add_electrode_charge(1, -0.2).add_electrode_charge(2, 0.2);
    run_once(c2, f2);
    EXPECT_NEAR(c2.delta_psi(), 2.0 * c1.delta_psi(), 1e-12);
    EXPECT_GT(c1.delta_psi(), 0.0);
}

TEST(ConstChargeElectrode, ConqDualityReproducesConpPerSite) {
    // conp at ΔΨ = 1 V (neutrality on) → totals Q_e*; conq fed Q_e* must
    // reproduce the SAME per-site distribution and ΔΨ = 1 V (linear-algebra
    // duality of the constrained BO solve).
    auto fp = multi_site_plates();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.add_electrode(2, 1.0);
    conp.electroneutrality = true;
    run_once(conp, fp);
    const double q_bot = conp.total_charge(0);
    const double q_top = conp.total_charge(1);

    auto fq = multi_site_plates();
    cpu::md::ConstPotBO conq(2.0);
    conq.add_electrode_charge(1, q_bot);
    conq.add_electrode_charge(2, q_top);
    run_once(conq, fq);

    EXPECT_NEAR(conq.total_charge(0), q_bot, 1e-10);
    EXPECT_NEAR(conq.total_charge(1), q_top, 1e-10);
    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(static_cast<double>(fq.span(kAtom_charge)[i]),
                    static_cast<double>(fp.span(kAtom_charge)[i]), 1e-8)
            << "site " << i;
    EXPECT_NEAR(conq.delta_psi(), 1.0, 1e-8);
}

TEST(ConstChargeElectrode, DistributionNotUniformOnAsymmetricElectrolyte) {
    auto f = multi_site_plates();
    cpu::md::ConstPotBO conq(2.0);
    conq.add_electrode_charge(1, -0.3);
    conq.add_electrode_charge(2, 0.3);
    run_once(conq, f);
    // Off-center electrolyte ⇒ per-site charges on a plate differ.
    double qmin = 1e300, qmax = -1e300;
    for (int i = 0; i < 4; ++i) {
        const double qi = static_cast<double>(f.span(kAtom_charge)[i]);
        qmin = std::min(qmin, qi);
        qmax = std::max(qmax, qi);
    }
    EXPECT_GT(qmax - qmin, 1e-4);
    EXPECT_NEAR(conq.total_charge(0), -0.3, 1e-12);
}

TEST(ConstChargeElectrode, CgMatchesDense) {
    auto fd = multi_site_plates();
    cpu::md::ConstPotBO dense(2.0);
    dense.add_electrode_charge(1, -0.3).add_electrode_charge(2, 0.3);
    run_once(dense, fd);

    auto fc = multi_site_plates();
    cpu::md::ConstPotBO cg(2.0);
    cg.add_electrode_charge(1, -0.3).add_electrode_charge(2, 0.3);
    cg.use_cg = true;
    cg.cg_tol = 1e-12;
    run_once(cg, fc);

    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(static_cast<double>(fc.span(kAtom_charge)[i]),
                    static_cast<double>(fd.span(kAtom_charge)[i]), 1e-8);
    EXPECT_NEAR(cg.delta_psi(), dense.delta_psi(), 1e-8);
    EXPECT_NEAR(cg.total_charge(1), 0.3, 1e-10);
}

TEST(ConstChargeElectrode, ElectrolyteChargesUntouched) {
    auto f = multi_site_plates();
    cpu::md::ConstPotBO conq(2.0);
    conq.add_electrode_charge(1, -0.3).add_electrode_charge(2, 0.3);
    run_once(conq, f);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[8]), 0.4);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[9]), -0.4);
}

TEST(ConstChargeElectrode, MixedConstPsiConstQGroups) {
    // Bottom plate held at Ψ = 0 (const-Ψ), top plate at fixed Q (const-Q):
    // the const-Q total is exact and the joint solve is consistent.
    auto f = multi_site_plates();
    cpu::md::ConstPotBO mixed(2.0);
    mixed.add_electrode(1, 0.0);
    mixed.add_electrode_charge(2, 0.2);
    run_once(mixed, f);
    EXPECT_NEAR(mixed.total_charge(1), 0.2, 1e-12);
    EXPECT_LE(mixed.solve_residual(), 1e-9);
    // Const-Ψ group still reports its applied potential.
    EXPECT_DOUBLE_EQ(mixed.electrode_potential(0), 0.0);
}

TEST(ConstChargeElectrode, RepeatedLaunchIsStationary) {
    // Static geometry + static electrolyte: a second launch must reproduce
    // the same charges and solved potentials (base Ψ reset each launch).
    auto f = multi_site_plates();
    cpu::md::ConstPotBO conq(2.0);
    conq.add_electrode_charge(1, -0.3).add_electrode_charge(2, 0.3);
    cpu::Context ctx;
    conq.build({ctx, f});
    conq.launch({ctx, f, 0, 0});
    const double dpsi1 = conq.delta_psi();
    const double q0_1 = static_cast<double>(f.span(kAtom_charge)[0]);
    conq.launch({ctx, f, 0, 1});
    EXPECT_NEAR(conq.delta_psi(), dpsi1, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]), q0_1, 1e-12);
    EXPECT_NEAR(conq.total_charge(1), 0.3, 1e-12);
}
