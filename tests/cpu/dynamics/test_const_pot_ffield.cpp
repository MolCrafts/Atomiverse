/// @file tests/cpu/dynamics/test_const_pot_ffield.cpp
/// @brief Unit tests for ConstPotBO finite-field RHS (single methods).

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Vacuum single electrode site at (0,0,z).
cpu::Frame vacuum_site(double z) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = z;
    f.span(kAtom_charge)[0] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 196.97;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

void run_once(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

double A_self(double eta) {
    constexpr double kSqrt2OverPi = 0.79788456080286535588;
    return units::coulomb * eta * kSqrt2OverPi;
}

} // namespace

TEST(ConstPotBOFfield, VacuumSingleSiteMatchesAnalytic) {
    constexpr double eta = 2.0;
    constexpr double Ez = 0.1; // V/Å
    constexpr double z = 10.0; // Å
    constexpr double Psi = 0.0;
    const double Aii = A_self(eta);
    const double Q_pin = (Psi + Ez * z) / Aii; // AQ = Ψ + E·r

    auto f = vacuum_site(z);
    cpu::md::ConstPotBO conp(eta);
    conp.add_electrode(1, Psi);
    conp.Ez = Ez;
    conp.electroneutrality = false;
    run_once(conp, f);

    EXPECT_NEAR(conp.total_charge(0), Q_pin, 1e-12);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_charge)[0]), Q_pin, 1e-12);
}

TEST(ConstPotBOFfield, ZeroFieldLeavesVacuumQZeroAtZeroPsi) {
    auto f = vacuum_site(10.0);
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0);
    conp.electroneutrality = false;
    // Ex=Ey=Ez default 0
    run_once(conp, f);
    EXPECT_NEAR(conp.total_charge(0), 0.0, 1e-14);
}

TEST(ConstPotBOFfield, FieldSignFlipsCharge) {
    constexpr double eta = 2.0;
    constexpr double z = 10.0;
    const double Aii = A_self(eta);
    auto f_pos = vacuum_site(z);
    cpu::md::ConstPotBO cpos(eta);
    cpos.add_electrode(1, 0.0);
    cpos.Ez = 0.1;
    cpos.electroneutrality = false;
    run_once(cpos, f_pos);

    auto f_neg = vacuum_site(z);
    cpu::md::ConstPotBO cneg(eta);
    cneg.add_electrode(1, 0.0);
    cneg.Ez = -0.1;
    cneg.electroneutrality = false;
    run_once(cneg, f_neg);

    EXPECT_NEAR(cpos.total_charge(0), -cneg.total_charge(0), 1e-12);
    EXPECT_NEAR(cpos.total_charge(0), (0.1 * z) / Aii, 1e-12);
}

TEST(ConstPotBOFfield, ElectrolyteEntersB) {
    // Electrode at z=10, ion q=+1 at z=15; vacuum formula plus Gaussian b.
    constexpr double eta = 2.0;
    constexpr double Ez = 0.1;
    constexpr double z_e = 10.0;
    constexpr double z_i = 15.0;
    constexpr double q_i = 1.0;
    const double r = std::abs(z_e - z_i);
    const double b = units::coulomb * q_i * std::erf(eta * r) / r;
    const double Aii = A_self(eta);
    const double Q_pin = (0.0 - b + Ez * z_e) / Aii;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[0] = f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = z_e;
    f.span(kAtom_z)[1] = z_i;
    f.span(kAtom_charge)[0] = 0.0;
    f.span(kAtom_charge)[1] = q_i;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_mass)[0] = 196.97;
    f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);

    cpu::md::ConstPotBO conp(eta);
    conp.add_electrode(1, 0.0);
    conp.Ez = Ez;
    conp.electroneutrality = false;
    run_once(conp, f);
    EXPECT_NEAR(conp.total_charge(0), Q_pin, 1e-10);
    // Electrolyte charge must not be overwritten.
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[1]), q_i);
}
