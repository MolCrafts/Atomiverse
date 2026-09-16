/// @file tests/cpu/observe/test_classical_polarization.cpp
/// @brief Unit tests for ClassicalPolarization (single methods).

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/classical_polarization.h"

using namespace ATV;

TEST(ClassicalPolarization, VacuumDipoleOverVolume) {
    // Two opposite charges on z-axis → P_z = (q z_+ + (-q) z_-)/V
    constexpr double q = 1.0;
    constexpr double z_p = 12.0;
    constexpr double z_m = 8.0;
    constexpr double L = 20.0;
    const double V = L * L * L;
    const double Pz_pin = (q * z_p + (-q) * z_m) / V;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[0] = f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = z_p;
    f.span(kAtom_z)[1] = z_m;
    f.span(kAtom_charge)[0] = q;
    f.span(kAtom_charge)[1] = -q;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(L);

    cpu::md::ClassicalPolarization pol;
    cpu::Context ctx;
    pol.launch({ctx, f, /*stage=*/0, /*step=*/0});

    EXPECT_NEAR(pol.Pz(), Pz_pin, 1e-14);
    EXPECT_NEAR(pol.Px(), 0.0, 1e-14);
    EXPECT_NEAR(pol.Py(), 0.0, 1e-14);
    EXPECT_NEAR(pol.dipole()[2], q * z_p - q * z_m, 1e-14);
}

TEST(ClassicalPolarization, Name) {
    cpu::md::ClassicalPolarization pol;
    EXPECT_EQ(pol.name(), "classical_polarization");
}
