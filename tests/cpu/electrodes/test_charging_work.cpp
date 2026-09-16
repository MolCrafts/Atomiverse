/// @file tests/cpu/electrodes/test_charging_work.cpp
/// @brief Unit tests for cpu::electrodes::ChargingWork (E1 charging work).
///
/// Pins MetalWalls JOSS eq. (6) second term \(E = -\sum \Psi_i Q_i\) on a
/// fixed parallel-plate charge assignment (no e2e MD, no external engine).

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/electrodes/charging_work.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Parallel-plate toy: 4 left (type 1, Q=+0.1, Ψ=+0.5 V) + 4 right
/// (type 2, Q=-0.1, Ψ=-0.5 V) → E = -0.4 eV.
cpu::Frame make_parallel_plate() {
    cpu::Frame f(8);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 0.0;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_charge)[i] = 0.10;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    for (int i = 4; i < 8; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i - 4);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 10.0;
        f.span(kAtom_type)[i] = 2;
        f.span(kAtom_charge)[i] = -0.10;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(ChargingWork, ParallelPlatePinMinusSumPsiQ) {
    cpu::Frame f = make_parallel_plate();
    cpu::Context ctx;
    f.meta.get(kEnergy) = 0.0;
    f.meta.get(kEnergyChargingWork) = 0.0;

    cpu::electrodes::ChargingWork work(/*types=*/{1, 2},
                                       /*psi_V=*/{0.5, -0.5});
    work.build({ctx, f});
    work.launch({ctx, f, 0, 0});

    const double pin = -0.4; // eV — MetalWalls JOSS eq. 6 / Scalfi eq. 12
    EXPECT_NEAR(f.meta.get(kEnergy), pin, 1e-12);
    EXPECT_NEAR(f.meta.get(kEnergyChargingWork), pin, 1e-12);
    EXPECT_NEAR(work.last_work(), pin, 1e-12);
}

TEST(ChargingWork, NonElectrodeTypesIgnored) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_type)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(10.0);
    f.meta.get(kEnergy) = 0.0;
    f.meta.get(kEnergyChargingWork) = 0.0;

    cpu::Context ctx;
    cpu::electrodes::ChargingWork work({1, 2}, {1.0, -1.0});
    work.build({ctx, f});
    work.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-15);
    EXPECT_NEAR(work.last_work(), 0.0, 1e-15);
}

TEST(ChargingWork, LengthMismatchThrows) {
    EXPECT_THROW(cpu::electrodes::ChargingWork({1, 2}, {0.5}),
                 std::invalid_argument);
}

TEST(ChargingWork, DuplicateTypeThrows) {
    EXPECT_THROW(cpu::electrodes::ChargingWork({1, 1}, {0.5, -0.5}),
                 std::invalid_argument);
}

TEST(ChargingWork, ZeroEnergyClearsChargingWorkMeta) {
    cpu::Frame f = make_parallel_plate();
    cpu::Context ctx;
    f.meta.get(kEnergy) = 1.0;
    f.meta.get(kEnergyChargingWork) = 1.0;

    cpu::ZeroEnergy zero;
    zero.launch({ctx, f, 0, 0});
    EXPECT_EQ(f.meta.get(kEnergy), 0.0);
    EXPECT_EQ(f.meta.get(kEnergyChargingWork), 0.0);
}

TEST(ChargingWork, ScalfiFormQLDeltaPsi) {
    // Scalfi eq. (12): q^T Ψ = Q_L Δψ with Δψ = Ψ_L − Ψ_R.
    // E_charge = −q^T Ψ = −Q_L Δψ.
    cpu::Frame f = make_parallel_plate();
    const double Q_L = 4 * 0.10;
    const double delta_psi = 0.5 - (-0.5);
    const double pin = -Q_L * delta_psi;

    cpu::Context ctx;
    f.meta.get(kEnergy) = 0.0;
    f.meta.get(kEnergyChargingWork) = 0.0;
    cpu::electrodes::ChargingWork work({1, 2}, {0.5, -0.5});
    work.build({ctx, f});
    work.launch({ctx, f, 0, 0});
    EXPECT_NEAR(work.last_work(), pin, 1e-12);
    EXPECT_NEAR(pin, -0.4, 1e-15);
}
