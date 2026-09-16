/// @file tests/cpu/observe/test_capacitance_charge_fluctuation.cpp
/// @brief Unit: CapacitanceChargeFluctuation moments and C_diff / C_int.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/capacitance_charge_fluctuation.h"

using namespace ATV;
using cpu::CapacitanceChargeFluctuation;

namespace {

/// Parallel-plate toy: 4 left (type 1) + 4 right (type 2); per-atom charges set by caller.
cpu::Frame parallel_plate_shell() {
    cpu::Frame f(8);
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 0.0;
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    for (int i = 4; i < 8; ++i) {
        f.span(kAtom_x)[i] = static_cast<double>(i - 4);
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = 10.0;
        f.span(kAtom_type)[i] = 2;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

void set_wall_total_Q(cpu::Frame& f, double QL, double QR) {
    // 4 atoms per wall → equal share.
    const double qL = QL / 4.0;
    const double qR = QR / 4.0;
    for (int i = 0; i < 4; ++i)
        f.span(kAtom_charge)[i] = qL;
    for (int i = 4; i < 8; ++i)
        f.span(kAtom_charge)[i] = qR;
}

} // namespace

TEST(CapacitanceChargeFluctuation, KnownSeriesDiffAndIntegral) {
    // Pin series (REFERENCE.md / literature formula):
    // Q_L samples: {0.4, 0.6, 0.4, 0.6} e at T=300 K, ΔΨ=1 V
    // ⟨Q⟩=0.5, Var=0.01, C_diff=0.01/(kB·300), C_int=0.5
    constexpr double T = 300.0;
    constexpr double dPsi = 1.0;
    const double Q_samples[] = {0.4, 0.6, 0.4, 0.6};

    auto frame = parallel_plate_shell();
    CapacitanceChargeFluctuation obs({1, 2}, T, dPsi);
    cpu::Context ctx;

    for (double QL : Q_samples) {
        set_wall_total_Q(frame, QL, -QL);
        cpu::LaunchContext lc{ctx, frame, 0, 0};
        obs.launch(lc);
    }

    EXPECT_EQ(obs.n_samples(), 4u);
    EXPECT_NEAR(obs.mean_charge(), 0.5, 1e-15);
    EXPECT_NEAR(obs.charge_variance(), 0.01, 1e-15);

    const double kT = units::kB_ev_per_k * T;
    const double c_diff_pin = 0.01 / kT;
    EXPECT_NEAR(obs.differential_capacitance(), c_diff_pin, 1e-12);
    EXPECT_NEAR(obs.integral_capacitance(), 0.5, 1e-15);

    ASSERT_TRUE(frame.meta.contains(kCapacitanceDiff.key));
    EXPECT_NEAR(frame.meta.get(kCapacitanceDiff), c_diff_pin, 1e-12);
    ASSERT_TRUE(frame.meta.contains(kCapacitanceIntegral.key));
    EXPECT_NEAR(frame.meta.get(kCapacitanceIntegral), 0.5, 1e-15);

    // Last sample Q_L=+0.6, Q_R=-0.6
    EXPECT_NEAR(obs.total_charge(0), 0.6, 1e-15);
    EXPECT_NEAR(obs.total_charge(1), -0.6, 1e-15);
}

TEST(CapacitanceChargeFluctuation, ZeroDeltaPsiIntegralNaN) {
    auto frame = parallel_plate_shell();
    set_wall_total_Q(frame, 0.5, -0.5);
    CapacitanceChargeFluctuation obs({1, 2}, 300.0, /*delta_psi_V=*/0.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    EXPECT_TRUE(std::isnan(obs.integral_capacitance()));
    EXPECT_FALSE(frame.meta.contains(kCapacitanceIntegral.key));
}

TEST(CapacitanceChargeFluctuation, ResetClearsMoments) {
    auto frame = parallel_plate_shell();
    set_wall_total_Q(frame, 1.0, -1.0);
    CapacitanceChargeFluctuation obs({1}, 300.0, 1.0);
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    EXPECT_EQ(obs.n_samples(), 1u);
    obs.reset();
    EXPECT_EQ(obs.n_samples(), 0u);
    EXPECT_NEAR(obs.mean_charge(), 0.0, 1e-15);
    EXPECT_NEAR(obs.charge_variance(), 0.0, 1e-15);
}

TEST(CapacitanceChargeFluctuation, EmptyTypesThrows) {
    EXPECT_THROW(CapacitanceChargeFluctuation(std::vector<int>{}, 300.0),
                 std::invalid_argument);
}

TEST(CapacitanceChargeFluctuation, NonPositiveTemperatureThrows) {
    EXPECT_THROW(CapacitanceChargeFluctuation({1}, 0.0), std::invalid_argument);
    EXPECT_THROW(CapacitanceChargeFluctuation({1}, -10.0),
                 std::invalid_argument);
}
