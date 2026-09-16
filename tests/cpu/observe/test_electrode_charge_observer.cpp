/// @file tests/cpu/observe/test_electrode_charge_observer.cpp
/// @brief Unit: ElectrodeChargeObserver sums kAtom_charge by electrode type.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/electrode_charge.h"

using namespace ATV;
using cpu::ElectrodeChargeObserver;

namespace {

/// Parallel-plate toy: 4 left (type 1, q=+0.125), 4 right (type 2, q=-0.125),
/// 2 electrolyte (type 3, q=1) that must not enter electrode totals.
cpu::Frame parallel_plate_toy() {
    cpu::Frame f(10);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto type = f.span(kAtom_type);
    auto q = f.span(kAtom_charge);
    auto mass = f.span(kAtom_mass);
    auto tag = f.span(kAtom_tag);

    const double left_x[4] = {2, 6, 10, 14};
    for (int i = 0; i < 4; ++i) {
        x[i] = left_x[i];
        y[i] = 2.0;
        z[i] = 1.0;
        type[i] = 1;
        q[i] = 0.125;
        mass[i] = 12.0;
        tag[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
        const int j = 4 + i;
        x[j] = left_x[i];
        y[j] = 2.0;
        z[j] = 9.0;
        type[j] = 2;
        q[j] = -0.125;
        mass[j] = 12.0;
        tag[j] = j + 1;
    }
    x[8] = 5.0;
    y[8] = 5.0;
    z[8] = 5.0;
    type[8] = 3;
    q[8] = 1.0;
    mass[8] = 23.0;
    tag[8] = 9;
    x[9] = 15.0;
    y[9] = 5.0;
    z[9] = 5.0;
    type[9] = 3;
    q[9] = 1.0;
    mass[9] = 23.0;
    tag[9] = 10;

    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(ElectrodeChargeObserver, ParallelPlateTotals) {
    auto frame = parallel_plate_toy();
    ElectrodeChargeObserver obs({1, 2});

    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);

    EXPECT_EQ(obs.n_electrodes(), 2u);
    EXPECT_NEAR(obs.total_charge(0), 0.5, 1e-12);
    EXPECT_NEAR(obs.total_charge(1), -0.5, 1e-12);

    ASSERT_TRUE(frame.meta.contains(kElectrodeCharge.key));
    const auto& pack = frame.meta.get(kElectrodeCharge);
    EXPECT_NEAR(pack[0], 0.5, 1e-12);
    EXPECT_NEAR(pack[1], -0.5, 1e-12);
    EXPECT_NEAR(pack[2], 0.0, 1e-15);
}

TEST(ElectrodeChargeObserver, ElectrolyteExcluded) {
    auto frame = parallel_plate_toy();
    // If type 3 were wrongly included as an electrode, sum would be ±0.5 + 2.0.
    ElectrodeChargeObserver obs({1, 2});
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    EXPECT_NEAR(obs.total_charge(0) + obs.total_charge(1), 0.0, 1e-12);
    EXPECT_NE(std::abs(obs.total_charge(0) + obs.total_charge(1) - 2.0), 0.0);
}

TEST(ElectrodeChargeObserver, EmptyTypesThrows) {
    EXPECT_THROW(ElectrodeChargeObserver(std::vector<int>{}),
                 std::invalid_argument);
}
