/// @file tests/cpu/test_evaluate_driver_capabilities.cpp
/// @brief Evaluate accepts energy/force Potential sources; MD requires Integrator.
#include <gtest/gtest.h>

#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubIntegrator;

namespace {

struct EnergyOnlyPotential : ATV::cpu::Potential {
    int launch_count = 0;
    std::string name() const override { return "energy_only"; }
    void launch(const ATV::cpu::LaunchContext&) override { ++launch_count; }
};

} // namespace

TEST(EvaluateCapabilities, EnergyOnlyAcceptedByEvaluate) {
    using namespace ATV;
    cpu::Evaluate e;
    EnergyOnlyPotential energy;
    e.on(evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(e.build());

    auto frame = make_trivial_frame();
    e.run(frame);
    EXPECT_EQ(energy.launch_count, 1);
}

TEST(EvaluateCapabilities, MDStillNeedsIntegrator) {
    using namespace ATV;
    cpu::md::MD md;
    EnergyOnlyPotential energy;
    md.on(md::Phase::force).add_ref(energy);
    // No integrator → build fails.
    EXPECT_THROW(md.build(), std::exception);
}
