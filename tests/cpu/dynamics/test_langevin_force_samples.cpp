/// @file tests/cpu/dynamics/test_langevin_force_samples.cpp
/// @brief Independent FixLangevin force samples at a fixed trajectory step.
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"

#include <array>

using namespace ATV;

namespace {

cpu::Frame atom() {
    cpu::Frame frame(1);
    frame.span(kAtom_tag)[0] = 29;
    frame.span(kAtom_mass)[0] = 12.0;
    frame.span(kAtom_vx)[0] = 0.0;
    frame.span(kAtom_vy)[0] = 0.0;
    frame.span(kAtom_vz)[0] = 0.0;
    return frame;
}

std::array<AccReal, 3> sample(cpu::md::FixLangevin& thermostat,
                              cpu::Context& context, cpu::Frame& frame) {
    frame.span(kAtom_fx)[0] = 0.0;
    frame.span(kAtom_fy)[0] = 0.0;
    frame.span(kAtom_fz)[0] = 0.0;
    thermostat.launch({context, frame, slot_of(md::Phase::post_force), 0});
    return {frame.span(kAtom_fx)[0], frame.span(kAtom_fy)[0],
            frame.span(kAtom_fz)[0]};
}

} // namespace

TEST(LangevinForceSamples, SameStepGetsDifferentRandomVectors) {
    cpu::Context context;
    auto frame = atom();
    cpu::md::FixLangevin thermostat{300.0, 600.0, 100.0, 0.5, 173};
    thermostat.set_run_span(0, 10);
    thermostat.build({context, frame});

    const auto first = sample(thermostat, context, frame);
    const auto second = sample(thermostat, context, frame);

    EXPECT_NE(first, second);
    EXPECT_DOUBLE_EQ(thermostat.T_target(), 300.0);
}

TEST(LangevinForceSamples, FreshBuildReproducesTheSampleSequence) {
    cpu::Context context;
    auto frame = atom();
    cpu::md::FixLangevin thermostat{300.0, -1.0, 100.0, 0.5, 173};
    thermostat.build({context, frame});
    const auto first = sample(thermostat, context, frame);
    const auto second = sample(thermostat, context, frame);

    thermostat.build({context, frame});

    EXPECT_EQ(sample(thermostat, context, frame), first);
    EXPECT_EQ(sample(thermostat, context, frame), second);
}

TEST(LangevinForceSamples, EmptyOwnedLaunchStillConsumesOneSample) {
    cpu::Context context;
    auto frame = atom();
    cpu::Frame empty(0);
    cpu::md::FixLangevin reference{300.0, -1.0, 100.0, 0.5, 173};
    reference.build({context, frame});
    const auto first = sample(reference, context, frame);
    const auto second = sample(reference, context, frame);
    cpu::md::FixLangevin thermostat{300.0, -1.0, 100.0, 0.5, 173};
    thermostat.build({context, empty});

    thermostat.launch({context, empty, slot_of(md::Phase::post_force), 0});
    const auto after_empty = sample(thermostat, context, frame);

    EXPECT_EQ(after_empty, second);
    EXPECT_NE(after_empty, first);
}
