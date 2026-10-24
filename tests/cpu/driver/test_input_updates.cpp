/// Unit witnesses for exact host-input comparison and Runtime invalidation.
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/input_snapshot.h"

using namespace ATV;

namespace {
cpu::Frame inputs() {
    cpu::Frame frame(3);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = i + .5;
    }
    return frame;
}

struct InputForce : cpu::Potential {
    int builds = 0, invalidations = 0, initial_calls = 0;
    cpu::Runtime* runtime = nullptr;
    std::string name() const override { return "input_force"; }
    bool supports_input_changes(InputChange) const override { return true; }
    void build(const cpu::BuildContext& ctx) override {
        ++builds;
        runtime = ctx.runtime.exec.runtime;
    }
    void invalidate_inputs(InputChange, const cpu::BuildContext&) override {
        ++invalidations;
    }
    void launch(const cpu::LaunchContext& ctx) override {
        EXPECT_EQ(ctx.runtime.exec.runtime, runtime);
        initial_calls += ctx.runtime.exec.initializing_forces;
        for (int i = 0; i < ctx.frame.nlocal(); ++i)
            ctx.frame.span(kAtom_fx)[i] = -ctx.frame.span(kAtom_x)[i];
    }
};

struct UnknownCache : cpu::Component {
    std::string name() const override { return "unknown_cache"; }
    void launch(const cpu::LaunchContext&) override {}
};
struct ThrowingInputCache : cpu::Component {
    bool fail = false;
    std::string name() const override { return "throwing_input_cache"; }
    bool supports_input_changes(InputChange) const override { return true; }
    void invalidate_inputs(InputChange, const cpu::BuildContext&) override {
        if (fail)
            throw std::runtime_error("input cache refresh failed");
    }
    void launch(const cpu::LaunchContext&) override {}
};
} // namespace

TEST(FrameInputSnapshot, IgnoresOutputsAndDetectsNumericalInputs) {
    auto frame = inputs();
    cpu::FrameInputSnapshot snapshot;
    snapshot.capture(frame, cpu::FrameInputSnapshot::state_input);
    frame.span(kAtom_fx)[0] = 99;
    frame.meta.get(kEnergy) = -99;
    EXPECT_EQ(snapshot.changes(frame, cpu::FrameInputSnapshot::state_input),
              InputChange::None);
    frame.span(kAtom_x)[0] += .2;
    frame.span(kAtom_vy)[1] = .1;
    frame.span(kAtom_charge)[2] = .5;
    frame.box = cpu::Box::cubic(15);
    const auto changed =
        snapshot.changes(frame, cpu::FrameInputSnapshot::state_input);
    EXPECT_TRUE(has_input_change(changed, InputChange::Geometry));
    EXPECT_TRUE(has_input_change(changed, InputChange::Velocity));
    EXPECT_TRUE(has_input_change(changed, InputChange::Charge));
    EXPECT_TRUE(has_input_change(changed, InputChange::Cell));
    EXPECT_FALSE(has_input_change(changed, InputChange::Identity));
}

TEST(FrameInputSnapshot, IdentityStrideAndImageArePhysicalInputs) {
    auto frame = inputs();
    cpu::FrameInputSnapshot snapshot;
    snapshot.capture(frame, cpu::FrameInputSnapshot::state_input);
    frame.span(kAtom_image)(0, 1) = 1;
    EXPECT_EQ(snapshot.changes(frame, cpu::FrameInputSnapshot::state_input),
              InputChange::Geometry);
    snapshot.capture(frame, cpu::FrameInputSnapshot::state_input);
    frame.span(kAtom_type)[0] = 8;
    EXPECT_TRUE(has_input_change(
        snapshot.changes(frame, cpu::FrameInputSnapshot::state_input),
        InputChange::Identity));
    snapshot.capture(frame, cpu::FrameInputSnapshot::state_input);
    frame.meta.get(kBondPerAtom) += 1;
    EXPECT_TRUE(has_input_change(
        snapshot.changes(frame, cpu::FrameInputSnapshot::state_input),
        InputChange::Identity));
    snapshot.capture(frame, cpu::FrameInputSnapshot::state_input);
    frame.set_nlocal(2);
    EXPECT_TRUE(has_input_change(
        snapshot.changes(frame, cpu::FrameInputSnapshot::state_input),
        InputChange::Identity));
}

TEST(FrameInputSnapshot, SelectionExcludesUntransferredFieldsAndTheirStride) {
    auto frame = inputs();
    const auto positions = [](std::string_view block,
                              const cpu::ColumnRef& column) {
        return block == "atoms" && column.key == "x";
    };
    cpu::FrameInputSnapshot snapshot;
    snapshot.capture(frame, positions);
    frame.span(kAtom_mass)[0] = 2;
    frame.meta.get(kImproperPerAtom) += 1;
    EXPECT_EQ(snapshot.changes(frame, positions), InputChange::None);
    frame.span(kAtom_x)[0] += .1;
    EXPECT_EQ(snapshot.changes(frame, positions), InputChange::Geometry);
}

TEST(CpuInputUpdates, NumericalEditsReseedForcesWithoutRebuildingComponents) {
    auto frame = inputs();
    InputForce force;
    cpu::md::MD md;
    md.set_integrator(cpu::md::VelocityVerlet{.01});
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    md.run(frame, 0);
    EXPECT_EQ(force.initial_calls, 1);
    frame.span(kAtom_x)[0] = 3;
    frame.span(kAtom_vx)[0] = .4;
    md.run(frame, 0);
    EXPECT_EQ(frame.span(kAtom_fx)[0], -3);
    EXPECT_EQ(frame.span(kAtom_vx)[0], static_cast<StorageReal>(.4));
    EXPECT_EQ(force.builds, 1);
    EXPECT_EQ(force.initial_calls, 2);
    EXPECT_EQ(force.invalidations, 1);
    md.run(frame, 0);
    EXPECT_EQ(force.initial_calls, 2);
}

TEST(CpuInputUpdates, IdentityRefreshPreservesUserVelocityAndRuntime) {
    auto frame = inputs();
    InputForce force;
    cpu::md::MD md;
    md.set_integrator(cpu::md::VelocityVerlet{.01});
    md.on(ATV::md::Phase::initial_integrate)
        .add(cpu::md::InitVelocity{300, 17});
    md.on(ATV::md::Phase::force).add_ref(force);
    md.build();
    md.run(frame, 0);
    frame.set_nlocal(2);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_vx)[0] = .321;
    md.run(frame, 0);
    EXPECT_EQ(force.builds, 1);
    EXPECT_EQ(force.invalidations, 1);
    EXPECT_EQ(frame.span(kAtom_vx)[0], static_cast<StorageReal>(.321));
    EXPECT_EQ(frame.nlocal(), 2);
}

TEST(CpuInputUpdates,
     WholePlanPreflightRejectsUnknownIdentityCacheBeforeInvalidating) {
    auto frame = inputs();
    InputForce force;
    cpu::md::MD md;
    md.set_integrator(cpu::md::VelocityVerlet{.01});
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add(UnknownCache{});
    md.build();
    md.run(frame, 0);
    frame.span(kAtom_type)[0] = 8;
    EXPECT_THROW(md.run(frame, 0), std::runtime_error);
    EXPECT_EQ(force.invalidations, 0);
    frame.span(kAtom_type)[0] = 1;
    EXPECT_NO_THROW(md.run(frame, 0));
    EXPECT_EQ(force.builds, 1);
}

TEST(CpuInputUpdates,
     FailedRefreshStillInvalidatesAfterRestoringCheckpointInputs) {
    auto frame = inputs();
    InputForce force;
    ThrowingInputCache cache;
    cpu::md::MD md;
    md.set_integrator(cpu::md::VelocityVerlet{.01});
    md.on(ATV::md::Phase::force).add_ref(force);
    md.on(ATV::md::Phase::end_of_step).add_ref(cache);
    md.build();
    md.run(frame, 0);
    frame.span(kAtom_type)[0] = 8;
    cache.fail = true;
    EXPECT_THROW(md.run(frame, 0), std::runtime_error);
    EXPECT_EQ(force.invalidations, 1);
    frame.span(kAtom_type)[0] = 1;
    cache.fail = false;
    EXPECT_NO_THROW(md.run(frame, 0));
    EXPECT_EQ(force.invalidations, 2);
    EXPECT_EQ(force.builds, 1);
    EXPECT_EQ(force.initial_calls, 2);
}
