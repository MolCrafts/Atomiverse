/// @file tests/cpu/driver/test_driver_live_table.cpp
/// @brief Single-method gates for the live mount table after build().

#include <functional>
#include <gtest/gtest.h>

#include "atomiverse/core/driver_template.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"

namespace {

class LiveProbe : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "live_probe"; }
    void launch(const ATV::cpu::LaunchContext&) override { ++launches; }
    std::span<const ATV::GhostField> declare_ghosts() const override {
        static constexpr ATV::GhostField fields[] = {
            ATV::GhostField::read_stable(ATV::kAtom_charge, "live_probe")};
        static constexpr ATV::GhostField invalid[] = {
            ATV::GhostField::read_stable("missing", ATV::ColDType::Storage, 1,
                                         "live_probe")};
        if (missing)
            return invalid;
        return charges ? std::span<const ATV::GhostField>(fields)
                       : std::span<const ATV::GhostField>{};
    }
    bool requires_full_frame() const override { return full_frame; }
    bool full_frame = false;
    bool charges = false;
    bool missing = false;
    int launches = 0;
};

} // namespace

TEST(TestDriver, BuildRetainsMountsAndSecondCompileMatchesPlan) {
    ATV::cpu::Evaluate eval;
    LiveProbe a;
    LiveProbe b;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(a);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(b);
    eval.build();
    // 2 user probes + 4 preset-mounted infrastructure components. The halo
    // adapters stopped being auto-mounted when Comm moved onto the Runtime
    // (lammps-comm-brick-05), dropping the count from 7.
    EXPECT_EQ(eval.mounted().size(), 6u);
    EXPECT_TRUE(eval.is_mounted(ATV::PluginIdentity{&a}));
    EXPECT_TRUE(eval.is_mounted(ATV::PluginIdentity{&b}));
    EXPECT_FALSE((ATV::PluginIdentity{&a} == ATV::PluginIdentity{&b}));

    const auto first = eval.plan();
    eval.build();
    const auto& second = eval.plan();
    ASSERT_EQ(first.slots.size(), second.slots.size());
    for (std::size_t s = 0; s < first.slots.size(); ++s) {
        ASSERT_EQ(first.slots[s].ops.size(), second.slots[s].ops.size());
        for (std::size_t o = 0; o < first.slots[s].ops.size(); ++o)
            EXPECT_EQ(first.slots[s].ops[o].component,
                      second.slots[s].ops[o].component);
    }
}

TEST(TestDriver, LiveLoadRefreshesGhostContractAndTimingShape) {
    LiveProbe first, added;
    added.charges = true;
    ATV::PlanBuilder<ATV::cpu::Component> original, addition, combined;
    original.add({0, 0, &first});
    addition.add({0, 0, &added});
    combined.add({0, 0, &first});
    combined.add({0, 1, &added});
    const auto initial = original.build(1);
    const auto extra = addition.build(1);
    const auto expanded = combined.build(1);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.enable_timing(true);
    runtime.init();
    runtime.load(initial);
    runtime.set_frame(frame);
    runtime.build_components();
    ASSERT_FALSE(runtime.context().md.ghost->contains(ATV::kAtom_charge.key));
    runtime.build_plan(extra);
    runtime.load(expanded);
    EXPECT_TRUE(runtime.context().md.ghost->contains(ATV::kAtom_charge.key));
    EXPECT_NO_THROW(runtime.step(1));
    runtime.destroy_plan(extra);
    runtime.load(initial);
    EXPECT_FALSE(runtime.context().md.ghost->contains(ATV::kAtom_charge.key));
    runtime.destroy_components();
}

TEST(TestDriver, FailedLiveRefreshCannotBeBypassedByStepping) {
    LiveProbe first, added;
    added.missing = true;
    ATV::PlanBuilder<ATV::cpu::Component> original, addition, combined;
    original.add({0, 0, &first});
    addition.add({0, 0, &added});
    combined.add({0, 0, &first});
    combined.add({0, 1, &added});
    const auto initial = original.build(1);
    const auto extra = addition.build(1);
    const auto expanded = combined.build(1);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(initial);
    runtime.set_frame(frame);
    runtime.build_components();
    runtime.build_plan(extra);
    EXPECT_THROW(runtime.load(expanded), std::runtime_error);
    EXPECT_THROW(runtime.step(1), std::runtime_error);
    EXPECT_EQ(first.launches, 0);
    EXPECT_EQ(added.launches, 0);
    runtime.destroy_plan(extra);
    runtime.load(initial);
    runtime.destroy_components();
}

#ifdef ATV_HAS_MPI
TEST(TestDriver, RejectsWholeSystemComponentsBeforePartitioning) {
    LiveProbe evaluator;
    evaluator.full_frame = true;
    ATV::PlanBuilder<ATV::cpu::Component> builder;
    builder.add({0, 0, &evaluator});
    const auto plan = builder.build(1);
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = -0.1;
    ATV::cpu::md::mpi::DomainDecomposition spatial;
    // Only query the borrowed communicator token; no MPI operation is called.
    spatial.cart_comm = MPI_COMM_SELF;
    ATV::cpu::Runtime runtime;
    runtime.load(plan);
    runtime.context().mpi_size = 2;
    runtime.context().md.dd = &spatial;
    EXPECT_THROW(runtime.set_frame(frame), std::runtime_error);
    EXPECT_EQ(frame.nlocal(), 2);
    EXPECT_DOUBLE_EQ(frame.span(ATV::kAtom_x)[0], -0.1);
    spatial.cart_comm = MPI_COMM_NULL;
    EXPECT_NO_THROW(runtime.set_frame(frame));
}
#endif

namespace {
class LifecycleProbe : public ATV::cpu::Component {
  public:
    explicit LifecycleProbe(std::string label) : label_(std::move(label)) {}
    std::string name() const override { return label_; }
    void build(const ATV::cpu::BuildContext&) override { ++builds; }
    void launch(const ATV::cpu::LaunchContext&) override {
        ++launches;
        if (action)
            action();
    }
    void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
    int frequency() const override { return cadence; }
    std::function<void()> action;
    int builds = 0, launches = 0, destroys = 0, cadence = 0;

  private:
    std::string label_;
};
class LiveLoop : public ATV::cpu::Driver<ATV::sampling::Phase> {
  public:
    void validate_components() const override {}
    using Driver::begin_run_;
    using Driver::flush_live_;
    ATV::cpu::Runtime& runtime() { return *this->backend_; }
};
} // namespace

TEST(TestDriver, NestedLiveBindingsShareLifecycleAndRefreshBeforeDispatch) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LifecycleProbe first("first"), added("added");
    LiveLoop child, parent;
    child.on(P::energy).add_ref(first);
    parent.on(P::energy).add_ref(child, 0, 1);
    parent.build();
    parent.begin_run_(frame);
    EXPECT_EQ(first.builds, 1);
    child.on(P::energy).add_ref(added);
    parent.begin_run_(frame);
    EXPECT_EQ(first.builds, 1);
    EXPECT_EQ(added.builds, 1);
    parent.runtime().step_stage(ATV::slot_of(P::energy), 0);
    EXPECT_EQ(added.launches, 1);
    child.unfix("first");
    EXPECT_EQ(first.destroys, 1);
    parent.begin_run_(frame);
    parent.runtime().step_stage(ATV::slot_of(P::energy), 1);
    EXPECT_EQ(first.launches, 1);
    EXPECT_EQ(added.launches, 2);
    parent.flush_live_();
    EXPECT_EQ(first.destroys, 1);
    EXPECT_EQ(added.destroys, 1);
}

TEST(TestDriver, CachedPhaseBuilderRejectsMutationDuringNestedDispatch) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LifecycleProbe mutator("mutator"), added("added");
    LiveLoop child, parent;
    auto cached = child.on(P::energy);
    cached.add_ref(mutator);
    mutator.action = [&] { cached.add_ref(added); };
    parent.on(P::energy).add_ref(child, 0, 1);
    parent.build();
    parent.begin_run_(frame);
    EXPECT_THROW(parent.runtime().step_stage(ATV::slot_of(P::energy), 0),
                 std::logic_error);
    EXPECT_FALSE(child.is_mounted({&added}));
    EXPECT_NO_THROW(cached.add_ref(added));
    parent.begin_run_(frame);
    EXPECT_EQ(added.builds, 1);
    parent.flush_live_();
}

TEST(TestDriver, PhysicalSiblingContractsAreIsolated) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LiveProbe valid, invalid;
    valid.charges = true;
    invalid.missing = true;
    ATV::cpu::Evaluate first, second;
    first.set_evaluator_ref(valid);
    second.set_evaluator_ref(invalid);
    LiveLoop parent;
    parent.on(P::energy).add_ref(first, 0, 0).add_ref(second, 1, 0);
    parent.build();
    ASSERT_NO_THROW(parent.begin_run_(frame));
    EXPECT_FALSE(
        parent.runtime().context().md.ghost->contains(ATV::kAtom_charge.key));
    {
        ATV::cpu::Runtime::ScopedPlan active(parent.runtime(), first.plan());
        EXPECT_TRUE(parent.runtime().context().md.ghost->contains(
            ATV::kAtom_charge.key));
    }
    EXPECT_THROW(
        (ATV::cpu::Runtime::ScopedPlan(parent.runtime(), second.plan())),
        std::runtime_error);
    parent.flush_live_();
}

TEST(TestDriver, StagedDispatchPreservesFrequencyAndCountsCompletedSteps) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LifecycleProbe probe("cadence");
    probe.cadence = 2;
    LiveLoop loop;
    loop.on(P::end_of_step).add_ref(probe);
    loop.build();
    loop.enable_timing();
    loop.begin_run_(frame);
    auto& runtime = loop.runtime();
    runtime.step_stage(ATV::slot_of(P::energy), 1);
    EXPECT_EQ(runtime.timing().steps(), 0);
    runtime.step_stage(ATV::slot_of(P::end_of_step), 1);
    EXPECT_EQ(probe.launches, 0);
    runtime.step_stage(ATV::slot_of(P::end_of_step), 2);
    EXPECT_EQ(probe.launches, 1);
    EXPECT_EQ(runtime.timing().steps(), 2);
    loop.flush_live_();
}

TEST(TestDriver, BoundDriverMoveFailsBeforeChangingTheSource) {
    ATV::cpu::Frame frame(1);
    LifecycleProbe probe("move_probe");
    LiveLoop loop;
    loop.on(ATV::sampling::Phase::energy).add_ref(probe);
    loop.build();
    loop.begin_run_(frame);
    EXPECT_THROW((LiveLoop(std::move(loop))), std::logic_error);
    EXPECT_NO_THROW(loop.runtime().step_stage(
        ATV::slot_of(ATV::sampling::Phase::energy), 0));
    EXPECT_EQ(probe.launches, 1);
    loop.flush_live_();
}

TEST(TestDriver, ShutdownReleasesLiveComponentsAndRestartsCleanly) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LifecycleProbe probe("shutdown_probe");
    LiveLoop loop;
    loop.on(P::energy).add_ref(probe);
    loop.build();
    loop.begin_run_(frame);
    probe.action = [&] { loop.shutdown(); };
    EXPECT_THROW(loop.runtime().step_stage(ATV::slot_of(P::energy), 0),
                 std::logic_error);
    // Shutdown itself is refused during dispatch. Its escaping exception
    // aborts the Driver-owned run while all mounted objects still live.
    EXPECT_EQ(probe.destroys, 1);
    EXPECT_TRUE(loop.runtime().failed());
    probe.action = {};
    EXPECT_NO_THROW(loop.begin_run_(frame));
    EXPECT_EQ(probe.builds, 2);
    EXPECT_NO_THROW(loop.runtime().step_stage(ATV::slot_of(P::energy), 0));
    EXPECT_EQ(probe.destroys, 1);
    ATV::Driver<P, ATV::cpu::device_tag>& base = loop;
    base.shutdown();
    EXPECT_EQ(probe.destroys, 2);
    EXPECT_NO_THROW(loop.build());
    loop.begin_run_(frame);
    EXPECT_EQ(probe.builds, 3);
    loop.shutdown();
    loop.shutdown();
    EXPECT_EQ(probe.destroys, 3);
}

TEST(TestDriver, MountedChildCannotShutdownItsParentRuntime) {
    using P = ATV::sampling::Phase;
    ATV::cpu::Frame frame(1);
    LifecycleProbe probe("nested_shutdown_probe");
    LiveLoop child, parent;
    child.on(P::energy).add_ref(probe);
    parent.on(P::energy).add_ref(child, 0, 1);
    parent.build();
    parent.begin_run_(frame);
    EXPECT_THROW(child.shutdown(), std::logic_error);
    EXPECT_EQ(probe.destroys, 0);
    parent.shutdown();
    EXPECT_EQ(probe.destroys, 1);
    EXPECT_NO_THROW(child.shutdown());
}
