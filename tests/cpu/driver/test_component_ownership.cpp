/// @file tests/cpu/driver/test_component_ownership.cpp
/// @brief Driver/PhaseBuilder component lifetime is explicit in the API.

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/async_consumer.h"
#include "atomiverse/cpu/asynchronous/capture_buffer.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

namespace {

struct ProbeState {
    int builds = 0;
    int launches = 0;
    int destroys = 0;
};

class LifetimeProbe final : public ATV::cpu::Potential {
  public:
    explicit LifetimeProbe(std::shared_ptr<ProbeState> state)
        : state_(std::move(state)) {}

    LifetimeProbe(const LifetimeProbe&) = delete;
    LifetimeProbe& operator=(const LifetimeProbe&) = delete;
    LifetimeProbe(LifetimeProbe&& other) noexcept
        : state_(std::move(other.state_)) {}
    LifetimeProbe& operator=(LifetimeProbe&&) = delete;

    ~LifetimeProbe() override {
        if (state_)
            ++state_->destroys;
    }

    std::string name() const override { return "lifetime_probe"; }
    ATV::Role role() const override { return 0; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::evaluate::Phase::evaluate, 0)};
    }
    void build(const ATV::cpu::BuildContext&) override { ++state_->builds; }
    void launch(const ATV::cpu::LaunchContext&) override { ++state_->launches; }

  private:
    std::shared_ptr<ProbeState> state_;
};

template <typename Driver, typename Component, typename = void>
struct can_implicitly_add_lvalue : std::false_type {};

template <typename Driver, typename Component>
struct can_implicitly_add_lvalue<
    Driver, Component,
    std::void_t<decltype(std::declval<Driver&>().add(
        std::declval<Component&>()))>> : std::true_type {};

template <typename Builder, typename Component, typename = void>
struct can_implicitly_bind_lvalue : std::false_type {};

template <typename Builder, typename Component>
struct can_implicitly_bind_lvalue<
    Builder, Component,
    std::void_t<decltype(std::declval<Builder&>().add(
        std::declval<Component&>(), 0))>> : std::true_type {};

template <typename Driver, typename Component, typename = void>
struct can_implicitly_set_observer_lvalue : std::false_type {};

template <typename Driver, typename Component>
struct can_implicitly_set_observer_lvalue<
    Driver, Component,
    std::void_t<decltype(std::declval<Driver&>().set_observer(
        std::declval<Component&>()))>> : std::true_type {};

using Evaluate = ATV::cpu::Evaluate;
using EvaluatePhaseBuilder =
    decltype(std::declval<Evaluate&>().on(ATV::evaluate::Phase::evaluate));

// Public Driver::add is gone — only on().add / set_evaluator remain.
static_assert(!can_implicitly_add_lvalue<Evaluate, LifetimeProbe>::value,
              "Driver::add must not exist on the public surface");
static_assert(
    !can_implicitly_bind_lvalue<EvaluatePhaseBuilder, LifetimeProbe>::value,
    "PhaseBuilder::add(lvalue) must stay deleted; use add_ref");
static_assert(
    !can_implicitly_set_observer_lvalue<Evaluate, LifetimeProbe>::value,
    "categorical setters must not reintroduce implicit lvalue borrowing");

TEST(ComponentOwnership, RvalueIsOwnedByDriver) {
    auto state = std::make_shared<ProbeState>();
    {
        Evaluate eval;
        eval.set_evaluator(LifetimeProbe{state});
        eval.build();
        ATV::cpu::Frame frame(1);
        eval.run(frame);
        EXPECT_EQ(state->builds, 1);
        EXPECT_EQ(state->launches, 1);
        EXPECT_EQ(state->destroys, 0);
    }
    EXPECT_EQ(state->destroys, 1);
}

TEST(ComponentOwnership, UniquePointerTransfersOwnership) {
    auto state = std::make_shared<ProbeState>();
    {
        Evaluate eval;
        eval.set_evaluator(std::make_unique<LifetimeProbe>(state));
        eval.build();
        ATV::cpu::Frame frame(1);
        eval.run(frame);
        EXPECT_EQ(state->launches, 1);
        EXPECT_EQ(state->destroys, 0);
    }
    EXPECT_EQ(state->destroys, 1);
}

TEST(ComponentOwnership, AddRefBorrowsAndLeavesObjectAlive) {
    auto state = std::make_shared<ProbeState>();
    {
        LifetimeProbe probe(state);
        {
            Evaluate eval;
            eval.set_evaluator_ref(probe);
            eval.build();
            ATV::cpu::Frame frame(1);
            eval.run(frame);
            EXPECT_EQ(state->launches, 1);
        }
        EXPECT_EQ(state->destroys, 0);
    }
    EXPECT_EQ(state->destroys, 1);
}

TEST(ComponentOwnership, PhaseBuilderAddRefBorrows) {
    auto state = std::make_shared<ProbeState>();
    LifetimeProbe probe(state);
    {
        Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate).add_ref(probe);
        eval.build();
        ATV::cpu::Frame frame(1);
        eval.run(frame);
        EXPECT_EQ(state->launches, 1);
    }
    EXPECT_EQ(state->destroys, 0);
}

TEST(ComponentOwnership, PhaseBuilderRvalueIsOwned) {
    auto state = std::make_shared<ProbeState>();
    {
        Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate).add(LifetimeProbe{state});
        eval.build();
        ATV::cpu::Frame frame(1);
        eval.run(frame);
        EXPECT_EQ(state->launches, 1);
        EXPECT_EQ(state->destroys, 0);
    }
    EXPECT_EQ(state->destroys, 1);
}

TEST(ComponentOwnership, PhaseBuilderUniquePointerTransfersOwnership) {
    auto state = std::make_shared<ProbeState>();
    {
        Evaluate eval;
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(std::make_unique<LifetimeProbe>(state));
        eval.build();
        ATV::cpu::Frame frame(1);
        eval.run(frame);
        EXPECT_EQ(state->launches, 1);
        EXPECT_EQ(state->destroys, 0);
    }
    EXPECT_EQ(state->destroys, 1);
}

TEST(ComponentOwnership, NullOwnedPointerFailsLoudly) {
    Evaluate eval;
    std::unique_ptr<LifetimeProbe> null_probe;
    EXPECT_THROW(eval.set_evaluator(std::move(null_probe)),
                 std::invalid_argument);
}

TEST(ComponentOwnership, PhaseBuilderNullOwnedPointerFailsLoudly) {
    Evaluate eval;
    std::unique_ptr<LifetimeProbe> null_probe;
    EXPECT_THROW(
        eval.on(ATV::evaluate::Phase::evaluate).add(std::move(null_probe)),
        std::invalid_argument);
}

} // namespace

namespace {
class RuntimeLifecycleProbe : public ATV::cpu::Component {
  public:
    std::string name() const override { return "runtime_lifecycle_probe"; }
    void build(const ATV::cpu::BuildContext&) override {
        if (fail_build)
            throw std::runtime_error("build failed");
        ++builds;
    }
    void launch(const ATV::cpu::LaunchContext&) override {}
    void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
    int builds = 0;
    int destroys = 0;
    bool fail_build = false;
};
} // namespace

TEST(ComponentOwnership, RuntimeBuildsAnObjectOnceAcrossPhaseBindings) {
    RuntimeLifecycleProbe probe;
    ATV::PlanBuilder<ATV::cpu::Component> builder;
    builder.add({0, 0, &probe});
    builder.add({1, 0, &probe});
    const auto plan = builder.build(2);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.build_components();
    EXPECT_EQ(probe.builds, 1);
    runtime.destroy_components();
    EXPECT_EQ(probe.destroys, 1);
}

TEST(ComponentOwnership, NestedPlansShareObjectLifetime) {
    RuntimeLifecycleProbe probe;
    ATV::PlanBuilder<ATV::cpu::Component> builder;
    builder.add({0, 0, &probe});
    const auto plan = builder.build(1);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.build_plan(plan);
    runtime.build_plan(plan);
    EXPECT_EQ(probe.builds, 1);
    runtime.destroy_plan(plan);
    EXPECT_EQ(probe.destroys, 0);
    runtime.destroy_plan(plan);
    EXPECT_EQ(probe.destroys, 1);
}

TEST(ComponentOwnership, FailedPlanBuildReleasesEarlierObjects) {
    RuntimeLifecycleProbe first, failing;
    failing.fail_build = true;
    ATV::PlanBuilder<ATV::cpu::Component> builder;
    builder.add({0, 0, &first});
    builder.add({0, 1, &failing});
    const auto plan = builder.build(1);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    EXPECT_THROW(runtime.build_plan(plan), std::runtime_error);
    EXPECT_EQ(first.builds, 1);
    EXPECT_EQ(first.destroys, 1);
    failing.fail_build = false;
    runtime.build_plan(plan);
    EXPECT_EQ(first.builds, 2);
    runtime.destroy_plan(plan);
    EXPECT_EQ(first.destroys, 2);
}

TEST(ComponentOwnership, DestroyingAsyncObjectUnsubscribesAndFinalizesIt) {
    class Observer : public ATV::cpu::AsyncConsumer {
      public:
        std::string name() const override { return "async_lifetime"; }
        ATV::CaptureColumns capture_columns() const override { return {}; }
        void on_snapshot(const ATV::cpu::FrameSnapshot&) override {}
        void finalize() override { ++finalized; }
        int finalized = 0;
    } observer;
    ATV::PlanBuilder<ATV::cpu::Component> builder;
    builder.add({0, 0, &observer});
    const auto plan = builder.build(1);
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.build_plan(plan);
    auto* buffer = runtime.context().capture.buffer;
    ASSERT_EQ(buffer, &runtime.capture());
    EXPECT_TRUE(buffer->has_consumers());
    runtime.destroy_plan(plan);
    EXPECT_FALSE(buffer->has_consumers());
    EXPECT_EQ(observer.finalized, 1);
    runtime.shutdown();
    EXPECT_EQ(observer.finalized, 1);
}

TEST(ComponentOwnership, PolledCriterionParticipatesInObjectLifecycle) {
    class Criterion : public ATV::cpu::Criterion {
      public:
        std::string name() const override { return "polled_lifetime"; }
        void launch(const ATV::cpu::LaunchContext&) override {}
        ATV::Verdict verdict(const ATV::cpu::LoopState&) override {
            return ATV::Verdict::Continue;
        }
        void build(const ATV::cpu::BuildContext&) override { ++builds; }
        void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
        int builds = 0, destroys = 0;
    } criterion;
    auto state = std::make_shared<ProbeState>();
    Evaluate eval;
    eval.set_evaluator(LifetimeProbe{state});
    eval.add_criterion_ref(criterion);
    eval.build();
    ATV::cpu::Frame frame(2);
    ATV::cpu::Runtime runtime;
    runtime.init();
    runtime.load(eval.plan());
    runtime.set_frame(frame);
    runtime.build_components();
    EXPECT_EQ(criterion.builds, 1);
    runtime.destroy_components();
    EXPECT_EQ(criterion.destroys, 1);
    eval.unfix("polled_lifetime");
    EXPECT_TRUE(eval.plan().lifecycle_only.empty());
    for (const auto* component : eval.plan().components())
        EXPECT_NE(component, &criterion);
}

namespace {
class FailingLifecycleProbe : public ATV::cpu::Potential {
  public:
    std::string name() const override { return "failing_lifecycle_probe"; }
    void build(const ATV::cpu::BuildContext&) override {
        if (fail_build)
            throw std::runtime_error("build failure");
        ++builds;
    }
    void launch(const ATV::cpu::LaunchContext&) override {
        if (fail_launch)
            throw std::runtime_error("launch failure");
    }
    void destroy(const ATV::cpu::DestroyContext& ctx) override {
        ++destroys;
        observed_rows = ctx.frame.nlocal();
    }
    bool fail_build = false, fail_launch = true;
    int builds = 0, destroys = 0;
    std::size_t observed_rows = 0;
};
class DirectScfLoop : public ATV::cpu::Driver<ATV::scf::Phase> {
  public:
    void run(ATV::cpu::Frame& frame) {
        setup_pass(frame);
        step_once(0);
        teardown_pass();
    }

  protected:
    void validate_components() const override {}
};
} // namespace

TEST(ComponentOwnership,
     DriverLaunchFailureDestroysWhileFrameLivesAndRetryRebuilds) {
    FailingLifecycleProbe probe;
    Evaluate driver;
    driver.set_evaluator_ref(probe);
    driver.build();
    {
        ATV::cpu::Frame transient(2);
        EXPECT_THROW(driver.run(transient), std::runtime_error);
        EXPECT_EQ(probe.destroys, 1);
        EXPECT_EQ(probe.observed_rows, 2u);
    }
    probe.fail_launch = false;
    ATV::cpu::Frame next(3);
    EXPECT_NO_THROW(driver.run(next));
    EXPECT_EQ(probe.builds, 2);
    EXPECT_EQ(probe.destroys, 2);
    EXPECT_EQ(probe.observed_rows, 3u);
    EXPECT_NO_THROW(driver.shutdown());
    EXPECT_EQ(probe.destroys, 2);
}

TEST(ComponentOwnership, DriverPartialBuildFailureUnwindsThenRetries) {
    FailingLifecycleProbe first, second;
    first.fail_launch = second.fail_launch = false;
    second.fail_build = true;
    Evaluate driver;
    driver.on(ATV::evaluate::Phase::evaluate).add_ref(first, 0);
    driver.on(ATV::evaluate::Phase::evaluate).add_ref(second, 1);
    driver.build();
    ATV::cpu::Frame frame(1);
    EXPECT_THROW(driver.run(frame), std::runtime_error);
    EXPECT_EQ(first.builds, 1);
    EXPECT_EQ(first.destroys, 1);
    EXPECT_EQ(second.destroys, 0); // build never completed
    second.fail_build = false;
    EXPECT_NO_THROW(driver.run(frame));
    EXPECT_EQ(first.builds, 2);
    EXPECT_EQ(first.destroys, 2);
    EXPECT_EQ(second.destroys, 1);
}

TEST(ComponentOwnership,
     NestedDriverFailureDetachesChildAndUnwindsSharedObjectOnce) {
    FailingLifecycleProbe shared;
    DirectScfLoop child, parent;
    child.on(ATV::scf::Phase::check).add_ref(shared);
    parent.on(ATV::scf::Phase::fock).add_ref(shared);
    parent.on(ATV::scf::Phase::check).add_ref(child, 0, 1);
    parent.build();
    ATV::cpu::Frame frame(1);
    // Throw inside the child rather than the parent's first shared binding.
    shared.fail_launch = false;
    class Throwing : public ATV::cpu::Component {
      public:
        bool fail = true;
        std::string name() const override { return "nested_throwing"; }
        void launch(const ATV::cpu::LaunchContext&) override {
            if (fail)
                throw std::runtime_error("nested failure");
        }
    } throwing;
    child.on(ATV::scf::Phase::check).add_ref(throwing, 1);
    EXPECT_THROW(parent.run(frame), std::runtime_error);
    EXPECT_EQ(shared.builds, 1);
    EXPECT_EQ(shared.destroys, 1);
    EXPECT_NO_THROW(child.shutdown()); // no stale parent Runtime attachment
    throwing.fail = false;
    EXPECT_NO_THROW(parent.run(frame));
    EXPECT_EQ(shared.builds, 2);
    EXPECT_EQ(shared.destroys, 2);
}

TEST(ComponentOwnership, SameNestedObjectAcrossStagesBuildsAndDestroysOnce) {
    class Probe : public ATV::cpu::Component {
      public:
        int builds = 0, destroys = 0, launches = 0;
        std::string name() const override { return "shared_nested_probe"; }
        void build(const ATV::cpu::BuildContext&) override { ++builds; }
        void launch(const ATV::cpu::LaunchContext&) override { ++launches; }
        void destroy(const ATV::cpu::DestroyContext&) override { ++destroys; }
    } probe;
    DirectScfLoop child, parent;
    child.on(ATV::scf::Phase::check).add_ref(probe);
    parent.on(ATV::scf::Phase::fock).add_ref(child, 0, 0);
    parent.on(ATV::scf::Phase::diagonalize).add_ref(child, 0, 2);
    parent.build();
    ATV::cpu::Frame frame(1);
    EXPECT_NO_THROW(parent.run(frame));
    EXPECT_EQ(probe.builds, 1);
    EXPECT_EQ(probe.launches, 2);
    EXPECT_EQ(probe.destroys, 1);
    EXPECT_NO_THROW(child.shutdown());
    EXPECT_NO_THROW(parent.run(frame));
    EXPECT_EQ(probe.builds, 2);
    EXPECT_EQ(probe.launches, 4);
    EXPECT_EQ(probe.destroys, 2);
}

TEST(ComponentOwnership, SharedNodeCannotBeAcquiredByAnotherRuntime) {
    FailingLifecycleProbe probe;
    probe.fail_launch = false;
    DirectScfLoop child, other_parent;
    child.on(ATV::scf::Phase::check).add_ref(probe);
    child.build();
    ATV::cpu::Frame frame(1);
    ATV::cpu::Runtime original;
    original.init();
    original.load(child.plan());
    original.set_frame(frame);
    child.build_on(original);
    other_parent.on(ATV::scf::Phase::fock).add_ref(child, 0, 0);
    other_parent.on(ATV::scf::Phase::check).add_ref(child, 0, 1);
    other_parent.build();
    EXPECT_THROW(other_parent.run(frame), std::logic_error);
    EXPECT_EQ(probe.builds, 1);
    EXPECT_EQ(probe.destroys, 0);
    EXPECT_NO_THROW(child.run_nested(original.context(), frame, 1));
    child.destroy_on(original);
    EXPECT_EQ(probe.destroys, 1);
}
