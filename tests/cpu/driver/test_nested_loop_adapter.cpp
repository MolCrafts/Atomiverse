/// @file tests/cpu/driver/test_nested_loop_adapter.cpp
/// @brief Single-method gates for cpu::NestedLoopAdapter.

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/nested_loop_adapter.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

class OuterLoop : public cpu::Driver<ATV::md::Phase> {
  public:
    int run(cpu::Frame& frame, int budget) {
        setup_pass(frame);
        int iters = 0;
        for (int s = 0; s < budget; ++s) {
            mutable_outer_status().reset();
            step_once(s);
            ++iters;
        }
        teardown_pass();
        return iters;
    }

  protected:
    void validate_components() const override {}
};

class InnerLoop : public cpu::Driver<ATV::minimizer::Phase> {
  public:
    int default_budget() const override { return 10; }

  protected:
    void validate_components() const override {}
};

class HfInner : public cpu::Driver<ATV::scf::Phase>, public cpu::HF {
  protected:
    void validate_components() const override {}
};

class Counter : public cpu::Component {
  public:
    std::string name() const override { return "counter"; }
    void build(const cpu::BuildContext&) override { ++builds; }
    void destroy(const cpu::DestroyContext&) override { ++destroys; }
    void launch(const cpu::LaunchContext&) override { ++launches; }
    int builds = 0;
    int destroys = 0;
    int launches = 0;
};

class RecordingAdapter : public cpu::NestedLoopAdapter {
  public:
    using NestedLoopAdapter::NestedLoopAdapter;
    void launch(const cpu::LaunchContext& ctx) override {
        stage_at_entry = ctx.stage;
        NestedLoopAdapter::launch(ctx);
        stage_at_exit = ctx.stage;
    }
    int stage_at_entry = -1;
    int stage_at_exit = -1;
};

cpu::Frame two_atoms() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(ATV::kAtom_x)[i] = 0.5 * i;
        f.span(ATV::kAtom_y)[i] = 0.0;
        f.span(ATV::kAtom_z)[i] = 0.0;
        f.span(ATV::kAtom_mass)[i] = 1.0;
    }
    return f;
}

} // namespace

class TestNestedLoopAdapter : public ::testing::Test {};

TEST_F(TestNestedLoopAdapter, BuildCallsBuildOn) {
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    cpu::NestedLoopAdapter adapter(inner, 1);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();
    EXPECT_EQ(counter.builds, 0);
    auto frame = two_atoms();
    outer.run(frame, 1);
    EXPECT_EQ(counter.builds, 1);
}

TEST_F(TestNestedLoopAdapter, LaunchRunsInnerLoopToBudget) {
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    cpu::NestedLoopAdapter adapter(inner, 3);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();
    auto frame = two_atoms();
    outer.run(frame, 1);
    EXPECT_EQ(counter.launches, 3);
}

TEST_F(TestNestedLoopAdapter, LaunchKeepsOuterStageOnComponentContext) {
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    RecordingAdapter adapter(inner, 1);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();
    auto frame = two_atoms();
    outer.run(frame, 1);
    EXPECT_EQ(adapter.stage_at_entry, ATV::slot_of(ATV::md::Phase::force));
    EXPECT_EQ(adapter.stage_at_exit, ATV::slot_of(ATV::md::Phase::force));
    const auto& ops =
        outer.plan().slots[ATV::slot_of(ATV::md::Phase::force)].ops;
    ASSERT_FALSE(ops.empty());
    EXPECT_NE(ops.front().component, nullptr);
    EXPECT_NE(dynamic_cast<cpu::NestedLoopAdapter*>(ops.front().component),
              nullptr);
}

TEST_F(TestNestedLoopAdapter, DestroyCallsDestroyOn) {
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    cpu::NestedLoopAdapter adapter(inner, 1);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();
    auto frame = two_atoms();
    outer.run(frame, 1);
    EXPECT_EQ(counter.destroys, 1);
}

TEST_F(TestNestedLoopAdapter, ProvidesEnergyWhenInnerIsHf) {
    HfInner inner;
    cpu::NestedLoopAdapter adapter(inner);
    EXPECT_TRUE(adapter.provides_energy());
}

TEST_F(TestNestedLoopAdapter, ProvidesForceWhenInnerIsHf) {
    HfInner inner;
    cpu::NestedLoopAdapter adapter(inner);
    EXPECT_TRUE(adapter.provides_force());
}

TEST_F(TestNestedLoopAdapter, ProvidesEnergyFalseWhenInnerIsNotHf) {
    InnerLoop inner;
    cpu::NestedLoopAdapter adapter(inner);
    EXPECT_FALSE(adapter.provides_energy());
}

TEST_F(TestNestedLoopAdapter, ProvidesForceFalseWhenInnerIsNotHf) {
    InnerLoop inner;
    cpu::NestedLoopAdapter adapter(inner);
    EXPECT_FALSE(adapter.provides_force());
}

TEST_F(TestNestedLoopAdapter, NullUniquePtrThrows) {
    std::unique_ptr<cpu::DriverNode> empty;
    EXPECT_THROW(cpu::NestedLoopAdapter(std::move(empty)),
                 std::invalid_argument);
}

TEST_F(TestNestedLoopAdapter, NegativeBudgetUsesDefault) {
    InnerLoop inner;
    Counter counter;
    inner.on(ATV::minimizer::Phase::force).add_ref(counter);
    cpu::NestedLoopAdapter adapter(inner, -1);
    OuterLoop outer;
    outer.on(ATV::md::Phase::force).add_ref(adapter);
    outer.build();
    auto frame = two_atoms();
    outer.run(frame, 1);
    EXPECT_EQ(counter.launches, 10);
}

TEST_F(TestNestedLoopAdapter, AdapterIsComponentNotDriverNode) {
    static_assert(std::is_base_of_v<cpu::Component, cpu::NestedLoopAdapter>);
    static_assert(!std::is_base_of_v<cpu::DriverNode, cpu::NestedLoopAdapter>);
    static_assert(!std::is_base_of_v<cpu::Component, InnerLoop>);
    SUCCEED();
}
