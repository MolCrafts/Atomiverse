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
