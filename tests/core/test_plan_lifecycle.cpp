/// @file tests/core/test_plan_lifecycle.cpp
/// @brief core Plan: the shared component lifecycle ledger and the MD
///        initial-force preparation schedule (used by the CUDA backend).
#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"

namespace {
struct LifecycleProbe {
    int id;
    std::string name() const { return "probe"; }
};

ATV::Plan<LifecycleProbe>
lifecycle_plan(std::initializer_list<LifecycleProbe*> components) {
    ATV::Plan<LifecycleProbe> plan;
    for (auto* component : components) {
        ATV::StageSlot<LifecycleProbe> slot{};
        slot.ops.push_back({component});
        plan.slots.push_back(std::move(slot));
    }
    return plan;
}
} // namespace

TEST(ComponentLifecycle, MultipleBindingsBuildAndDestroyOneInstance) {
    LifecycleProbe a{1}, b{2};
    auto plan = lifecycle_plan({&a, &b, &a});
    ATV::detail::ComponentLifecycle<LifecycleProbe> lifecycle;
    std::vector<int> events;
    auto destroy = [&](LifecycleProbe& c) { events.push_back(-c.id); };
    lifecycle.build(
        plan, [&](LifecycleProbe& c) { events.push_back(c.id); }, destroy);
    EXPECT_EQ(events, (std::vector<int>{1, 2}));
    lifecycle.destroy_all(destroy);
    EXPECT_EQ(events, (std::vector<int>{1, 2, -2, -1}));
    lifecycle.destroy_all(destroy);
    EXPECT_EQ(events.size(), 4u);
}

TEST(ComponentLifecycle, FailedBuildRollsBackAndCanRetry) {
    LifecycleProbe a{1}, b{2};
    auto plan = lifecycle_plan({&a, &b, &a});
    ATV::detail::ComponentLifecycle<LifecycleProbe> lifecycle;
    std::vector<int> events;
    bool fail = true;
    auto build = [&](LifecycleProbe& c) {
        events.push_back(c.id);
        if (fail && c.id == 2)
            throw std::runtime_error("partial build");
    };
    auto destroy = [&](LifecycleProbe& c) { events.push_back(-c.id); };
    EXPECT_THROW(lifecycle.build(plan, build, destroy), std::runtime_error);
    EXPECT_EQ(events, (std::vector<int>{1, 2, -2, -1}));
    fail = false;
    events.clear();
    lifecycle.build(plan, build, destroy);
    lifecycle.destroy_all(destroy);
    EXPECT_EQ(events, (std::vector<int>{1, 2, -2, -1}));
}

TEST(ComponentLifecycle, SharedNestedInstancesReleaseOnLastPlan) {
    LifecycleProbe a{1}, b{2};
    auto outer = lifecycle_plan({&a});
    auto inner = lifecycle_plan({&a, &b, &b});
    ATV::detail::ComponentLifecycle<LifecycleProbe> lifecycle;
    std::vector<int> events;
    auto build = [&](LifecycleProbe& c) { events.push_back(c.id); };
    auto destroy = [&](LifecycleProbe& c) { events.push_back(-c.id); };
    lifecycle.build(outer, build, destroy);
    lifecycle.build(inner, build, destroy);
    lifecycle.destroy(inner, destroy);
    EXPECT_EQ(events, (std::vector<int>{1, 2, -2}));
    lifecycle.destroy(outer, destroy);
    EXPECT_EQ(events, (std::vector<int>{1, 2, -2, -1}));
}

TEST(PlanPreparation, PlacementPrecedesNeighborsAndIntegrationIsExcluded) {
    LifecycleProbe probe{1};
    ATV::Plan<LifecycleProbe> plan;
    plan.kind = ATV::PhaseKind::MD;
    plan.slots.resize(ATV::slot_count<ATV::md::Phase>());
    for (int p = 0; p < int(plan.slots.size()); ++p) {
        plan.slots[p].phase = p;
        plan.slots[p].ops.push_back({&probe});
    }
    std::vector<int> phases, normal_launches;
    plan.prepare_forces([](int) {},
                        [&](const auto& slot, const auto&, bool normal) {
                            phases.push_back(slot.phase);
                            if (normal)
                                normal_launches.push_back(slot.phase);
                        });
    EXPECT_EQ(phases.front(), ATV::slot_of(ATV::md::Phase::post_integrate));
    EXPECT_EQ(phases.back(), ATV::slot_of(ATV::md::Phase::end_of_step));
    EXPECT_EQ(std::count(phases.begin(), phases.end(),
                         ATV::slot_of(ATV::md::Phase::initial_integrate)),
              0);
    EXPECT_EQ(std::count(phases.begin(), phases.end(),
                         ATV::slot_of(ATV::md::Phase::final_integrate)),
              0);
    EXPECT_EQ(normal_launches,
              (std::vector<int>{ATV::slot_of(ATV::md::Phase::force)}));
}
