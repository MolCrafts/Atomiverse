/// @file tests/cuda/driver/test_nested_requests.cu
/// @brief Request and exception contracts for one CUDA dispatch operation.
/// These probes launch no kernels and execute no scientific convergence loop.
#include "atomiverse/cuda/driver/backend_traits.cuh"
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
using namespace ATV;

class RequestProbe : public cuda::Component {
  public:
    std::string name() const override { return "request_probe"; }
    void launch(const cuda::LaunchContext& ctx) override {
        observed = {ctx.forces_requested, ctx.stress_requested,
                    ctx.initializing_forces, ctx.trial_evaluation};
        ++calls;
        if (fail)
            throw std::runtime_error("injected dispatch failure");
    }
    cuda::ExecServices observed{};
    int calls = 0;
    bool fail = false;
};

class RequestNode : public cuda::DriverNode {
  public:
    bool provides_energy() const { return energy; }
    bool provides_force() const { return force; }
    bool provides_system_stress() const { return stress; }
    bool requires_system_stress() const { return stress_required; }
    void build_on(cuda::Context&, cuda::Frame&) override {}
    void destroy_on(cuda::Context&, cuda::Frame&) override {}
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override {
        runtime = &ctx;
        device_frame = &frame;
        seen_budget = budget;
        seen = ctx.exec;
        probe.launch({ctx, frame, 0, 0});
        // A child owns its loop; even an exceptional exit must not leave the
        // parent pointing to child state or carrying narrowed requests.
        ctx.loop = {&status, &metrics};
        ctx.exec = {false, false, true, true};
        if (fail)
            throw std::runtime_error("injected nested failure");
        return 1;
    }
    bool energy = true, force = true, stress = true, stress_required = false;
    bool fail = false;
    cuda::Context* runtime = nullptr;
    cuda::Frame* device_frame = nullptr;
    int seen_budget = 0;
    cuda::ExecServices seen{};
    RequestProbe probe;
    IterationStatus status;
    ConvergenceMetrics metrics;
};
} // namespace

TEST(CudaNestedRequests, LaunchContextInheritsAllParentRequests) {
    cuda::Context runtime;
    cuda::Frame frame;
    runtime.exec = {false, false, true, true};
    const cuda::LaunchContext launch{runtime, frame, 0, 2};
    EXPECT_FALSE(launch.forces_requested);
    EXPECT_FALSE(launch.stress_requested);
    EXPECT_TRUE(launch.initializing_forces);
    EXPECT_TRUE(launch.trial_evaluation);
}

TEST(CudaNestedRequests, NestedSlotPropagatesIntentAndRestoresParent) {
    cuda::Context runtime;
    cuda::Frame frame;
    IterationStatus parent_status;
    ConvergenceMetrics parent_metrics;
    runtime.loop = {&parent_status, &parent_metrics};
    RequestNode node;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<RequestNode>{}, &node, 7, true, true);
    cuda::LaunchContext launch{runtime, frame, 0, 2};
    launch.forces_requested = false;
    launch.stress_requested = false;
    launch.initializing_forces = true;
    launch.trial_evaluation = true;
    slot->launch(launch);
    EXPECT_EQ(node.runtime, &runtime);
    EXPECT_EQ(node.device_frame, &frame);
    EXPECT_EQ(node.seen_budget, 7);
    EXPECT_FALSE(node.probe.observed.forces_requested);
    EXPECT_FALSE(node.probe.observed.stress_requested);
    EXPECT_TRUE(node.probe.observed.initializing_forces);
    EXPECT_TRUE(node.probe.observed.trial_evaluation);
    EXPECT_TRUE(runtime.exec.forces_requested);
    EXPECT_TRUE(runtime.exec.stress_requested);
    EXPECT_FALSE(runtime.exec.initializing_forces);
    EXPECT_FALSE(runtime.exec.trial_evaluation);
    EXPECT_EQ(runtime.loop.status, &parent_status);
    EXPECT_EQ(runtime.loop.metrics, &parent_metrics);
}

TEST(CudaNestedRequests, NestedSlotCannotWidenParentRequests) {
    cuda::Context runtime;
    cuda::Frame frame;
    runtime.exec = {false, false, true, true};
    RequestNode node;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<RequestNode>{}, &node, 1, true, true);
    cuda::LaunchContext launch{runtime, frame, 0, 0};
    launch.forces_requested = true;
    launch.stress_requested = true;
    launch.initializing_forces = false;
    launch.trial_evaluation = false;
    slot->launch(launch);
    EXPECT_FALSE(node.seen.forces_requested);
    EXPECT_FALSE(node.seen.stress_requested);
    EXPECT_TRUE(node.seen.initializing_forces);
    EXPECT_TRUE(node.seen.trial_evaluation);
}

TEST(CudaNestedRequests, NestedFailureRestoresParentAndAllowsRetry) {
    cuda::Context runtime;
    cuda::Frame frame;
    IterationStatus parent_status;
    ConvergenceMetrics parent_metrics;
    runtime.loop = {&parent_status, &parent_metrics};
    RequestNode node;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<RequestNode>{}, &node, 1, true, true);
    node.fail = true;
    EXPECT_THROW(slot->launch({runtime, frame, 0, 0}), std::runtime_error);
    EXPECT_EQ(runtime.loop.status, &parent_status);
    EXPECT_EQ(runtime.loop.metrics, &parent_metrics);
    EXPECT_TRUE(runtime.exec.forces_requested);
    EXPECT_TRUE(runtime.exec.stress_requested);
    EXPECT_FALSE(runtime.exec.initializing_forces);
    EXPECT_FALSE(runtime.exec.trial_evaluation);
    node.fail = false;
    EXPECT_NO_THROW(slot->launch({runtime, frame, 0, 0}));
    EXPECT_TRUE(node.seen.forces_requested);
    EXPECT_TRUE(node.seen.stress_requested);
    EXPECT_FALSE(node.seen.initializing_forces);
    EXPECT_FALSE(node.seen.trial_evaluation);
}

TEST(CudaNestedRequests, NestedOutputDeclarationsFollowMethodState) {
    RequestNode node;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<RequestNode>{}, &node, 1, true, true);
    EXPECT_TRUE(slot->provides_energy());
    EXPECT_TRUE(slot->provides_force());
    EXPECT_TRUE(slot->provides_system_stress());
    EXPECT_FALSE(slot->requires_system_stress());
    node.energy = node.force = node.stress = false;
    node.stress_required = true;
    EXPECT_FALSE(slot->provides_energy());
    EXPECT_FALSE(slot->provides_force());
    EXPECT_FALSE(slot->provides_system_stress());
    EXPECT_TRUE(slot->requires_system_stress());
}

TEST(CudaNestedRequests, StressCapabilityVetoPreservesForceAndRestoresParent) {
    cuda::Context runtime;
    cuda::Frame frame;
    RequestNode node;
    node.stress = false;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<RequestNode>{}, &node, 1, true, true);
    node.fail = true;
    EXPECT_THROW(slot->launch({runtime, frame, 0, 0}), std::runtime_error);
    EXPECT_TRUE(node.seen.forces_requested);
    EXPECT_FALSE(node.seen.stress_requested);
    EXPECT_TRUE(runtime.exec.forces_requested);
    EXPECT_TRUE(runtime.exec.stress_requested);
    node.fail = false;
    EXPECT_NO_THROW(slot->launch({runtime, frame, 0, 0}));
    EXPECT_TRUE(node.seen.forces_requested);
    EXPECT_FALSE(node.seen.stress_requested);
    EXPECT_TRUE(runtime.exec.stress_requested);
    node.stress = true;
    slot->launch({runtime, frame, 0, 0});
    EXPECT_TRUE(node.seen.stress_requested);
}

TEST(CudaNestedRequests, BackendCombinesPhaseAndInheritedRequests) {
    RequestProbe probe;
    Plan<cuda::Component> plan;
    plan.slots.push_back({0, {{&probe}}});
    plan.slots[0].forces_requested = false;
    plan.slots[0].stress_requested = false;
    cuda::Backend backend;
    backend.load(plan);
    backend.execute_phases(0, 0, 1);
    EXPECT_FALSE(probe.observed.forces_requested);
    EXPECT_FALSE(probe.observed.stress_requested);
    EXPECT_FALSE(probe.observed.initializing_forces);
    EXPECT_FALSE(probe.observed.trial_evaluation);
    plan.slots[0].forces_requested = true;
    plan.slots[0].stress_requested = true;
    backend.context().exec = {false, false, true, true};
    backend.execute_phases(1, 0, 1);
    EXPECT_FALSE(probe.observed.forces_requested);
    EXPECT_FALSE(probe.observed.stress_requested);
    EXPECT_TRUE(probe.observed.initializing_forces);
    EXPECT_TRUE(probe.observed.trial_evaluation);
}

TEST(CudaNestedRequests,
     PreparationFailureRemainsRetryableAndDoesNotLeakIntent) {
    RequestProbe probe;
    Plan<cuda::Component> plan;
    plan.kind = PhaseKind::Minimizer;
    plan.slots.push_back({slot_of(minimizer::Phase::force), {{&probe}}});
    plan.slots[0].stress_requested = false;
    cuda::Backend backend;
    backend.load(plan);
    probe.fail = true;
    EXPECT_THROW(backend.prepare_run(), std::runtime_error);
    EXPECT_TRUE(probe.observed.initializing_forces);
    EXPECT_FALSE(probe.observed.stress_requested);
    EXPECT_FALSE(backend.context().exec.initializing_forces);
    probe.fail = false;
    EXPECT_NO_THROW(backend.prepare_run());
    EXPECT_EQ(probe.calls, 2);
    backend.prepare_run();
    EXPECT_EQ(probe.calls, 2);
    backend.execute_phases(0, 0, 1);
    EXPECT_EQ(probe.calls, 3);
    EXPECT_FALSE(probe.observed.initializing_forces);
}

namespace {
class HistoryProbe : public cuda::Component, public EvaluationHistory {
  public:
    std::string name() const override { return "history_probe"; }
    void launch(const cuda::LaunchContext& ctx) override {
        auto& transaction = *ctx.runtime.exec.evaluation_transaction;
        EvaluationTransaction::Scope borrowed(transaction);
        transaction.enlist(*this);
        pending = true;
        borrowed.accept();
        EXPECT_EQ(commits, 0);
    }
    void commit_evaluation() noexcept override {
        pending = false;
        ++commits;
    }
    void discard_evaluation() noexcept override {
        pending = false;
        ++discards;
    }
    bool pending = false;
    int commits = 0, discards = 0;
};
} // namespace

TEST(CudaNestedRequests, CompletePhysicalStepOwnsHistoryCommit) {
    HistoryProbe history;
    RequestProbe trailing;
    Plan<cuda::Component> plan;
    plan.slots.push_back({0, {{&history}, {&trailing}}});
    cuda::Backend backend;
    backend.init();
    backend.load(plan);
    trailing.fail = true;
    EXPECT_THROW(backend.step(0), std::runtime_error);
    EXPECT_EQ(history.commits, 0);
    EXPECT_EQ(history.discards, 1);
    EXPECT_FALSE(history.pending);
    EXPECT_FALSE(backend.evaluation_transaction().active());
    trailing.fail = false;
    EXPECT_NO_THROW(backend.step(1));
    EXPECT_EQ(history.commits, 1);
    EXPECT_FALSE(history.pending);
    EXPECT_FALSE(backend.evaluation_transaction().active());
    backend.shutdown();
}
