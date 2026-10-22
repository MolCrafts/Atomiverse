#include "atomiverse/cuda/component/optimizer.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include <gtest/gtest.h>

using namespace ATV;
namespace {
class Probe : public cuda::Driver<minimizer::Phase> {
  public:
    Verdict poll() { return poll_criteria_(0, frame, context); }
    void validate_components() const override {}
    cuda::Frame frame{};
    cuda::Context context{};
};
class Requirement : public cuda::Criterion {
  public:
    explicit Requirement(Verdict v) : value(v) {}
    std::string name() const override { return "requirement"; }
    bool contributes_to_convergence() const override { return true; }
    Verdict verdict(const cuda::LoopState&) override { return value; }
    Verdict value;
};
class BadGuard : public cuda::Criterion {
  public:
    std::string name() const override { return "bad_guard"; }
    Verdict verdict(const cuda::LoopState&) override {
        return Verdict::Converged;
    }
};
} // namespace
TEST(CudaMinimizeContract, RequiresAllCriteria) {
    Probe p;
    p.add_criterion(Requirement{Verdict::Converged});
    p.add_criterion(Requirement{Verdict::Continue});
    EXPECT_EQ(p.poll(), Verdict::Continue);
    EXPECT_FALSE(p.converged());
}
TEST(CudaMinimizeContract, GuardsDoNotBlockSuccess) {
    Probe p;
    p.add_criterion(Requirement{Verdict::Converged});
    p.add_criterion(cuda::StopOnNaN{});
    EXPECT_EQ(p.poll(), Verdict::Converged);
}
TEST(CudaMinimizeContract, StopOverridesSuccess) {
    Probe p;
    p.add_criterion(Requirement{Verdict::Converged});
    p.add_criterion(Requirement{Verdict::Stop});
    EXPECT_EQ(p.poll(), Verdict::Stop);
    EXPECT_FALSE(p.converged());
}
TEST(CudaMinimizeContract, GuardCannotDeclareSuccess) {
    Probe p;
    p.add_criterion(BadGuard{});
    EXPECT_THROW(p.poll(), std::logic_error);
}
TEST(CudaMinimizeContract, NoRequirementsCannotConverge) {
    Probe p;
    p.add_criterion(cuda::StopOnNaN{});
    EXPECT_EQ(p.poll(), Verdict::Continue);
}
TEST(CudaMinimizeContract, NegativeBudgetRejectedBeforeSetup) {
    cuda::minimizer::Minimize min;
    cpu::Frame frame;
    EXPECT_THROW(min.run(frame, -1), std::invalid_argument);
}

namespace {
class StubOptimizer : public cuda::Optimizer {
  public:
    std::string name() const override { return "stub_optimizer"; }
    void launch(const cuda::LaunchContext&) override {}
};
class StubForce : public cuda::Component {
  public:
    std::string name() const override { return "stub_force"; }
    bool provides_force() const override { return true; }
    void launch(const cuda::LaunchContext& ctx) override {
        ++calls;
        initializing = ctx.initializing_forces;
    }
    int calls = 0;
    bool initializing = false;
};
} // namespace
TEST(CudaMinimizeContract, GuardOnlyStillInstallsConvergenceRequirement) {
    cuda::minimizer::Minimize min;
    StubOptimizer optimizer;
    StubForce force;
    min.on(minimizer::Phase::search_direction).add_ref(optimizer);
    min.on(minimizer::Phase::force).add_ref(force);
    min.add_criterion(cuda::StopOnNaN{});
    min.build();
    EXPECT_TRUE(min.has_convergence_criterion());
}
TEST(CudaMinimizeContract, ZeroBudgetPreparesForcesWithoutOptimizerLaunch) {
    cuda::minimizer::Minimize min;
    StubOptimizer optimizer;
    StubForce force;
    min.on(minimizer::Phase::search_direction).add_ref(optimizer);
    min.on(minimizer::Phase::force).add_ref(force);
    min.build();
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    EXPECT_EQ(min.run(frame, 0), 0);
    EXPECT_EQ(force.calls, 1);
    EXPECT_TRUE(force.initializing);
}

namespace {
class CountedCriterion : public cuda::Criterion {
  public:
    std::string name() const override { return "counted_criterion"; }
    void build(const cuda::BuildContext&) override { ++builds; }
    void destroy(const cuda::DestroyContext&) override { ++destroys; }
    Verdict verdict(const cuda::LoopState&) override {
        return Verdict::Continue;
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {make_binding(minimizer::Phase::convergence_check),
                make_binding(minimizer::Phase::end_of_step)};
    }
    int builds = 0, destroys = 0;
};
} // namespace
TEST(CudaMinimizeContract, CriterionUsesSingleInstanceLifecycle) {
    Probe driver;
    CountedCriterion criterion;
    driver.add_criterion_ref(criterion);
    driver.build();
    cuda::Backend backend;
    backend.init();
    backend.load(driver.plan());
    backend.build_components();
    EXPECT_EQ(criterion.builds, 1);
    backend.destroy_components();
    EXPECT_EQ(criterion.destroys, 1);
    backend.shutdown();
    driver.unfix("counted_criterion");
    EXPECT_FALSE(driver.has_criterion());
    EXPECT_TRUE(driver.plan().components().empty());
}
TEST(CudaMinimizeContract, UnscheduledCriterionCanBeUnmountedBeforeBuild) {
    Probe driver;
    driver.add_criterion(cuda::StopOnNaN{});
    driver.unfix("stop_on_nan");
    EXPECT_FALSE(driver.has_criterion());
    EXPECT_TRUE(driver.plan().components().empty());
}

TEST(CudaMinimizeContract, WideBudgetDoesNotNarrowBeforeStopping) {
    cuda::minimizer::Minimize min;
    StubOptimizer optimizer;
    StubForce force;
    min.on(minimizer::Phase::search_direction).add_ref(optimizer);
    min.on(minimizer::Phase::force).add_ref(force);
    min.add_criterion(Requirement{Verdict::Stop});
    min.build();
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    EXPECT_EQ(min.run(frame, BigInt{1} << 40), 1);
    EXPECT_EQ(force.calls, 2); // Preparation plus one iteration.
}

namespace {
class MutatingCriterion : public cuda::Criterion {
  public:
    explicit MutatingCriterion(Probe& p) : driver(p) {}
    std::string name() const override { return "mutating_criterion"; }
    Verdict verdict(const cuda::LoopState&) override {
        driver.add_criterion(cuda::StopOnNaN{});
        return Verdict::Continue;
    }
    Probe& driver;
};
} // namespace
TEST(CudaMinimizeContract, PollRejectsStructuralMutation) {
    Probe driver;
    driver.add_criterion(MutatingCriterion{driver});
    EXPECT_THROW(driver.poll(), std::logic_error);
    EXPECT_NO_THROW(driver.add_criterion(cuda::StopOnNaN{}));
}

TEST(CudaMinimizeContract, FirstArmijoTrialUsesPreparedOriginEnergy) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.span(kAtom_x)[0] = 1.0;
    frame.span(kAtom_fx)[0] = 1.0;
    frame.span(kAtom_mass)[0] = 1.0;
    frame.meta.get(kEnergy) = 100.0;
    cuda::Backend backend;
    backend.init();
    backend.materialize(frame);
    // materialize zeroes device forces; the origin force goes on explicitly.
    double* d_force = nullptr;
    ASSERT_EQ(cudaMalloc(&d_force, 3 * sizeof(double)), cudaSuccess);
    const double host_force[3] = {1.0, 0.0, 0.0};
    ASSERT_EQ(cudaMemcpy(d_force, host_force, sizeof(host_force),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    cuda::store_forces(backend.frame(), d_force, d_force + 1, d_force + 2, 1,
                       backend.context().stream);
    cuda::minimizer::LBFGS optimizer;
    optimizer.line_search();
    optimizer.build({backend.context(), backend.frame()});
    auto& ctx = backend.context();
    auto& device = backend.frame();
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    optimizer.launch(
        {ctx, device, slot_of(minimizer::Phase::search_direction), 0});
    // A downhill trial from E=100 to E=99 must be accepted. Comparing E=99
    // against a fictitious origin E=0 would instead contract the first move.
    const double trial_energy = 99.0;
    ASSERT_EQ(cudaMemcpyAsync(&device.scalars->energy, &trial_energy,
                              sizeof(double), cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    optimizer.launch(
        {ctx, device, slot_of(minimizer::Phase::convergence_check), 0});
    optimizer.launch(
        {ctx, device, slot_of(minimizer::Phase::search_direction), 1});
    backend.sync();
    backend.retrieve(frame);
    EXPECT_NEAR(frame.span(kAtom_x)[0], 1.02, 1e-7);
    optimizer.destroy({ctx, device});
    cudaFree(d_force);
    backend.shutdown();
}
