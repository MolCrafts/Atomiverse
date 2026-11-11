#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

namespace {
using namespace ATV;

__global__ void publish_probe_result(cuda::Frame frame, int* workspace) {
    *workspace = 1;
    frame.scalars->energy = 7.0;
    frame.atoms.x[0] = 2.0;
    frame.atoms.charge[0] = 0.5;
    frame.atoms.fx[0] = 0;
    frame.atoms.fy[0] = 0;
    frame.atoms.fz[0] = 0;
}

class FailingEvaluator : public cuda::Component {
  public:
    explicit FailingEvaluator(std::string label = "failing_evaluator")
        : label_(std::move(label)) {}
    std::string name() const override { return label_; }
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
    void build(const cuda::BuildContext& ctx) override {
        ++builds;
        runtime = &ctx.runtime;
        resident = &ctx.frame;
        EXPECT_EQ(workspace, nullptr);
        ATV_CUDA_CHECK(cudaMalloc(&workspace, sizeof(int)));
        if (fail_build)
            throw std::runtime_error("evaluate build failure");
    }
    void launch(const cuda::LaunchContext& ctx) override {
        ++launches;
        publish_probe_result<<<1, 1, 0,
                               static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame, workspace);
        ATV_CUDA_CHECK(cudaGetLastError());
        if (fail_launch)
            throw std::runtime_error("evaluate launch failure");
    }
    void destroy(const cuda::DestroyContext&) override {
        ++destroys;
        ATV_CUDA_CHECK(cudaFree(workspace));
        workspace = nullptr;
        if (fail_destroy)
            throw std::runtime_error("evaluate destroy failure");
    }
    bool fail_build = false, fail_launch = false, fail_destroy = false;
    int builds = 0, launches = 0, destroys = 0;
    int* workspace = nullptr;
    cuda::Context* runtime = nullptr;
    cuda::Frame* resident = nullptr;

  private:
    std::string label_;
};

class CudaEvaluateRecovery : public ::testing::TestWithParam<bool> {};

TEST_P(CudaEvaluateRecovery, FailedPassDiscardsResultsAndSameObjectRetries) {
    FailingEvaluator component;
    cuda::Evaluate evaluator;
    evaluator.set_evaluator_ref(component);
    evaluator.build();
    evaluator.enable_timing();
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.meta.get(kEnergy) = 99.0;
    frame.span(kAtom_x)[0] = 1.0;
    frame.span(kAtom_charge)[0] = -0.25;
    frame.span(kAtom_fx)[0] = 10.0;
    frame.span(kAtom_fy)[0] = 11.0;
    frame.span(kAtom_fz)[0] = 12.0;
    component.fail_build = GetParam();
    component.fail_launch = !GetParam();

    EXPECT_THROW(evaluator.run(frame), std::runtime_error);
    EXPECT_EQ(component.builds, 1);
    EXPECT_EQ(component.launches, GetParam() ? 0 : 1);
    EXPECT_EQ(component.destroys, 1);
    EXPECT_EQ(component.workspace, nullptr);
    ASSERT_NE(component.runtime, nullptr);
    ASSERT_NE(component.resident, nullptr);
    EXPECT_EQ(component.runtime->stream, nullptr);
    EXPECT_EQ(component.runtime->defer_stream, nullptr);
    EXPECT_EQ(component.runtime->exec.backend, nullptr);
    EXPECT_EQ(component.runtime->host_box.resident, nullptr);
    EXPECT_EQ(component.resident->atoms.x, nullptr);
    EXPECT_EQ(component.resident->scalars, nullptr);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 99.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_charge)[0], -0.25);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 10.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fy)[0], 11.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fz)[0], 12.0);

    component.fail_build = component.fail_launch = false;
    for (int pass = 0; pass < 2; ++pass) {
        ASSERT_NO_THROW(evaluator.run(frame));
        EXPECT_EQ(component.builds, pass + 2);
        EXPECT_EQ(component.destroys, pass + 2);
        EXPECT_EQ(component.launches, pass + (GetParam() ? 1 : 2));
        EXPECT_EQ(component.workspace, nullptr);
        EXPECT_EQ(component.runtime->stream, nullptr);
        EXPECT_EQ(component.resident->scalars, nullptr);
        EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 7.0);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 2.0);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_charge)[0], 0.5);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 0.0);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fy)[0], 0.0);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fz)[0], 0.0);
    }
    EXPECT_NE(evaluator.summary().find("failing_evaluator"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(BuildOrLaunch, CudaEvaluateRecovery,
                         ::testing::Bool());

TEST(CudaEvaluateRecovery,
     DestroyFailureContinuesCleanupAndPreservesLaunchError) {
    FailingEvaluator earlier("earlier"), failing;
    cuda::Evaluate evaluator;
    evaluator.on(evaluate::Phase::evaluate).add_ref(earlier, -1);
    evaluator.set_evaluator_ref(failing);
    evaluator.build();
    evaluator.enable_timing();
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    frame.meta.get(kEnergy) = 99.0;
    failing.fail_launch = failing.fail_destroy = true;
    try {
        evaluator.run(frame);
        FAIL() << "expected evaluate launch failure";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "evaluate launch failure");
    }
    // Destroy runs in reverse order: the failing component must not prevent
    // cleanup of the component acquired before it or the Backend itself.
    for (const auto* component : {&earlier, &failing}) {
        EXPECT_EQ(component->destroys, 1);
        EXPECT_EQ(component->workspace, nullptr);
        ASSERT_NE(component->runtime, nullptr);
        EXPECT_EQ(component->runtime->stream, nullptr);
        EXPECT_EQ(component->runtime->defer_stream, nullptr);
        EXPECT_EQ(component->runtime->host_box.resident, nullptr);
        EXPECT_EQ(component->runtime->exec.backend, nullptr);
        EXPECT_EQ(component->resident->scalars, nullptr);
    }
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 99.0);
    failing.fail_launch = failing.fail_destroy = false;
    ASSERT_NO_THROW(evaluator.run(frame));
    for (const auto* component : {&earlier, &failing}) {
        EXPECT_EQ(component->builds, 2);
        EXPECT_EQ(component->destroys, 2);
        EXPECT_EQ(component->workspace, nullptr);
    }
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 7.0);
    EXPECT_NE(evaluator.summary().find("failing_evaluator"), std::string::npos);
}
} // namespace
