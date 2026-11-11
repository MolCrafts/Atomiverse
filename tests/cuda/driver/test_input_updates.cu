#include "atomiverse/cuda/driver/backend_traits.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <gtest/gtest.h>

namespace {
using namespace ATV;
class InputProbe : public cuda::Component {
  public:
    std::string name() const override { return "input_probe"; }
    void build(const cuda::BuildContext&) override { ++builds; }
    void launch(const cuda::LaunchContext&) override { ++launches; }
    void destroy(const cuda::DestroyContext&) override { ++destroys; }
    bool supports_input_changes(InputChange changes) const override {
        return identity || Component::supports_input_changes(changes);
    }
    void invalidate_inputs(InputChange changes,
                           const cuda::BuildContext&) override {
        ++invalidations;
        seen = changes;
        if (fail) {
            fail = false;
            throw std::runtime_error("input cache refresh failure");
        }
    }
    bool identity = false, fail = false;
    int builds = 0, launches = 0, destroys = 0, invalidations = 0;
    InputChange seen = InputChange::None;
};
Plan<cuda::Component> input_plan(InputProbe& probe) {
    Plan<cuda::Component> plan;
    plan.kind = PhaseKind::MD;
    plan.slots.push_back({slot_of(md::Phase::force), {{&probe}}});
    return plan;
}
} // namespace

TEST(CudaInputUpdates, NumericEditsReuseStorageLifecycleAndInvalidateForce) {
    cpu::Frame frame(1);
    frame.span(kAtom_mass)[0] = 1;
    frame.span(kAtom_type)[0] = 1;
    InputProbe probe;
    auto plan = input_plan(probe);
    cuda::Backend backend;
    backend.init();
    backend.load(plan);
    backend.materialize(frame);
    backend.build_components();
    backend.prepare_run();
    backend.retrieve(frame);
    auto* positions = backend.frame().atoms.x;
    auto* stream = backend.context().stream;
    frame.span(kAtom_x)[0] = 0.375;
    frame.span(kAtom_vx)[0] = 0.125;
    backend.refresh_inputs(frame);
    EXPECT_EQ(backend.frame().atoms.x, positions);
    EXPECT_EQ(backend.context().stream, stream);
    EXPECT_EQ(probe.builds, 1);
    EXPECT_EQ(probe.destroys, 0);
    EXPECT_EQ(probe.invalidations, 1);
    EXPECT_TRUE(has_input_change(probe.seen, InputChange::Geometry));
    EXPECT_TRUE(has_input_change(probe.seen, InputChange::Velocity));
    backend.prepare_run();
    EXPECT_EQ(probe.launches, 2);
    backend.retrieve(frame);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 0.375);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_vx)[0], 0.125);
    backend.refresh_inputs(frame);
    EXPECT_EQ(probe.invalidations, 1);
    // Outputs do not invalidate a resident input checkpoint.
    frame.span(kAtom_fx)[0] = 7;
    backend.refresh_inputs(frame);
    EXPECT_EQ(probe.invalidations, 1);
    backend.shutdown();
    EXPECT_EQ(probe.destroys, 1);
}

TEST(CudaInputUpdates, IdentityPreflightRejectsBeforeAnyMutation) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    InputProbe probe;
    auto plan = input_plan(probe);
    cuda::Backend backend;
    backend.init();
    backend.load(plan);
    backend.materialize(frame);
    backend.build_components();
    backend.retrieve(frame);
    frame.span(kAtom_type)[0] = 8;
    EXPECT_THROW(backend.refresh_inputs(frame), std::invalid_argument);
    EXPECT_EQ(probe.invalidations, 0);
    int type = 0;
    ASSERT_EQ(cudaMemcpy(&type, backend.frame().atoms.type, sizeof(type),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(type, 1);
    EXPECT_EQ(frame.span(kAtom_type)[0], 8);
    frame.span(kAtom_type)[0] = 1;
    EXPECT_NO_THROW(backend.refresh_inputs(frame));
    backend.shutdown();
}

TEST(CudaInputUpdates, PopulationChangePreservesSupportedComponents) {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    InputProbe probe;
    probe.identity = true;
    auto plan = input_plan(probe);
    cuda::Backend backend;
    backend.init();
    backend.load(plan);
    backend.materialize(frame);
    backend.build_components();
    backend.retrieve(frame);
    auto* stream = backend.context().stream;
    frame = cpu::Frame(2);
    frame.span(kAtom_type)[0] = 6;
    frame.span(kAtom_type)[1] = 8;
    backend.refresh_inputs(frame);
    EXPECT_EQ(backend.frame().n_atoms(), 2);
    EXPECT_EQ(backend.context().stream, stream);
    EXPECT_EQ(probe.builds, 1);
    EXPECT_EQ(probe.destroys, 0);
    EXPECT_TRUE(has_input_change(probe.seen, InputChange::Identity));
    backend.retrieve(frame);
    backend.shutdown();
    EXPECT_EQ(probe.destroys, 1);
}

TEST(CudaInputUpdates, FailedRefreshRestoresCheckpointInputsOnRetry) {
    cpu::Frame frame(1);
    frame.span(kAtom_x)[0] = 0.25;
    InputProbe probe;
    auto plan = input_plan(probe);
    cuda::Backend backend;
    backend.init();
    backend.load(plan);
    backend.materialize(frame);
    backend.build_components();
    backend.retrieve(frame);
    frame.span(kAtom_x)[0] = 1.5;
    probe.fail = true;
    EXPECT_THROW(backend.refresh_inputs(frame), std::runtime_error);
    // The user corrects input to the exact previous checkpoint. A pure
    // snapshot comparison would miss the partially refreshed device state.
    frame.span(kAtom_x)[0] = 0.25;
    cuda::ForceFault fault{};
    fault.kinds = cuda::ForceFault::kConversion;
    ASSERT_EQ(cudaMemcpy(backend.frame().force_fault, &fault, sizeof(fault),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_NO_THROW(backend.refresh_inputs(frame));
    backend.retrieve(frame);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 0.25);
    EXPECT_EQ(probe.invalidations, 2);
    backend.shutdown();
}

namespace {
class InputNode : public cuda::DriverNode {
  public:
    void build_on(cuda::Context&, cuda::Frame&) override {}
    void destroy_on(cuda::Context&, cuda::Frame&) override {}
    int run_nested_erased(cuda::Context&, cuda::Frame&, int) override {
        return 0;
    }
    bool supports_input_changes(InputChange changes) const override {
        return !has_input_change(changes, InputChange::Charge);
    }
    void invalidate_inputs(InputChange changes, cuda::Context& ctx,
                           cuda::Frame& frame) override {
        seen = changes;
        runtime = &ctx;
        device = &frame;
        ++invalidations;
    }
    InputChange seen = InputChange::None;
    cuda::Context* runtime = nullptr;
    cuda::Frame* device = nullptr;
    int invalidations = 0;
};
} // namespace
TEST(CudaInputUpdates, NestedSlotForwardsPreflightAndInvalidation) {
    InputNode node;
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<InputNode>{}, &node, -1, false, false);
    cuda::Context ctx;
    cuda::Frame frame;
    EXPECT_FALSE(slot->supports_input_changes(InputChange::Charge));
    EXPECT_TRUE(slot->supports_input_changes(InputChange::Identity));
    slot->invalidate_inputs(InputChange::Identity, {ctx, frame});
    EXPECT_EQ(node.invalidations, 1);
    EXPECT_EQ(node.seen, InputChange::Identity);
    EXPECT_EQ(node.runtime, &ctx);
    EXPECT_EQ(node.device, &frame);
}
