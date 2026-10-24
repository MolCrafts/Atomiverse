#include "atomiverse/cuda/driver/driver.cuh"
#include <functional>
#include <gtest/gtest.h>

namespace {
using namespace ATV;

class BorrowedProbe : public cuda::Component {
  public:
    explicit BorrowedProbe(std::string label) : label_(std::move(label)) {}
    std::string name() const override { return label_; }
    int frequency() const override { return every; }
    void build(const cuda::BuildContext& ctx) override {
        ++builds;
        runtime = &ctx.runtime;
        frame = &ctx.frame;
        if (fail_build)
            throw std::runtime_error("probe build failure");
    }
    void launch(const cuda::LaunchContext& ctx) override {
        ++launches;
        seen = ctx.runtime.exec;
        force_request = ctx.forces_requested;
        if (callback)
            callback();
        if (fail_launch)
            throw std::runtime_error("probe launch failure");
    }
    void destroy(const cuda::DestroyContext&) override { ++destroys; }
    int builds = 0, launches = 0, destroys = 0, every = 0;
    bool fail_build = false, fail_launch = false, force_request = true;
    cuda::Context* runtime = nullptr;
    cuda::Frame* frame = nullptr;
    cuda::ExecServices seen{};
    std::function<void()> callback;

  private:
    std::string label_;
};

class BorrowedNode : public cuda::Driver<evaluate::Phase>,
                     public cuda::DriverNode {
  public:
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override {
        build_borrowed_(ctx, frame);
    }
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int) override {
        step_borrowed_(ctx, frame, iteration);
        ++iteration;
        return 1;
    }
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override {
        destroy_borrowed_(ctx, frame);
    }
    int iteration = 0;

  protected:
    void validate_components() const override {}
};

// Declare after probes/nodes: cleanup must run while their borrowed addresses live.
struct Session {
    cuda::Backend backend;
    Session() { backend.init(); }
    ~Session() { backend.shutdown(); }
};

TEST(CudaBorrowedExecution, SharesLifecycleAndPreservesParentSchedule) {
    BorrowedProbe shared("shared"), periodic("periodic"), added("added");
    periodic.every = 2;
    BorrowedNode child;
    child.on(evaluate::Phase::evaluate).add_ref(shared);
    child.on(evaluate::Phase::evaluate).add_ref(periodic);
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<BorrowedNode>{}, &child, 1, false, false);
    Session session;
    auto& backend = session.backend;
    Plan<cuda::Component> parent;
    parent.slots.resize(1);
    parent.slots[0].ops = {{&shared}, {slot.get()}};
    backend.load(parent);
    backend.build_components();
    EXPECT_EQ(shared.builds, 1);
    EXPECT_EQ(periodic.builds, 1);
    EXPECT_EQ(shared.runtime, &backend.context());
    EXPECT_EQ(shared.frame, &backend.frame());
    child.build_on(backend.context(), backend.frame());
    EXPECT_EQ(shared.builds, 1);

    backend.context().exec.forces_requested = false;
    backend.context().exec.trial_evaluation = true;
    backend.execute_phases(0, 0, 1);
    EXPECT_EQ(shared.launches, 2);
    EXPECT_EQ(periodic.launches, 1);
    EXPECT_FALSE(shared.force_request);
    EXPECT_TRUE(shared.seen.trial_evaluation);
    EXPECT_EQ(shared.seen.backend, &backend);

    child.on(evaluate::Phase::evaluate).add_ref(added);
    child.build_on(backend.context(), backend.frame());
    EXPECT_EQ(added.builds, 1);
    backend.execute_phases(1, 0, 1);
    EXPECT_EQ(shared.launches,
              4); // Parent plan was not replaced by child refresh.
    EXPECT_EQ(periodic.launches, 1);
    EXPECT_EQ(added.launches, 1);
    child.unfix("added");
    EXPECT_EQ(added.destroys, 1);
    backend
        .destroy_components(); // Reentrant child release through the same ledger.
    EXPECT_EQ(shared.destroys, 1);
    EXPECT_EQ(periodic.destroys, 1);
    EXPECT_EQ(added.destroys, 1);
    EXPECT_NO_THROW(child.destroy_on(backend.context(), backend.frame()));
}

TEST(CudaBorrowedExecution, FailedBuildAndNewMountBatchRollBackAndRetry) {
    BorrowedProbe stable("stable"), first("first"), failing("failing");
    BorrowedNode child;
    child.on(evaluate::Phase::evaluate).add_ref(stable);
    Session session;
    auto& ctx = session.backend.context();
    auto& frame = session.backend.frame();
    stable.fail_build = true;
    EXPECT_THROW(child.build_on(ctx, frame), std::runtime_error);
    EXPECT_EQ(stable.destroys, 1);
    EXPECT_NO_THROW(child.destroy_on(ctx, frame));
    stable.fail_build = false;
    child.build_on(ctx, frame);

    child.on(evaluate::Phase::evaluate).add_ref(first);
    child.on(evaluate::Phase::evaluate).add_ref(failing);
    failing.fail_build = true;
    EXPECT_THROW(child.build_on(ctx, frame), std::runtime_error);
    EXPECT_EQ(stable.builds, 2);
    EXPECT_EQ(stable.destroys, 1);
    EXPECT_EQ(first.destroys, 1);
    EXPECT_EQ(failing.destroys, 1);
    failing.fail_build = false;
    child.build_on(ctx, frame);
    EXPECT_EQ(first.builds, 2);
    EXPECT_EQ(failing.builds, 2);
    child.run_nested_erased(ctx, frame, 1);
    child.destroy_on(ctx, frame);
    EXPECT_EQ(stable.destroys, 2);
    EXPECT_EQ(first.destroys, 2);
    EXPECT_EQ(failing.destroys, 2);
}

TEST(CudaBorrowedExecution,
     ExceptionRestoresMutationLocksAndRejectsForeignOwner) {
    BorrowedProbe probe("probe"), added("added");
    BorrowedNode child;
    child.on(evaluate::Phase::evaluate).add_ref(probe);
    Session session;
    auto& ctx = session.backend.context();
    auto& frame = session.backend.frame();
    cuda::Frame other_frame;
    EXPECT_THROW(child.build_on(ctx, other_frame), std::invalid_argument);
    child.build_on(ctx, frame);
    EXPECT_THROW({ BorrowedNode moved(std::move(child)); }, std::logic_error);
    Session other;
    EXPECT_THROW(child.build_on(other.backend.context(), other.backend.frame()),
                 std::logic_error);
    EXPECT_THROW(
        child.destroy_on(other.backend.context(), other.backend.frame()),
        std::logic_error);
    probe.callback = [&] {
        EXPECT_THROW(child.unfix("probe"), std::runtime_error);
        EXPECT_THROW(child.on(evaluate::Phase::evaluate).add_ref(added),
                     std::logic_error);
    };
    probe.fail_launch = true;
    EXPECT_THROW(child.run_nested_erased(ctx, frame, 1), std::runtime_error);
    EXPECT_FALSE(*session.backend.mutation_lock());
    probe.fail_launch = false;
    EXPECT_NO_THROW(child.run_nested_erased(ctx, frame, 1));
    child.unfix("probe");
    EXPECT_EQ(probe.destroys, 1);
    child.destroy_on(ctx, frame);
}

TEST(CudaBorrowedExecution, NestedTimingIsIncludedOnceInParentEntry) {
    BorrowedProbe probe("inner_probe");
    BorrowedNode child;
    child.on(evaluate::Phase::evaluate).add_ref(probe);
    auto slot = backend_traits<cuda::device_tag>::nested_slot(
        std::unique_ptr<BorrowedNode>{}, &child, 1, false, false);
    cuda::Timer timer;
    timer.init();
    Session session;
    session.backend.set_timer(&timer);
    Plan<cuda::Component> parent;
    parent.slots.resize(1);
    parent.slots[0].ops = {{slot.get()}};
    session.backend.load(parent);
    session.backend.build_components();
    probe.fail_launch = true;
    EXPECT_THROW(session.backend.execute_phases(0, 0, 1), std::runtime_error);
    probe.fail_launch = false;
    session.backend.execute_phases(1, 0, 1);
    timer.step_done();
    const auto summary = timer.summary({"evaluate"});
    EXPECT_NE(summary.find("nested_driver"), std::string::npos);
    EXPECT_EQ(summary.find("inner_probe"), std::string::npos);
    session.backend.destroy_components();
    timer.shutdown();
}
} // namespace
