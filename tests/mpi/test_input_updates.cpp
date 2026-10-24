/// Collective Runtime input decisions; no physical solver or time-step loop.
#include <gtest/gtest.h>
#include <mpi.h>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/driver/runtime.h"

using namespace ATV;

namespace {
class InputCache : public cpu::Component {
  public:
    bool reject_identity = false;
    bool throw_preflight = false;
    bool throw_invalidation = false;
    int invalidations = 0;
    InputChange seen = InputChange::None;

    std::string name() const override { return "mpi_input_cache"; }
    bool supports_input_changes(InputChange changes) const override {
        if (throw_preflight)
            throw std::runtime_error("local preflight failure");
        return !reject_identity
               || !has_input_change(changes, InputChange::Identity);
    }
    void invalidate_inputs(InputChange changes,
                           const cpu::BuildContext&) override {
        ++invalidations;
        seen = changes;
        if (throw_invalidation)
            throw std::runtime_error("local cache failure");
    }
    void launch(const cpu::LaunchContext&) override {}
};

class MpiInputUpdates : public ::testing::Test {
  protected:
    cpu::Frame frame{1};
    InputCache first, second;
    Plan<cpu::Component> plan;
    cpu::md::mpi::DomainDecomposition dd;
    cpu::Runtime runtime;
    int rank = 0;

    void SetUp() override {
        auto& context = runtime.context();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        context.mpi_rank = rank;
        MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
        dd.cart_comm = MPI_COMM_WORLD;
        context.md.dd = &dd;
        plan.lifecycle_only = {&first, &second};
        runtime.load(plan);
        runtime.set_frame(frame);
        runtime.checkpoint_inputs();
    }
    void TearDown() override {
        runtime.context().md.dd = nullptr;
        dd.cart_comm = MPI_COMM_NULL;
    }
};
} // namespace

TEST_F(MpiInputUpdates, OwnedEditsCombineBeforeAnyRankReturns) {
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::None);
    if (rank == 0)
        frame.span(kAtom_x)[0] = 1;
    else
        frame.span(kAtom_vx)[0] = .5;
    const auto expected = InputChange::Geometry | InputChange::Velocity;
    EXPECT_EQ(runtime.refresh_external_inputs(), expected);
    EXPECT_EQ(first.seen, expected);
    EXPECT_EQ(first.invalidations, 1);
    runtime.checkpoint_inputs();
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::None);
}

TEST_F(MpiInputUpdates, PeerIdentityRejectionIsCollectiveAndPrecedesHooks) {
    if (rank == 0)
        frame.span(kAtom_type)[0] = 8;
    second.reject_identity = rank == 1;
    EXPECT_THROW(runtime.refresh_external_inputs(), std::runtime_error);
    EXPECT_EQ(first.invalidations, 0);
    EXPECT_EQ(second.invalidations, 0);
    second.reject_identity = false;
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::Identity);
    EXPECT_EQ(first.invalidations, 1);
}

TEST_F(MpiInputUpdates, PeerPreflightExceptionIsCollective) {
    if (rank == 0)
        frame.span(kAtom_x)[0] = 1;
    second.throw_preflight = rank == 1;
    EXPECT_THROW(runtime.refresh_external_inputs(), std::runtime_error);
    EXPECT_EQ(first.invalidations, 0);
    second.throw_preflight = false;
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::Geometry);
}

TEST_F(MpiInputUpdates, PeerMalformedInputFailsBeforePreflightEverywhere) {
    const auto saved_meta = frame.meta;
    if (rank == 0)
        frame.meta = cpu::MetaMap{};
    EXPECT_THROW(runtime.refresh_external_inputs(), std::runtime_error);
    EXPECT_EQ(first.invalidations, 0);
    frame.meta = saved_meta;
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::None);
}

TEST_F(MpiInputUpdates, PeerHookFailureRetainsPendingChangesAfterHostRestore) {
    const auto original_type = frame.span(kAtom_type)[0];
    if (rank == 0)
        frame.span(kAtom_type)[0] = 8;
    second.throw_invalidation = rank == 0;
    EXPECT_THROW(runtime.refresh_external_inputs(), std::runtime_error);
    EXPECT_EQ(first.invalidations, 1);
    frame.span(kAtom_type)[0] = original_type;
    second.throw_invalidation = false;
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::Identity);
    EXPECT_EQ(first.invalidations, 2);
    runtime.checkpoint_inputs();
    EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::None);
}

TEST_F(MpiInputUpdates, IndependentReplicaNeedsNoWorldParticipation) {
    runtime.context().exec.serial_on_replica = true;
    dd.cart_comm = MPI_COMM_NULL;
    // Only rank zero calls refresh. A mistaken WORLD reduction would conflict
    // with the other ranks' barrier instead of completing this local operation.
    if (rank == 0) {
        frame.span(kAtom_x)[0] = 1;
        EXPECT_EQ(runtime.refresh_external_inputs(), InputChange::Geometry);
        EXPECT_EQ(first.invalidations, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
