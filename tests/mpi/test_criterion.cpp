#include "atomiverse/cpu/driver/criteria.h"
#include <gtest/gtest.h>
#include <limits>
#include <mpi.h>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/driver/driver.h"

using namespace ATV;

namespace {
class PollDriver : public cpu::Driver<ATV::md::Phase> {
  public:
    void validate_components() const override {}
    using cpu::Driver<ATV::md::Phase>::poll_criteria_;
    using cpu::Driver<ATV::md::Phase>::mutable_outer_status;
};

class Requirement : public cpu::Criterion {
  public:
    Verdict vote = Verdict::Continue;
    bool contributes_to_convergence() const override { return true; }
    std::string name() const override { return "requirement"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override { return vote; }
};

class Guard : public cpu::Criterion {
  public:
    Verdict vote = Verdict::Continue;
    std::string name() const override { return "guard"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override { return vote; }
};

class MpiCriterion : public ::testing::Test {
  protected:
    cpu::Context context;
    cpu::md::mpi::DomainDecomposition dd;
    cpu::Frame frame{1};

    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &context.mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &context.mpi_size);
        dd.cart_comm = MPI_COMM_WORLD;
        context.md.dd = &dd;
    }
    void TearDown() override { dd.cart_comm = MPI_COMM_NULL; }
};
} // namespace

TEST_F(MpiCriterion, SuccessRequiresEveryRequirementOnEveryRank) {
    Requirement first, second;
    Guard guard;
    PollDriver driver;
    driver.add_criterion_ref(first);
    driver.add_criterion_ref(second);
    driver.add_criterion_ref(guard);
    first.vote = Verdict::Converged;
    second.vote =
        context.mpi_rank == 0 ? Verdict::Converged : Verdict::Continue;
    EXPECT_EQ(driver.poll_criteria_(0, frame, context), Verdict::Continue);
    EXPECT_FALSE(driver.mutable_outer_status().converged);
    second.vote = Verdict::Converged;
    first.vote = context.mpi_rank == 0 ? Verdict::Continue : Verdict::Converged;
    EXPECT_EQ(driver.poll_criteria_(1, frame, context), Verdict::Continue);
    first.vote = Verdict::Converged;
    EXPECT_EQ(driver.poll_criteria_(2, frame, context), Verdict::Converged);
    EXPECT_TRUE(driver.mutable_outer_status().converged);
}

TEST_F(MpiCriterion, StopOnOneRankPreventsSuccessEverywhere) {
    for (bool stop_from_guard : {false, true}) {
        Requirement requirement;
        Guard guard;
        PollDriver driver;
        driver.add_criterion_ref(requirement);
        driver.add_criterion_ref(guard);
        requirement.vote = Verdict::Converged;
        EXPECT_EQ(driver.poll_criteria_(0, frame, context), Verdict::Converged);
        if (context.mpi_rank == 0) {
            if (stop_from_guard)
                guard.vote = Verdict::Stop;
            else
                requirement.vote = Verdict::Stop;
        }
        EXPECT_EQ(driver.poll_criteria_(1, frame, context), Verdict::Stop);
        if (context.mpi_rank != 0)
            EXPECT_EQ(driver.mutable_outer_status().reason,
                      "criterion stop on another rank");
        EXPECT_FALSE(driver.mutable_outer_status().converged);
        EXPECT_TRUE(driver.mutable_outer_status().stop_requested);
    }
}

TEST_F(MpiCriterion, GuardOnlyLoopKeepsRunningOnEveryRank) {
    Guard guard;
    PollDriver driver;
    driver.add_criterion_ref(guard);
    EXPECT_EQ(driver.poll_criteria_(0, frame, context), Verdict::Continue);
}

TEST_F(MpiCriterion, UndeclaredSuccessOnOneRankFailsEverywhere) {
    Guard guard;
    Requirement requirement;
    PollDriver driver;
    driver.add_criterion_ref(guard);
    driver.add_criterion_ref(requirement);
    requirement.vote = Verdict::Converged;
    if (context.mpi_rank == 0)
        guard.vote = Verdict::Converged;
    EXPECT_THROW(driver.poll_criteria_(0, frame, context), std::logic_error);
    EXPECT_FALSE(driver.mutable_outer_status().converged);
}

TEST_F(MpiCriterion, NonfiniteStressOnOneRankStopsEverywhere) {
    frame.box = cpu::Box::cubic(10);
    frame.meta.get(kStress) = {};
    if (context.mpi_rank == 0)
        frame.meta.get(kStress)[0] = std::numeric_limits<double>::quiet_NaN();
    for (bool pressure : {false, true}) {
        PollDriver driver;
        if (pressure)
            driver.add_criterion(cpu::PressureConverged{});
        else
            driver.add_criterion(cpu::StressConverged{});
        EXPECT_EQ(driver.poll_criteria_(0, frame, context), Verdict::Stop);
        EXPECT_FALSE(driver.mutable_outer_status().converged);
    }
}
