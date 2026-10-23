/// @file tests/cpu/semiempirical/test_scc_driver.cpp
/// @brief Unit: SccDriver iteration, accept-versus-mix, and scc::Phase order.

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/semiempirical/scc_method.h"

using ATV::cpu::semiempirical::SccDriver;
using ATV::cpu::semiempirical::SccOutcome;

TEST(SccDriver, PhaseOrder) {
    static_assert(
        std::is_base_of_v<ATV::cpu::Driver<ATV::scc::Phase>, SccDriver>);
    static_assert(ATV::cpu::PhaseTraits<ATV::scc::Phase>::eval_phase
                  == ATV::scc::Phase::shift);
    static_assert(ATV::cpu::PhaseTraits<ATV::scc::Phase>::neighbor_phase
                  == ATV::scc::Phase::shift);
    static_assert(ATV::cpu::PhaseTraits<ATV::scc::Phase>::end_phase
                  == ATV::scc::Phase::check);
    EXPECT_EQ(ATV::slot_count<ATV::scc::Phase>(), 5);
    EXPECT_EQ(ATV::slot_of(ATV::scc::Phase::shift), 0);
    EXPECT_EQ(ATV::slot_of(ATV::scc::Phase::diagonalize), 1);
    EXPECT_EQ(ATV::slot_of(ATV::scc::Phase::charges), 2);
    EXPECT_EQ(ATV::slot_of(ATV::scc::Phase::mix), 3);
    EXPECT_EQ(ATV::slot_of(ATV::scc::Phase::check), 4);
    EXPECT_EQ(ATV::scc::kind_of(ATV::scc::Phase{}), ATV::PhaseKind::SCF);
    const std::vector<std::string> names =
        ATV::scc::phase_names(ATV::scc::Phase{});
    ASSERT_EQ(names.size(), 5u);
    EXPECT_EQ(names[0], "shift");
    EXPECT_EQ(names[1], "diagonalize");
    EXPECT_EQ(names[2], "charges");
    EXPECT_EQ(names[3], "mix");
    EXPECT_EQ(names[4], "check");
}

TEST(SccDriver, StagesRunShiftDiagonalizeChargesThenAccept) {
    std::string order;
    SccDriver scc;
    const SccOutcome out = scc.solve(
        nullptr, /*max_iter=*/1, /*tol=*/1e-8, [&] { order += "S"; },
        [&] { order += "D"; },
        [&] {
            order += "C";
            scc.set_residual(1e-12);
        },
        [&] { order += "A"; }, [&] { order += "M"; });
    EXPECT_EQ(order, "SDCA");
    EXPECT_EQ(out.iterations, 1);
    EXPECT_DOUBLE_EQ(out.residual, 1e-12);
    EXPECT_TRUE(out.converged);
}

TEST(SccDriver, AcceptsOnceAfterTwoMixes) {
    const double script[] = {0.5, 0.05, 1e-9};
    int step = 0;
    int accepts = 0;
    int mixes = 0;
    SccDriver scc;
    const SccOutcome out = scc.solve(
        nullptr, /*max_iter=*/3, /*tol=*/1e-8, [] {}, [] {},
        [&] { scc.set_residual(script[step++]); }, [&] { ++accepts; },
        [&] { ++mixes; });
    EXPECT_EQ(out.iterations, 3);
    EXPECT_TRUE(out.converged);
    EXPECT_DOUBLE_EQ(out.residual, 1e-9);
    EXPECT_EQ(mixes, 2);
    EXPECT_EQ(accepts, 1);
}

TEST(SccDriver, StaysUnconvergedWhenResidualStaysAboveTol) {
    int accepts = 0;
    int mixes = 0;
    SccDriver scc;
    const SccOutcome out = scc.solve(
        nullptr, /*max_iter=*/3, /*tol=*/1e-8, [] {}, [] {},
        [&] { scc.set_residual(1.0); }, [&] { ++accepts; }, [&] { ++mixes; });
    EXPECT_EQ(out.iterations, 3);
    EXPECT_FALSE(out.converged);
    EXPECT_DOUBLE_EQ(out.residual, 1.0);
    EXPECT_EQ(mixes, 3);
    EXPECT_EQ(accepts, 0);
}

TEST(SccDriver, EqualResidualDoesNotAccept) {
    int accepts = 0;
    int mixes = 0;
    SccDriver scc;
    const SccOutcome out = scc.solve(
        nullptr, /*max_iter=*/1, /*tol=*/1e-8, [] {}, [] {},
        [&] { scc.set_residual(1e-8); }, [&] { ++accepts; }, [&] { ++mixes; });
    EXPECT_EQ(out.iterations, 1);
    EXPECT_FALSE(out.converged);
    EXPECT_EQ(mixes, 1);
    EXPECT_EQ(accepts, 0);
}

TEST(SccDriver, ZeroBudgetReportsZeroResidual) {
    int calls = 0;
    SccDriver scc;
    const SccOutcome out = scc.solve(
        nullptr, /*max_iter=*/0, /*tol=*/1e-8, [&] { ++calls; },
        [&] { ++calls; }, [&] { ++calls; }, [&] { ++calls; }, [&] { ++calls; });
    EXPECT_EQ(out.iterations, 0);
    EXPECT_DOUBLE_EQ(out.residual, 0.0);
    EXPECT_FALSE(out.converged);
    EXPECT_EQ(calls, 0);
}

TEST(SccDriver, NegativeBudgetThrows) {
    SccDriver scc;
    EXPECT_THROW(scc.solve(
                     nullptr, -1, 1e-8, [] {}, [] {}, [] {}, [] {}, [] {}),
                 std::invalid_argument);
}

TEST(SccDriver, InvalidResidualNeverAcceptsOrMixesAndCanRetry) {
    SccDriver scc;
    int accepts = 0, mixes = 0;
    for (double residual : {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(), -1.0}) {
        EXPECT_THROW(scc.solve(
                         nullptr, 1, 1e-8, [] {}, [] {},
                         [&] { scc.set_residual(residual); },
                         [&] { ++accepts; }, [&] { ++mixes; }),
                     std::runtime_error);
    }
    EXPECT_THROW(scc.solve(
                     nullptr, 1, 1e-8, [] {}, [] {}, [] {}, [&] { ++accepts; },
                     [&] { ++mixes; }),
                 std::runtime_error);
    EXPECT_EQ(accepts, 0);
    EXPECT_EQ(mixes, 0);
    const auto out = scc.solve(
        nullptr, 1, 1e-8, [] {}, [] {}, [&] { scc.set_residual(0.0); },
        [&] { ++accepts; }, [&] { ++mixes; });
    EXPECT_TRUE(out.converged);
    EXPECT_EQ(accepts, 1);
    EXPECT_EQ(mixes, 0);
}

TEST(SccDriver, ZeroBudgetAfterSuccessCannotReuseConvergedStatus) {
    SccDriver scc;
    EXPECT_TRUE(scc.solve(
                       nullptr, 1, 1e-8, [] {}, [] {},
                       [&] { scc.set_residual(0.0); }, [] {}, [] {})
                    .converged);
    const auto out =
        scc.solve(nullptr, 0, 1e-8, [] {}, [] {}, [] {}, [] {}, [] {});
    EXPECT_FALSE(out.converged);
    EXPECT_EQ(out.iterations, 0);
}

TEST(SccDriver, InvalidToleranceFailsBeforeStages) {
    SccDriver scc;
    int calls = 0;
    for (double tol : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(
            scc.solve(
                nullptr, 1, tol, [&] { ++calls; }, [] {}, [] {}, [] {}, [] {}),
            std::invalid_argument);
    EXPECT_EQ(calls, 0);
}
