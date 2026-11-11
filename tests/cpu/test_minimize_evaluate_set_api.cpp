/// @file tests/cpu/test_minimize_evaluate_set_api.cpp
/// @brief Phase 3 (driver-infra-auto-injection) — categorical setters on
///        cpu::minimizer::Minimize and cpu::Evaluate.
///
/// Acceptance: T6 + T7 + parts of T11/T14 of the spec.

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include "cpu/driver_test_helpers.h"

using atv_test::StubEnergyEvaluator;

namespace {

constexpr double kCutoff = 6.0;
constexpr double kEps = 0.0104;
constexpr double kSigma = 3.4;
constexpr double kSkin = 0.3;

// ─── Minimize setters ───────────────────────────────────────────────

TEST(MinimizeSetApi, SetOptimizerSingularSameNameIdempotent) {
    ATV::cpu::minimizer::Minimize opt;
    EXPECT_NO_THROW(opt.set_optimizer(ATV::cpu::minimizer::LBFGS{}));
    EXPECT_NO_THROW(opt.set_optimizer(ATV::cpu::minimizer::LBFGS{}));
}

TEST(MinimizeSetApi, SetOptimizerRejectsDifferentTypeAfterFirst) {
    ATV::cpu::minimizer::Minimize opt;
    opt.set_optimizer(ATV::cpu::minimizer::LBFGS{});
    EXPECT_THROW(opt.set_optimizer(ATV::cpu::minimizer::CG{}),
                 std::invalid_argument);
}

TEST(MinimizeSetApi, PairWithNeighborListBuildsClean) {
    ATV::cpu::minimizer::Minimize opt;
    opt.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    opt.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    opt.set_optimizer(ATV::cpu::minimizer::LBFGS{});
    EXPECT_NO_THROW(opt.build());
}

TEST(MinimizeSetApi, PairWithoutNeighborListThrowsAtBuild) {
    ATV::cpu::minimizer::Minimize opt;
    opt.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    opt.set_optimizer(ATV::cpu::minimizer::LBFGS{});
    try {
        opt.build();
        FAIL() << "expected throw — Minimize pair without set_neighbor";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("neighbor"), std::string::npos) << msg;
        EXPECT_NE(msg.find("lj"), std::string::npos) << msg;
    }
}

// ─── Evaluate setters ────────────────────────────────────────────────

TEST(EvaluateSetApi, PairWithNeighborListBuildsClean) {
    ATV::cpu::Evaluate eval;
    eval.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    eval.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    EXPECT_NO_THROW(eval.build());
}

TEST(EvaluateSetApi, PairWithoutNeighborListThrowsAtBuild) {
    ATV::cpu::Evaluate eval;
    eval.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    try {
        eval.build();
        FAIL() << "expected throw — Evaluate pair without set_neighbor";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("neighbor"), std::string::npos) << msg;
    }
}

TEST(EvaluateSetApi, EnergyOnlyEvaluatorBuildsWithoutPair) {
    // No pair → no NL requirement.  An energy-only evaluator added via
    // the public `add()` escape hatch satisfies the role validation.
    ATV::cpu::Evaluate eval;
    StubEnergyEvaluator energy;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(energy);
    EXPECT_NO_THROW(eval.build());
}

} // namespace

TEST(MinimizeSetApi, NegativeBudgetRejectedBeforeSetup) {
    ATV::cpu::minimizer::Minimize opt;
    ATV::cpu::Frame frame;
    EXPECT_THROW(opt.run(frame, -1), std::invalid_argument);
}

TEST(MinimizeSetApi, UnfixOptimizerAllowsReplacement) {
    ATV::cpu::minimizer::Minimize opt;
    opt.set_evaluator(ATV::cpu::PairLJCut{kEps, kSigma, kCutoff});
    opt.set_neighbor(ATV::cpu::md::NeighborList<>{kCutoff, kSkin});
    opt.set_optimizer(ATV::cpu::minimizer::LBFGS{});
    opt.build();
    // Removing the only optimizer reports an incomplete plan, but removes it.
    EXPECT_THROW(opt.unfix("lbfgs"), std::runtime_error);
    EXPECT_NO_THROW(opt.set_optimizer(ATV::cpu::minimizer::CG{}));
    EXPECT_NO_THROW(opt.build());
}

TEST(MinimizeSetApi, WideBudgetDoesNotNarrowBeforeConvergence) {
    auto frame = atv_test::make_trivial_frame();
    ATV::cpu::minimizer::Minimize opt;
    atv_test::StubOptimizer optimizer;
    atv_test::StubForceEvaluator force;
    opt.on(ATV::minimizer::Phase::search_direction).add_ref(optimizer);
    opt.on(ATV::minimizer::Phase::force).add_ref(force);
    opt.build();
    EXPECT_EQ(opt.run(frame, ATV::BigInt{1} << 40), 1);
    EXPECT_EQ(optimizer.step_count, 1);
}
