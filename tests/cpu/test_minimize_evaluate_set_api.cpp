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
