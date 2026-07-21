/// @file tests/cpu/test_optimize_driver_early_stop.cpp
/// @brief Phase 1.7 — `cpu::minimizer::Minimize` breaks the loop when the metrics a
///        ConvergenceCheck-tagged component publishes satisfy the attached
///        `ForceConverged` criterion. Returns the iteration count actually run.
///
/// Acceptance: ac-005 (Optimize half) in
/// `.claude/specs/cpu-driver-taxonomy.acceptance.md`.

#include <gtest/gtest.h>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/facade/stage.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubEnergyEvaluator;
using atv_test::StubOptimizer;

namespace {

/// Flips outer_status.converged to true on the iteration whose 0-based
/// step index equals `converge_at_step`. Uses `LaunchContext::step` so
/// the trigger lines up with the driver's iteration counter.
struct ConvergeAtStep : ATV::cpu::Component {
    int converge_at_step;
    int launch_count = 0;
    explicit ConvergeAtStep(int s) : converge_at_step(s) {}
    std::string name() const override { return "converge_at_step"; }
    ATV::Role role() const override {
        return 0; // LoopControl-like metrics publisher
    }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        ++launch_count;
        // A component no longer declares convergence — it publishes what it measured.
        // Here: pretend the force collapses to zero once the target step is reached, and
        // let a ForceConverged criterion draw the conclusion. That separation IS the fix.
        if (ATV::ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            const bool done = ctx.step >= converge_at_step;
            m->max_force = done ? 0.0 : 1.0;
            m->rms_force = done ? 0.0 : 1.0;
            m->max_displacement = 0.0;
            m->rms_displacement = 0.0;
            m->energy_change = 0.0;
            m->has_displacement = true;
        }
    }
};

} // namespace

TEST(OptimizeEarlyStop, BreaksAtConvergeStep) {
    using namespace ATV;
    cpu::minimizer::Minimize opt;
    StubOptimizer optimizer;
    StubEnergyEvaluator energy;
    ConvergeAtStep converge{/*step=*/3};

    opt.on(minimizer::Phase::search_direction).add_ref(optimizer);
    opt.on(minimizer::Phase::force).add_ref(energy);
    opt.on(minimizer::Phase::convergence_check).add_ref(converge);
    // Termination is a POLICY, not a class — that is the point of the refactor. The
    // `Minimize` preset would inject `ForceConverged{}` itself if we attached nothing;
    // we attach it explicitly so this test pins the tolerances it judges against
    // instead of inheriting whatever the preset's defaults happen to be.
    opt.add_criterion(cpu::ForceConverged{});
    opt.build();

    auto frame = make_trivial_frame();
    int iters = opt.run(frame, /*max_iter=*/100);

    // 0-based step indices 0,1,2,3 — the convergence flips at step==3
    // and the driver breaks AFTER step 3 completes, so iters == 4.
    EXPECT_EQ(iters, 4) << "Expected 4 iterations (0,1,2,3 then break)";
    EXPECT_EQ(optimizer.step_count, 4);
    EXPECT_EQ(converge.launch_count, 4);
}

TEST(OptimizeEarlyStop, RunsToBudgetWhenNeverConverged) {
    using namespace ATV;
    cpu::minimizer::Minimize opt;
    StubOptimizer optimizer;
    StubEnergyEvaluator energy;
    ConvergeAtStep never{/*step=*/9999}; // far beyond max_iter

    opt.on(minimizer::Phase::search_direction).add_ref(optimizer);
    opt.on(minimizer::Phase::force).add_ref(energy);
    opt.on(minimizer::Phase::convergence_check).add_ref(never);
    // Termination is a POLICY, not a class — that is the point of the refactor. The
    // `Minimize` preset would inject `ForceConverged{}` itself if we attached nothing;
    // we attach it explicitly so this test pins the tolerances it judges against
    // instead of inheriting whatever the preset's defaults happen to be.
    opt.add_criterion(cpu::ForceConverged{});
    opt.build();

    auto frame = make_trivial_frame();
    int iters = opt.run(frame, /*max_iter=*/7);
    EXPECT_EQ(iters, 7);
    EXPECT_EQ(optimizer.step_count, 7);
}
