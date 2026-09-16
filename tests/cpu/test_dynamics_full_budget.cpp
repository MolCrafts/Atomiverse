/// @file tests/cpu/test_dynamics_full_budget.cpp
/// @brief Dynamics termination, after termination stopped being a category.
///
/// This file used to assert that Dynamics **ignores** `converged`. That rule existed for
/// one reason: any component could write `outer_status.converged = true`, and an MD run
/// must not be halted by some component's private idea of convergence. The rule was a
/// shield over a hole in the design.
///
/// The hole is gone. Under `Criterion`, **a component cannot declare convergence at
/// all** — it may *request a stop* (`status->stop_requested`) and it may *publish
/// metrics*, and a criterion does the judging. So Dynamics no longer needs to ignore
/// anything:
///
///   A) with no criterion attached, an MD run has no early exit — it runs its budget
///      (`cpu::md::MD`, unlike `Minimize`, attaches no default criterion);
///   B) a component that requests a stop still stops it;
///   C) a user-supplied criterion that returns `Converged` DOES stop it — which is new,
///      and is the point. "Run until the temperature settles" is a thing a user is
///      allowed to want, and there was no way to say it before.

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/dynamics/md.h"

#include "cpu/driver_test_helpers.h"

using atv_test::make_trivial_frame;
using atv_test::StubForceEvaluator;
using atv_test::StubIntegrator;

namespace {

/// A component may REQUEST a stop. It may not declare convergence — there is no longer
/// any field it could write to do so.
struct StopAtStep : ATV::cpu::Component {
    int stop_at;
    explicit StopAtStep(int s) : stop_at(s) {}
    std::string name() const override { return "stop_at_step"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        if (ctx.step >= stop_at && ctx.runtime.loop.status) {
            ctx.runtime.loop.status->stop_requested = true;
            ctx.runtime.loop.status->reason = "stop-requested";
        }
    }
};

/// A user-written criterion, defined outside the library's own vocabulary. THIS is the
/// thing that was impossible before: "stop or not" was a private `if` inside a driver.
class ConvergeAtIteration : public ATV::cpu::Criterion {
  public:
    explicit ConvergeAtIteration(int at) : at_(at) {}
    std::string name() const override { return "converge_at_iteration"; }
    void launch(const ATV::cpu::LaunchContext&) override {} // verdict-only
    ATV::Verdict verdict(const ATV::cpu::LoopState& s) override {
        return (s.iteration >= at_) ? ATV::Verdict::Converged
                                    : ATV::Verdict::Continue;
    }

  private:
    int at_;
};

} // namespace

/// A) No criterion ⇒ no early exit. The budget is the only termination.
TEST(DynamicsTermination, NoCriterionRunsTheFullBudget) {
    using namespace ATV;
    // Components BEFORE the driver: `cpu::md::MD` is continuable, so the component
    // destroy() pass is deferred to ~MD — the driver must be destroyed first (declared
    // last), else its destructor calls virtuals on already-destructed stack components
    // (pure-virtual abort).
    StubIntegrator integ;
    StubForceEvaluator force;
    cpu::md::MD d;

    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    d.build();

    auto frame = make_trivial_frame();
    d.run(frame, /*n_steps=*/5);
    EXPECT_EQ(integ.step_count, 5);
}

/// B) A component that requests a stop still stops the loop.
TEST(DynamicsTermination, BreaksOnStopRequested) {
    using namespace ATV;
    StubIntegrator integ;
    StubForceEvaluator force;
    StopAtStep stopper{/*stop_at=*/3};
    cpu::md::MD d;

    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    d.on(md::Phase::end_of_step).add_ref(stopper);
    d.build();

    auto frame = make_trivial_frame();
    d.run(frame, /*n_steps=*/10);
    // Steps 0,1,2,3 run; at step 3 the stopper flips stop_requested and the driver
    // breaks after the step.
    EXPECT_EQ(integ.step_count, 4);
}

/// C) THE NEW CAPABILITY: a user criterion halts an MD run. Dynamics does not "ignore
/// converged" any more — it simply has no default criterion, so nothing converges it
/// unless the user says what convergence would mean.
TEST(DynamicsTermination, AUserCriterionCanHaltAnMdRun) {
    using namespace ATV;
    StubIntegrator integ;
    StubForceEvaluator force;
    cpu::md::MD d;

    d.on(md::Phase::initial_integrate).add_ref(integ);
    d.on(md::Phase::force).add_ref(force);
    d.add_criterion(ConvergeAtIteration{2});
    d.build();

    auto frame = make_trivial_frame();
    d.run(frame, /*n_steps=*/10);
    EXPECT_EQ(integ.step_count, 3)
        << "iterations 0,1,2 run; the criterion halts after 2";
}
