/// @file tests/cpu/semiempirical/test_nddo_trial_reuse.cpp
/// @brief Unit: a minimizer keeps AM1's accepted line-search trial instead of
///        re-solving the SCF at the same geometry.
///
/// Mirrors `NddoMethod::trial_invariant` and its line-search trial density
/// staging (`src/cpu/semiempirical/nddo_method.cpp`): a trial stages its
/// density in the step's evaluation transaction, so the step commits the last
/// geometry it evaluated — the accepted trial when its forces are kept.
/// The witness is a physics-free probe counting committed force launches.

#include <cmath>
#include <utility>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

using namespace ATV;

namespace {

class LaunchProbe : public cpu::Component {
  public:
    int trials = 0, committed = 0;
    std::string name() const override { return "launch_probe"; }
    bool trial_invariant() const override { return true; }
    void launch(const cpu::LaunchContext& ctx) override {
        if (ctx.runtime.exec.initializing_forces)
            return;
        if (ctx.runtime.exec.trial_evaluation)
            ++trials;
        else
            ++committed;
    }
};

class OpaqueHook : public cpu::Component {
  public:
    std::string name() const override { return "opaque_hook"; }
    void launch(const cpu::LaunchContext&) override {}
};

struct MinimizeRun {
    cpu::Frame frame{3};
    int trials = 0, committed = 0;
};

/// Water with one stretched O–H, AM1, L-BFGS, fixed budget.
MinimizeRun minimize_water(int iterations, bool veto_reuse) {
    MinimizeRun r;
    constexpr int z[3] = {8, 1, 1};
    constexpr double x[3][3] = {{0, 0, 0}, {1.05, 0, 0}, {-0.26, 1.0, 0}};
    for (int i = 0; i < 3; ++i) {
        r.frame.span(kAtom_atomic_number)[i] = z[i];
        r.frame.span(kAtom_tag)[i] = i + 1;
        r.frame.span(kAtom_mass)[i] = z[i] == 8 ? 15.999 : 1.008;
        r.frame.span(kAtom_x)[i] = x[i][0];
        r.frame.span(kAtom_y)[i] = x[i][1];
        r.frame.span(kAtom_z)[i] = x[i][2];
    }
    r.frame.box = cpu::Box::cubic(20.0);

    LaunchProbe probe;
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{});
    opt.on(ATV::minimizer::Phase::force).add(cpu::semiempirical::AM1{});
    opt.on(ATV::minimizer::Phase::force).add_ref(probe, /*order=*/5);
    if (veto_reuse)
        opt.on(ATV::minimizer::Phase::force).add(OpaqueHook{}, 6);
    opt.add_criterion(cpu::ForceConverged{1e-14, 1e-14, 1e-14, 1e-14, 1e-16});
    opt.build();
    opt.run(r.frame, iterations);
    r.trials = probe.trials;
    r.committed = probe.committed;
    return r;
}

} // namespace

TEST(NddoTrialReuse, Am1AcceptedTrialIsNotSolvedTwice) {
    constexpr int kIterations = 4;
    const MinimizeRun replayed =
        minimize_water(kIterations, /*veto_reuse=*/true);
    const MinimizeRun reused =
        minimize_water(kIterations, /*veto_reuse=*/false);

    EXPECT_EQ(replayed.committed, kIterations);
    EXPECT_LT(reused.committed, replayed.committed);
    EXPECT_EQ(reused.trials, replayed.trials);

    // Same optimization up to the SCF tolerance of the warm-started solves.
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(reused.frame.span(kAtom_x)[i],
                    replayed.frame.span(kAtom_x)[i], 1e-6);
        EXPECT_NEAR(reused.frame.span(kAtom_y)[i],
                    replayed.frame.span(kAtom_y)[i], 1e-6);
    }
    EXPECT_NEAR(reused.frame.meta.get(kEnergy),
                replayed.frame.meta.get(kEnergy), 1e-6);
}
