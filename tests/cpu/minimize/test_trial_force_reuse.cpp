/// @file tests/cpu/minimize/test_trial_force_reuse.cpp
/// @brief Unit: a minimizer keeps the accepted line-search trial's forces
///        instead of replaying the force phase at the same geometry.
///
/// Mirrors `Runtime::step_linear_` (the skip), `Runtime::
/// force_slot_trial_invariant_` (the gate), `OptimizeServices::
/// trial_forces_reusable` / `forces_current`, and the accept branch of the
/// `line_search` phase in `src/cpu/minimize/lbfgs.cpp` and `cg.cpp`.
///
/// The witness is a launch counter on a closed-form harmonic well that tells
/// trial launches from committed ones. Without reuse every iteration pays one
/// committed evaluation on top of its trials; with reuse an iteration whose
/// accepted step was the last trial pays none — and the trajectory is the same
/// bit for bit, because the skipped evaluation would have reproduced the frame.

#include <utility>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"

using namespace ATV;

namespace {

constexpr double kSpring[3] = {1.0, 2.5, 4.0};                       // [eV/Å²]
constexpr double kCentre[2][3] = {{0.0, 0.0, 0.0}, {1.2, 0.0, 0.0}}; // [Å]

/// Anisotropic harmonic well, E = ½ Σ k_d (r − c)², counting its launches.
class CountingWell : public cpu::Potential {
  public:
    explicit CountingWell(bool invariant) : invariant_(invariant) {}
    int trials = 0, committed = 0, initial = 0;
    std::string name() const override { return "counting_well"; }
    bool trial_invariant() const override { return invariant_; }
    void launch(const cpu::LaunchContext& ctx) override {
        auto& f = ctx.frame;
        if (ctx.runtime.exec.initializing_forces)
            ++initial;
        else if (ctx.runtime.exec.trial_evaluation)
            ++trials;
        else
            ++committed;
        const StorageReal* x = f.span(kAtom_x);
        const StorageReal* y = f.span(kAtom_y);
        const StorageReal* z = f.span(kAtom_z);
        AccReal* fx = f.span(kAtom_fx);
        AccReal* fy = f.span(kAtom_fy);
        AccReal* fz = f.span(kAtom_fz);
        double e = 0;
        for (int i = 0; i < 2; ++i) {
            const double d[3] = {x[i] - kCentre[i][0], y[i] - kCentre[i][1],
                                 z[i] - kCentre[i][2]};
            e += 0.5
                 * (kSpring[0] * d[0] * d[0] + kSpring[1] * d[1] * d[1]
                    + kSpring[2] * d[2] * d[2]);
            fx[i] += -kSpring[0] * d[0];
            fy[i] += -kSpring[1] * d[1];
            fz[i] += -kSpring[2] * d[2];
        }
        f.meta.get(kEnergy) += e;
    }

  private:
    bool invariant_;
};

/// A force-phase component with no physics that does NOT declare itself
/// trial-invariant (the default) — it must veto the reuse.
class OpaqueHook : public cpu::Component {
  public:
    std::string name() const override { return "opaque_hook"; }
    void launch(const cpu::LaunchContext&) override {}
};

cpu::Frame displaced_pair() {
    cpu::Frame f(2);
    const double x0[2][3] = {{0.40, 0.30, -0.20}, {1.65, -0.25, 0.35}};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = x0[i][0];
        f.span(kAtom_y)[i] = x0[i][1];
        f.span(kAtom_z)[i] = x0[i][2];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 1;
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

/// Tolerances this well never reaches in the budget: the loop always runs
/// every iteration, so counts compare against a fixed reference.
cpu::ForceConverged never_converges() {
    return cpu::ForceConverged{1e-14, 1e-14, 1e-14, 1e-14, 1e-16};
}

constexpr int kIterations = 6;

struct ReuseRun {
    cpu::Frame frame;
    int trials, committed, initial;
};

template <class Optimizer>
ReuseRun minimize(Optimizer optimizer, bool invariant, bool with_hook) {
    ReuseRun r{displaced_pair(), 0, 0, 0};
    CountingWell well(invariant);
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(std::move(optimizer));
    opt.on(ATV::minimizer::Phase::force).add_ref(well, /*order=*/0);
    if (with_hook)
        opt.on(ATV::minimizer::Phase::force).add(OpaqueHook{}, 5);
    opt.add_criterion(never_converges());
    opt.build();
    EXPECT_EQ(opt.run(r.frame, kIterations), kIterations);
    r.trials = well.trials;
    r.committed = well.committed;
    r.initial = well.initial;
    return r;
}

void expect_same_state(const cpu::Frame& a, const cpu::Frame& b) {
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(a.span(kAtom_x)[i], b.span(kAtom_x)[i]);
        EXPECT_EQ(a.span(kAtom_y)[i], b.span(kAtom_y)[i]);
        EXPECT_EQ(a.span(kAtom_z)[i], b.span(kAtom_z)[i]);
        EXPECT_EQ(a.span(kAtom_fx)[i], b.span(kAtom_fx)[i]);
        EXPECT_EQ(a.span(kAtom_fy)[i], b.span(kAtom_fy)[i]);
        EXPECT_EQ(a.span(kAtom_fz)[i], b.span(kAtom_fz)[i]);
    }
    EXPECT_EQ(a.meta.get(kEnergy), b.meta.get(kEnergy));
}

} // namespace

TEST(TrialForceReuse, LbfgsSkipsTheCommittedReplayOfTheAcceptedTrial) {
    const ReuseRun reuse = minimize(cpu::minimizer::LBFGS{}, true, false);
    const ReuseRun replay = minimize(cpu::minimizer::LBFGS{}, false, false);

    // Without reuse: one committed evaluation per iteration.
    EXPECT_EQ(replay.committed, kIterations);
    // With reuse: strictly fewer, and nothing else changes.
    EXPECT_LT(reuse.committed, kIterations);
    EXPECT_EQ(reuse.trials, replay.trials);
    EXPECT_EQ(reuse.initial, 1);
    expect_same_state(reuse.frame, replay.frame);
}

TEST(TrialForceReuse, CgSkipsTheCommittedReplayOfTheAcceptedTrial) {
    const ReuseRun reuse = minimize(cpu::minimizer::CG{}, true, false);
    const ReuseRun replay = minimize(cpu::minimizer::CG{}, false, false);
    EXPECT_EQ(replay.committed, kIterations);
    EXPECT_LT(reuse.committed, kIterations);
    EXPECT_EQ(reuse.trials, replay.trials);
    expect_same_state(reuse.frame, replay.frame);
}

TEST(TrialForceReuse, OneOpaqueForceComponentKeepsEveryCommittedEvaluation) {
    // The well is invariant, but a component that does not say so shares the
    // force phase: the gate is "every op", so nothing is skipped.
    const ReuseRun run = minimize(cpu::minimizer::LBFGS{}, true, true);
    EXPECT_EQ(run.committed, kIterations);
}

TEST(TrialForceReuse, FixedStepHasNoTrialToReuse) {
    cpu::minimizer::LBFGS fixed;
    fixed.line_search(cpu::minimizer::LineSearchMode::Fixed);
    const ReuseRun run = minimize(std::move(fixed), true, false);
    EXPECT_EQ(run.trials, 0);
    EXPECT_EQ(run.committed, kIterations);
}
