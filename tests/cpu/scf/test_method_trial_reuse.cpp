/// @file tests/cpu/scf/test_method_trial_reuse.cpp
/// @brief Unit: a minimizer keeps an SCF method's accepted line-search trial
///        instead of re-solving the SCF at the same geometry.
///
/// Mirrors `cpu::DriverNode::trial_invariant` (forwarded by the nested-slot
/// adapter in `driver/backend_traits.h`) and `MeanFieldMethod::
/// trial_invariant` / its warm-start density on line-search trials
/// (`src/cpu/scf/mean_field_method.cpp`). The generic skip itself is
/// covered in tests/cpu/minimize/test_trial_force_reuse.cpp.
///
/// The witness is a physics-free probe in the same force phase that counts
/// committed launches: a committed launch is a replayed force phase, i.e. a
/// second SCF at a geometry its trial already solved.

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
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

/// Counts force-phase launches; contributes nothing.
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

/// Vetoes trial reuse (default `trial_invariant() == false`).
class OpaqueHook : public cpu::Component {
  public:
    std::string name() const override { return "opaque_hook"; }
    void launch(const cpu::LaunchContext&) override {}
};

struct MinimizeRun {
    cpu::Frame frame{2};
    int trials = 0, committed = 0;
};

/// H2/STO-3G from a stretched 0.95 Å bond, L-BFGS, fixed budget.
MinimizeRun minimize_h2(int iterations, bool veto_reuse) {
    MinimizeRun r;
    r.frame.span(kAtom_x)[1] = 0.95;
    for (int i = 0; i < 2; ++i) {
        r.frame.span(kAtom_atomic_number)[i] = 1;
        r.frame.span(kAtom_mass)[i] = 1.008;
    }
    r.frame.box = cpu::Box::cubic(20.0);

    LaunchProbe probe;
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{});
    opt.on(ATV::minimizer::Phase::force)
        .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                           /*tol=*/1e-10});
    opt.on(ATV::minimizer::Phase::force).add_ref(probe, /*order=*/5);
    if (veto_reuse)
        opt.on(ATV::minimizer::Phase::force).add(OpaqueHook{}, 6);
    // Tolerances H2 never reaches inside the budget: every iteration runs.
    opt.add_criterion(cpu::ForceConverged{1e-14, 1e-14, 1e-14, 1e-14, 1e-16});
    opt.build();
    opt.run(r.frame, iterations);
    r.trials = probe.trials;
    r.committed = probe.committed;
    return r;
}

double bond(const cpu::Frame& f) {
    const double dx = f.span(kAtom_x)[1] - f.span(kAtom_x)[0];
    const double dy = f.span(kAtom_y)[1] - f.span(kAtom_y)[0];
    const double dz = f.span(kAtom_z)[1] - f.span(kAtom_z)[0];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

TEST(MethodTrialReuse, RhfAcceptedTrialIsNotSolvedTwice) {
    constexpr int kIterations = 4;
    const MinimizeRun replayed = minimize_h2(kIterations, /*veto_reuse=*/true);
    const MinimizeRun reused = minimize_h2(kIterations, /*veto_reuse=*/false);

    // Without reuse every iteration re-solves its accepted geometry.
    EXPECT_EQ(replayed.committed, kIterations);
    // With reuse the accepted trial's SCF is kept.
    EXPECT_LT(reused.committed, replayed.committed);
    EXPECT_EQ(reused.trials, replayed.trials);

    // Same optimization: the kept trial is the converged SCF at the accepted
    // geometry, so only the starting guess differs (inside the SCF tol).
    EXPECT_NEAR(bond(reused.frame), bond(replayed.frame), 1e-7);
    EXPECT_NEAR(reused.frame.meta.get(kEnergy),
                replayed.frame.meta.get(kEnergy), 1e-7);
    EXPECT_NEAR(reused.frame.span(kAtom_fx)[0],
                replayed.frame.span(kAtom_fx)[0], 1e-6);
}
