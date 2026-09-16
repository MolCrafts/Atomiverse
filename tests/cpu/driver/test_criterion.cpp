/// @file tests/cpu/driver/test_criterion.cpp
/// @brief Unit: built-in Criterion verdicts, Verdict composition, and the
///        status/metrics split.
///
/// Termination used to be a class hierarchy twice over: first six driver
/// categories differing by one `if`, then a free-standing loop-handler policy
/// object that was a first-class citizen but not a Component. A `Criterion` IS
/// a Component — same lifecycle, same ownership rules — that the driver polls
/// for a `Verdict` once per iteration. These tests pin that the three built-in
/// criteria judge exactly as their handler ancestors did.

#include <gtest/gtest.h>

#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"

#include <limits>

using namespace ATV;

namespace {

/// A criterion written entirely outside the library's own vocabulary — the
/// thing that was impossible when "stop or not" was a private `if` inside a
/// driver class. This is LAMMPS's `fix halt`.
/// Note it reaches BOTH the frame (kinetic energy) and the context (the DOF
/// count) — temperature is a derived quantity, not a frame field, and a
/// criterion is trusted with enough to derive it.
class StopWhenTooHot : public cpu::Criterion {
  public:
    explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
    std::string name() const override { return "stop_when_too_hot"; }
    void launch(const cpu::LaunchContext&) override {} // verdict-only
    Verdict verdict(const cpu::LoopState& s) override {
        const int ndof = s.runtime.md.n_dof_global;
        if (ndof <= 0)
            return Verdict::Continue;
        const double T =
            2.0 * s.frame.meta.get(ATV::kKinetic) / (ndof * units::kB_ev_per_k);
        return (T > t_max_) ? Verdict::Stop : Verdict::Continue;
    }

  private:
    double t_max_;
};

/// Minimal LoopState plumbing — a criterion needs nothing but these four
/// references.
struct Harness {
    cpu::Frame frame{1};
    cpu::Context ctx;
    ConvergenceMetrics metrics;

    cpu::LoopState state(int iteration) {
        return cpu::LoopState{iteration, frame, ctx, metrics};
    }
};

} // namespace

/// The composition rule, stated once: Stop > Converged > Continue.
///
/// Stop outranks Converged and that ordering is load-bearing: a run that has
/// gone numerically insane must not be able to report SUCCESS by also happening
/// to satisfy a force tolerance. Silence would be better than a wrong answer; a
/// wrong answer wearing a "converged" badge is the worst outcome available.
TEST(CriterionTest, SeverityOrdering) {
    EXPECT_EQ(worse_of(Verdict::Continue, Verdict::Converged),
              Verdict::Converged);
    EXPECT_EQ(worse_of(Verdict::Converged, Verdict::Stop), Verdict::Stop);
    EXPECT_EQ(worse_of(Verdict::Stop, Verdict::Converged), Verdict::Stop);
    EXPECT_EQ(worse_of(Verdict::Continue, Verdict::Continue),
              Verdict::Continue);
    // Idempotent, and order-independent — it is a max over severity, which is
    // exactly what lets the driver reduce it across MPI ranks with a single
    // MPI_MAX.
    EXPECT_EQ(worse_of(Verdict::Stop, Verdict::Stop), Verdict::Stop);
}

/// The three built-ins are verdict-only Components: no phase bindings, so they
/// enter the driver's polling list and never the Plan.
TEST(CriterionTest, BuiltInCriteriaAreVerdictOnly) {
    EXPECT_TRUE(cpu::ForceConverged{}.default_bindings().empty());
    EXPECT_TRUE(cpu::AllReplicasConverged{}.default_bindings().empty());
    EXPECT_TRUE(cpu::StopOnNaN{}.default_bindings().empty());
}

/// ForceConverged judges the METRICS, never the iteration counter.
TEST(CriterionTest, ForceConvergedNeedsEveryCriterion) {
    Harness h;
    cpu::ForceConverged crit{/*ftol=*/0.05, /*rms_ftol=*/0.03,
                             /*max_dtol=*/0.002,
                             /*rms_dtol=*/0.001, /*etol=*/1e-5};

    // Everything satisfied.
    h.metrics = ConvergenceMetrics{};
    h.metrics.max_force = 0.01;
    h.metrics.rms_force = 0.005;
    h.metrics.max_displacement = 0.001;
    h.metrics.rms_displacement = 0.0005;
    h.metrics.energy_change = 1e-6;
    h.metrics.has_displacement = true;
    EXPECT_EQ(crit.verdict(h.state(7)), Verdict::Converged);

    // ONE criterion violated — the force is fine but the geometry is still
    // sliding. All five must hold; this is the case a force-only test would
    // wave through.
    h.metrics.max_displacement = 0.5;
    EXPECT_EQ(crit.verdict(h.state(7)), Verdict::Continue);
}

/// Before the optimizer has a previous geometry, the displacement criteria are
/// skipped — but ONLY because the component said so (`has_displacement`), never
/// because the criterion counted iterations. That distinction is not academic:
/// the counter version shipped first, and because the optimizer has already
/// taken a step by the time convergence_check runs, driver-iteration 0 is not
/// the optimizer's first step. Skipping the displacement test there declared an
/// untouched lattice converged, having moved nothing.
TEST(CriterionTest, ForceConvergedSkipsDisplacementOnlyWhenTheComponentSaysSo) {
    Harness h;
    cpu::ForceConverged crit{};

    h.metrics = ConvergenceMetrics{};
    h.metrics.max_force = 0.01;
    h.metrics.rms_force = 0.005;
    h.metrics.energy_change = 1e-9;
    h.metrics.max_displacement = 999.0; // garbage — and it must be IGNORED
    h.metrics.rms_displacement = 999.0;
    h.metrics.has_displacement =
        false; // ... because the component says it is meaningless

    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Converged);

    // The same garbage, now declared meaningful, must block convergence.
    h.metrics.has_displacement = true;
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Continue);
}

/// A chain is converged when EVERY replica is — never when the first one is.
TEST(CriterionTest, AllReplicasConvergedIsAnAnd) {
    Harness h;
    cpu::AllReplicasConverged crit;

    h.metrics.replicas_converged = 0.0;
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Continue);

    // 12 of 13 images converged. The band is NOT converged, however happy
    // image 0 is — and the tangent that defines the saddle is computed FROM the
    // image still sliding.
    h.metrics.replicas_converged = 12.0 / 13.0;
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Continue);

    h.metrics.replicas_converged = 1.0;
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Converged);
}

/// A blown-up run Stops. It does NOT converge — see the severity ordering.
TEST(CriterionTest, StopOnNaNRefusesToReportSuccess) {
    Harness h;
    cpu::StopOnNaN crit;

    h.metrics.energy = -28.8;
    EXPECT_EQ(crit.verdict(h.state(3)), Verdict::Continue);

    h.metrics.energy = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(crit.verdict(h.state(3)), Verdict::Stop);

    h.metrics.energy = std::numeric_limits<double>::infinity();
    EXPECT_EQ(crit.verdict(h.state(3)), Verdict::Stop);
}

/// A criterion defined outside the library, reading the frame directly. This is
/// the whole deliverable: `fix halt` was inexpressible before, because
/// termination was a class.
TEST(CriterionTest, AUserCriterionCanReadTheFrame) {
    Harness h;
    StopWhenTooHot crit{2000.0};

    h.ctx.md.n_dof_global = 300;
    // KE for T = 300 K on 300 DOF: KE = Ndof kB T / 2
    h.frame.meta.get(ATV::kKinetic) = 0.5 * 300 * units::kB_ev_per_k * 300.0;
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Continue);

    h.frame.meta.get(ATV::kKinetic) =
        0.5 * 300 * units::kB_ev_per_k * 5000.0; // runaway
    EXPECT_EQ(crit.verdict(h.state(0)), Verdict::Stop);
}

/// The Context does not OWN the status — it points at the active loop's.
///
/// This is the structural fix. With one status on the Context, a nested loop's
/// convergence stopped its parent; that is not a rule anyone can be asked to
/// remember, so it is now a pointer that simply does not go there. Swapping the
/// pointer is how a nested driver takes over, and restoring it is how the
/// parent gets its own back.
TEST(CriterionTest, StatusIsBorrowedByTheContextNotOwned) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.loop.status, nullptr)
        << "a Context attached to no loop owns no status";

    IterationStatus outer_status;
    ConvergenceMetrics outer_metrics;
    ctx.loop.status = &outer_status;
    ctx.loop.metrics = &outer_metrics;

    { // an inner loop takes over, exactly as a nested driver's setup would
        IterationStatus inner_status;
        ConvergenceMetrics inner_metrics;
        auto* saved_s = ctx.loop.status;
        auto* saved_m = ctx.loop.metrics;
        ctx.loop.status = &inner_status;
        ctx.loop.metrics = &inner_metrics;

        // A component inside the inner loop declares a stop. It reaches the
        // INNER status.
        ctx.loop.status->stop_requested = true;

        EXPECT_TRUE(inner_status.stop_requested);
        EXPECT_FALSE(outer_status.stop_requested)
            << "the inner loop reached its parent's status — the bug that "
               "shipped twice";

        ctx.loop.status = saved_s;
        ctx.loop.metrics = saved_m;
    }

    EXPECT_EQ(ctx.loop.status, &outer_status);
    EXPECT_FALSE(outer_status.stop_requested);
}
