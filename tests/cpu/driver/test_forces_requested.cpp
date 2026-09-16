/// @file tests/cpu/driver/test_forces_requested.cpp
/// @brief esmm-contract-04-forces-requested (U4) — `forces_requested` is a
///        compiled **Plan** property, not a permanently-true default.
///
/// Mirrors one production unit: the CPU loop node
/// `include/atomiverse/cpu/driver/driver.h` (`cpu::Driver<PhaseEnum>::build()`
/// stamping `StageSlot::forces_requested` from the new
/// `phase_wants_forces(PhaseEnum)` hook), observed through the one channel
/// components already read — `LaunchContext::forces_requested`.
///
/// `LaunchContext::forces_requested` was introduced FOR the Monte Carlo
/// energy-only trial (its own comment, `cpu/component/component.h:43-44`) and
/// never plumbed: it defaults `true` and no driver ever writes it
/// (`src/cpu/driver/runtime.cpp:136, 244, 296` all construct the context with
/// three fields and take the default). Seven shipping evaluators already guard
/// force assembly on it (GPW, PW, AM1, RM1, HarmonicBias, Metadynamics,
/// fix_efield/addforce/wall) — the flag is a read-only promise nobody keeps.
///
/// What is pinned here, per phase:
///
///   - MC's `sampling::Phase::energy` launches **energy-only** (`false`) —
///     a nested QM engine then skips its whole gradient for free;
///   - MC's other phases (`propose`, `end_of_step`) still say `true`, so the
///     veto is per-slot and not per-driver;
///   - MD `force`, Minimize `force`, Evaluate `evaluate` still say `true` —
///     the controls for "no behaviour change outside MC" (ac-005);
///   - the veto propagates INTO a nested driver's own slots (ac-002): a
///     `Driver<scf::Phase>` node bound at MC's energy phase must launch its
///     own inner plan energy-only, because the AND rides `ExecServices`
///     through `Runtime::step_plan`;
///   - the same nested node under MD `force` still launches `true`, so the
///     propagation is an AND with the outer slot and not a blanket
///     "nested ⇒ energy-only".
///
/// **Scope.** Unit only, stubs only. No physics is evaluated: every stub is a
/// no-op `launch`, so the whole file is microseconds and nothing here depends
/// on `ATV_BUILD_HF`. The QM-in-MC end-to-end witness (a nested RHF energy +
/// umbrella bias, pinned against the analytic stiff-spring variance) is
/// `regressions/mc/mc_bias_es`, not this file.
///
/// **Determinism.** No RNG consumed by any stub (the MC RNG stream is seeded
/// fixed and only the acceptance policy draws from it), no wall clock, no
/// filesystem, no network, no third-party oracle.
///
/// Spec: `.claude/specs/esmm-contract-04-forces-requested/spec.md`
/// (ac-001, ac-002, ac-005).

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/montecarlo/mc.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;

namespace {

/// @brief Records `LaunchContext::forces_requested` on every launch.
///
/// A plain `cpu::Component` on purpose: it must not double as an energy source
/// for any preset's `validate_components()`, so each test below supplies its
/// own explicit source and the probe only ever observes.
class ForcesFlagProbe final : public cpu::Component {
  public:
    explicit ForcesFlagProbe(std::string tag) : tag_(std::move(tag)) {}

    std::string name() const override { return tag_; }

    void launch(const cpu::LaunchContext& ctx) override {
        ++launches;
        if (ctx.forces_requested)
            ++true_launches;
        else
            ++false_launches;
    }

    int launches = 0;       ///< Total launches seen.
    int true_launches = 0;  ///< Launches that requested forces.
    int false_launches = 0; ///< Launches that were energy-only.

  private:
    std::string tag_;
};

/// @brief Minimal `Mover`-kind stub: proposes nothing, records nothing.
struct StubMover final : cpu::Mover {
    std::string name() const override { return "stub_mover"; }
    void launch(const cpu::LaunchContext&) override {}
};

/// @brief Minimal `Potential`-kind stub — the energy source a preset requires.
struct StubEnergy final : cpu::Potential {
    std::string name() const override { return "stub_energy"; }
    void launch(const cpu::LaunchContext&) override {}
};

/// @brief The node shape a concrete Gaussian method presents to an outer
///        driver: `Driver<scf::Phase>` (hence a `cpu::DriverNode`) plus the
///        pure `HF` kind tag — so it is simultaneously the outer preset's
///        energy source and a nesting boundary with its OWN plan slots.
///
/// `cpu::scf::RHF` is this plus physics. No stages, no integrals: the veto
/// channel under test is structural, and a stub keeps this file out of
/// `ATV_BUILD_HF` and in the microseconds.
struct StubMethodNode final : cpu::Driver<ATV::scf::Phase>, cpu::HF {
    void validate_components() const override {}
};

} // namespace

// ── Basics: the flag's own default ──────────────────────────────────────────

/// The default is `true`: a force consumer that is never told otherwise keeps
/// assembling forces. Every "still true" control below rests on this.
TEST(ForcesRequested, LaunchContextDefaultsToRequestingForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(1);
    cpu::Context runtime_ctx;
    const cpu::LaunchContext lc{runtime_ctx, frame, /*stage=*/0, /*step=*/0};
    EXPECT_TRUE(lc.forces_requested);
}

// ── ac-001: MC's energy phase is energy-only ────────────────────────────────

/// RED today: the probe records `true` on every launch because no driver ever
/// stamps the slot. A trial move discards forces by construction — every
/// gradient assembled at `sampling::Phase::energy` is wasted work, and for a
/// QM engine it is the dominant cost of the sweep.
TEST(ForcesRequested, McEnergyPhaseRunsEnergyOnly) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    ForcesFlagProbe probe{"probe_mc_energy"};
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::propose).add(StubMover{}, /*order=*/0);
    d.on(sampling::Phase::energy).add(StubEnergy{}, /*order=*/0);
    d.on(sampling::Phase::energy).add_ref(probe, /*order=*/10);
    d.build();
    d.run(frame, /*n_sweeps=*/1);

    // Bootstrap pass (step kBootstrapStep) + one sweep — the slot ran.
    ASSERT_GE(probe.launches, 2);
    EXPECT_EQ(probe.true_launches, 0)
        << "MC's energy phase requested forces on " << probe.true_launches
        << " of " << probe.launches
        << " launches — the trial energy pipeline is still paying for "
           "gradients the Metropolis test throws away";
    EXPECT_EQ(probe.false_launches, probe.launches);
}

/// The veto is a property of the SLOT, not of the driver: MC's `propose` and
/// `end_of_step` phases are ordinary force-capable slots (a Mover may need a
/// force, and `end_of_step` is where MD-shaped hooks live). Passes today for
/// the weaker reason that nothing is stamped anywhere; it pins the GREEN
/// against a per-driver flag.
TEST(ForcesRequested, McNonEnergyPhasesStillRequestForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    ForcesFlagProbe propose_probe{"probe_mc_propose"};
    ForcesFlagProbe end_probe{"probe_mc_end_of_step"};
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::propose).add(StubMover{}, /*order=*/0);
    d.on(sampling::Phase::propose).add_ref(propose_probe, /*order=*/10);
    d.on(sampling::Phase::energy).add(StubEnergy{}, /*order=*/0);
    d.on(sampling::Phase::end_of_step).add_ref(end_probe, /*order=*/10);
    d.build();
    d.run(frame, /*n_sweeps=*/1);

    ASSERT_GT(propose_probe.launches, 0);
    ASSERT_GT(end_probe.launches, 0);
    EXPECT_EQ(propose_probe.false_launches, 0)
        << "the energy-only veto leaked out of the energy slot into `propose`";
    EXPECT_EQ(end_probe.false_launches, 0)
        << "the energy-only veto leaked out of the energy slot into "
           "`end_of_step` — a slot flag was applied per driver";
}

// ── ac-005: nobody else changes ─────────────────────────────────────────────

/// MD's `force` phase is the canonical force consumer. Control: passes today
/// and must keep passing.
TEST(ForcesRequested, MdForcePhaseRequestsForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    atv_test::StubIntegrator integrator;
    atv_test::StubForceEvaluator force;
    ForcesFlagProbe probe{"probe_md_force"};

    cpu::md::MD sim;
    sim.on(md::Phase::initial_integrate).add_ref(integrator);
    sim.on(md::Phase::force).add_ref(force, /*order=*/0);
    sim.on(md::Phase::force).add_ref(probe, /*order=*/10);
    sim.build();
    sim.run(frame, /*n_steps=*/1);

    ASSERT_GT(probe.launches, 0);
    EXPECT_EQ(probe.false_launches, 0)
        << "MD's force phase was launched energy-only — the integrator would "
           "advance on stale or zeroed forces";
    EXPECT_EQ(probe.true_launches, probe.launches);
}

/// Minimize's `force` phase: the optimizer's search direction IS the gradient.
TEST(ForcesRequested, MinimizeForcePhaseRequestsForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    atv_test::StubOptimizer optimizer;
    atv_test::StubEnergyEvaluator energy;
    ForcesFlagProbe probe{"probe_min_force"};

    cpu::minimizer::Minimize opt;
    opt.on(minimizer::Phase::search_direction).add_ref(optimizer);
    opt.on(minimizer::Phase::force).add_ref(energy, /*order=*/0);
    opt.on(minimizer::Phase::force).add_ref(probe, /*order=*/10);
    opt.add_criterion(cpu::ForceConverged{});
    opt.build();
    opt.run(frame, /*max_iter=*/1);

    ASSERT_GT(probe.launches, 0);
    EXPECT_EQ(probe.false_launches, 0)
        << "Minimize's force phase was launched energy-only — the optimizer "
           "would step along a gradient nobody computed";
    EXPECT_EQ(probe.true_launches, probe.launches);
}

/// Evaluate's single pass: a single-point call may well be asked for forces.
TEST(ForcesRequested, EvaluatePhaseRequestsForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    atv_test::StubEnergyEvaluator energy;
    ForcesFlagProbe probe{"probe_evaluate"};

    cpu::Evaluate e;
    e.on(evaluate::Phase::evaluate).add_ref(energy, /*order=*/0);
    e.on(evaluate::Phase::evaluate).add_ref(probe, /*order=*/10);
    e.build();
    e.run(frame);

    ASSERT_GT(probe.launches, 0);
    EXPECT_EQ(probe.false_launches, 0)
        << "Evaluate's evaluate phase was launched energy-only";
    EXPECT_EQ(probe.true_launches, probe.launches);
}

// ── ac-002: the veto propagates through nesting ─────────────────────────────

/// A nested driver runs its OWN plan (`Runtime::step_plan`), whose slots were
/// stamped by ITS `phase_wants_forces` — which for a method loop says `true`
/// for every SCF stage. So a slot flag alone stops at the node boundary: the
/// outer energy-only veto has to ride a Context channel (`ExecServices`) and
/// AND itself into the child's slots.
///
/// The probe is bound INSIDE the nested node's plan, so it can only be reached
/// by actually re-entering the node. RED today on BOTH probes (`true`
/// everywhere); the inner one is the propagation claim.
TEST(ForcesRequested, McEnergyVetoReachesNestedDriverSlots) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    ForcesFlagProbe inner_probe{"probe_nested_fock"};
    StubMethodNode method; // the ONLY energy source (nested `HF` node)
    method.on(ATV::scf::Phase::fock).add_ref(inner_probe);

    ForcesFlagProbe outer_probe{"probe_mc_energy_outer"};
    ForcesFlagProbe end_probe{"probe_mc_end_of_step"};
    cpu::mc::MC d{/*temperature_K=*/300.0, /*seed=*/12345u};
    d.on(sampling::Phase::propose).add(StubMover{}, /*order=*/0);
    // NOTE: `add_ref`'s second positional argument is `order`, the third is the
    // nested budget — the base `default_budget()` is 0, so a nested node with
    // an unspecified budget would never run its plan at all.
    d.on(sampling::Phase::energy)
        .add_ref(method, /*order=*/0, /*budget=*/1);
    d.on(sampling::Phase::energy).add_ref(outer_probe, /*order=*/10);
    d.on(sampling::Phase::end_of_step).add_ref(end_probe, /*order=*/10);
    d.build();
    d.run(frame, /*n_sweeps=*/1);

    ASSERT_GT(inner_probe.launches, 0)
        << "the nested node never ran its own plan — budget wiring, not the "
           "flag, is what this case would be measuring";
    ASSERT_GT(outer_probe.launches, 0);
    ASSERT_GT(end_probe.launches, 0);

    EXPECT_EQ(outer_probe.true_launches, 0)
        << "outer MC energy slot still requested forces";
    EXPECT_EQ(inner_probe.true_launches, 0)
        << "the nested driver launched its own slots with forces_requested = "
           "true on " << inner_probe.true_launches << " of "
        << inner_probe.launches
        << " launches — the outer energy-only veto stopped at the node "
           "boundary, so a nested QM method still assembles the gradient the "
           "trial discards";
    // The channel is saved/restored around the nested dispatch, so a LATER
    // outer phase in the same sweep is unaffected.
    EXPECT_EQ(end_probe.false_launches, 0)
        << "the nested dispatch left the veto set on the Context — every "
           "phase after it in the same sweep went energy-only";
}

/// The propagation is an **AND with the outer slot**, never "nested ⇒
/// energy-only": the same node nested under MD's force phase must launch its
/// own slots requesting forces. Passes today (everything is true); it pins the
/// GREEN against stamping nested slots false unconditionally.
TEST(ForcesRequested, NestedDriverUnderMdForceStillRequestsForces) {
    cpu::Frame frame = atv_test::make_trivial_frame(2);

    ForcesFlagProbe inner_probe{"probe_nested_fock_md"};
    StubMethodNode method;
    method.on(ATV::scf::Phase::fock).add_ref(inner_probe);

    atv_test::StubIntegrator integrator;
    cpu::md::MD sim;
    sim.on(md::Phase::initial_integrate).add_ref(integrator);
    sim.on(md::Phase::force).add_ref(method, /*order=*/0, /*budget=*/1);
    sim.build();
    sim.run(frame, /*n_steps=*/1);

    ASSERT_GT(inner_probe.launches, 0);
    EXPECT_EQ(inner_probe.false_launches, 0)
        << "a nested method under MD was launched energy-only — AIMD would "
           "integrate on forces the method was told not to compute";
    EXPECT_EQ(inner_probe.true_launches, inner_probe.launches);
}
