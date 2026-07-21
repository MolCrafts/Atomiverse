/// @file tests/cpu/driver/test_scf_phase_plumbing.cpp
/// @brief RED-first unit test for the SCF-as-Driver type foundation
///        (scf-driver-refactor-01-plumb).
///
/// This pins the *plumbing* that lets the gaussian-basis SCF self-consistency loop
/// become a `cpu::CPUDriver<scf::Phase>` in later sub-specs — WITHOUT wiring any
/// consumer yet. Everything asserted here is pure-additive facade / core /
/// infrastructure type surface:
///
///   1. `ATV::scf::Phase` (fock → mix → diagonalize → density → check, + COUNT) and its
///      `kind_of` tag dispatch, reachable through the single sanctioned typed entry
///      points `slot_of` / `slot_count` / `make_binding` — NO `static_cast<int>` on a
///      Phase enum anywhere (the `test_stage_typing_scrub` grep gate stays green).
///   2. `PhaseKind::SCF` + its `phase_kind_name` case.
///   3. `PhaseTraits<scf::Phase>` (eval=fock, neighbor=fock, end=check), proven by
///      instantiating a `CPUDriver<scf::Phase>` subclass.
///   4. The `cpu::Criterion` component kind (`Verdict verdict(const LoopState&)`) plus the
///      driver's `add_criterion` / `add_criterion_ref` registration and its per-iteration,
///      severity-composed pull: Stop > Converged > Continue, with the loop — never the
///      component — writing convergence status.
///
/// None of these symbols exist yet: the expected RED is that this translation unit FAILS
/// TO COMPILE on missing `ATV::scf::Phase`, `PhaseKind::SCF`, `criterion.h`,
/// `add_criterion`, etc. The assertions must NOT be weakened to make it compile.
///
/// Harness mirrors `tests/cpu/driver/test_sub_driver.cpp` (`OuterLoop`): a minimal
/// `CPUDriver<>` subclass whose `run()` composes the protected loop primitives
/// (`setup_pass` / `step_once` / `poll_criteria_` / `teardown_pass`). Unit-scale: an
/// (almost) empty plan over a 1-atom frame, no physics, milliseconds.

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/facade/loop.h"
#include "atomiverse/facade/stage.h"

#include "cpu/driver_test_helpers.h"

using namespace ATV;

namespace {

// ── Toy stage component — proves the loop actually walked its stages ──────────

/// @brief A no-op `scf::Phase::fock` stage binder that counts its launches, so a
///        criterion polled "after stages" can be shown to see the stage having run.
class StageProbe : public cpu::Component {
  public:
    std::string name() const override { return "scf_stage_probe"; }
    void launch(const cpu::LaunchContext&) override { ++launches; }
    int launches = 0;
};

// ── Toy criteria — verdict-only (empty default_bindings ⇒ polling list only) ──

/// @brief Always votes Continue; records how many stages had run at its first poll.
///
/// The ordering witness: if the driver pulls `verdict()` AFTER the stages of the same
/// iteration, then at the first poll the stage probe has launched exactly once.
class ContinueCriterion : public cpu::Criterion {
  public:
    explicit ContinueCriterion(const StageProbe* stage) : stage_(stage) {}
    std::string name() const override { return "continue_criterion"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override {
        if (polls == 0 && stage_)
            stage_at_first_poll = stage_->launches;
        ++polls;
        return Verdict::Continue;
    }
    int polls = 0;
    int stage_at_first_poll = -1;

  private:
    const StageProbe* stage_;
};

/// @brief Votes Continue for the first `at_` polls, then Converged. It only ever RETURNS a
///        verdict — it has no field with which to write the loop's status itself.
class ConvergeAfter : public cpu::Criterion {
  public:
    explicit ConvergeAfter(int at) : at_(at) {}
    std::string name() const override { return "converge_after"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override {
        const bool done = polls >= at_;
        ++polls;
        return done ? Verdict::Converged : Verdict::Continue;
    }
    int polls = 0;

  private:
    int at_;
};

/// @brief Votes Converged every poll.
class AlwaysConverged : public cpu::Criterion {
  public:
    std::string name() const override { return "always_converged"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override {
        ++polls;
        return Verdict::Converged;
    }
    int polls = 0;
};

/// @brief Votes Stop every poll.
class AlwaysStop : public cpu::Criterion {
  public:
    std::string name() const override { return "always_stop"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState&) override {
        ++polls;
        return Verdict::Stop;
    }
    int polls = 0;
};

// ── The loop node under test — a bare CPUDriver over scf::Phase ───────────────
//
// Mirrors `OuterLoop` in test_sub_driver.cpp: the subclass supplies a tiny `run()` that
// composes the protected loop primitives, and `validate_components()` requires nothing.
// Instantiating this at all is the proof that `PhaseTraits<scf::Phase>` is specialized —
// without the specialization the inherited categorical setters do not compile.
struct ScfPhaseProbe : cpu::CPUDriver<ATV::scf::Phase> {
    /// @brief Walk `budget` iterations, polling the composed termination verdict after the
    ///        stages of each. Returns the number of iterations actually executed.
    int run(cpu::Frame& frame, int budget) {
        setup_pass(frame);
        int iters = 0;
        for (int s = 0; s < budget; ++s) {
            mutable_outer_status().reset();
            step_once(s);
            ++iters;
            if (poll_criteria_(s) != Verdict::Continue)
                break;
        }
        teardown_pass();
        return iters;
    }

  protected:
    void validate_components() const override {} // a bare loop requires nothing
};

} // namespace

// ── 1. scf::Phase enum shape — slot indices through the typed entry points ────

TEST(ScfPhasePlumbing, SlotIndices) {
    // The one enum→slot conversion, exercised through slot_of only — no static_cast<int>.
    EXPECT_EQ(slot_of(scf::Phase::fock), 0);
    EXPECT_EQ(slot_of(scf::Phase::mix), 1);
    EXPECT_EQ(slot_of(scf::Phase::diagonalize), 2);
    EXPECT_EQ(slot_of(scf::Phase::density), 3);
    EXPECT_EQ(slot_of(scf::Phase::check), 4);
    EXPECT_EQ(slot_count<scf::Phase>(), 5);

    // Constexpr all the way — the ordering and count are a compile-time contract.
    static_assert(slot_of(scf::Phase::fock) == 0, "fock is slot 0");
    static_assert(slot_of(scf::Phase::check) == 4,
                  "check is the last real slot");
    static_assert(slot_count<scf::Phase>() == 5,
                  "five phases: fock, mix, diagonalize, density, check");
}

// ── 2. PhaseKind routing — kind_of / phase_kind_name / make_binding ───────────

TEST(ScfPhasePlumbing, PhaseKindRouting) {
    // ADL tag dispatch, exactly like evaluate::kind_of.
    EXPECT_EQ(kind_of(scf::Phase::fock), PhaseKind::SCF);
    EXPECT_EQ(kind_of(scf::Phase::check), PhaseKind::SCF);
    static_assert(kind_of(scf::Phase::fock) == PhaseKind::SCF,
                  "every scf::Phase routes to PhaseKind::SCF");

    EXPECT_STREQ(phase_kind_name(PhaseKind::SCF), "SCF");

    // make_binding fuses the discriminant with the slot so the two can never disagree.
    const DefaultBinding b = make_binding(scf::Phase::fock, 0);
    EXPECT_EQ(b.kind, PhaseKind::SCF);
    EXPECT_EQ(b.phase, slot_of(scf::Phase::fock));
    EXPECT_EQ(b.order, 0);
    static_assert(make_binding(scf::Phase::fock, 0).kind == PhaseKind::SCF,
                  "the SCF workflow discriminant travels with the binding");
}

// ── 3. PhaseTraits<scf::Phase> — proven by instantiating the loop node ────────

TEST(ScfPhasePlumbing, PhaseTraitsSpecialization) {
    // Compare enum values directly — no casts.
    static_assert(cpu::PhaseTraits<scf::Phase>::eval_phase == scf::Phase::fock,
                  "the Hamiltonian/energy slot is fock");
    static_assert(cpu::PhaseTraits<scf::Phase>::neighbor_phase
                      == scf::Phase::fock,
                  "SCF has no neighbor list — it folds into fock");
    static_assert(cpu::PhaseTraits<scf::Phase>::end_phase == scf::Phase::check,
                  "the end-of-iteration slot is check");

    EXPECT_EQ(cpu::PhaseTraits<scf::Phase>::eval_phase, scf::Phase::fock);
    EXPECT_EQ(cpu::PhaseTraits<scf::Phase>::neighbor_phase, scf::Phase::fock);
    EXPECT_EQ(cpu::PhaseTraits<scf::Phase>::end_phase, scf::Phase::check);

    // The whole point of the specialization: `CPUDriver<scf::Phase>` must instantiate.
    static_assert(std::is_base_of_v<cpu::CPUDriver<scf::Phase>, ScfPhaseProbe>,
                  "the loop node instantiates over scf::Phase");
    ScfPhaseProbe probe; // and it is actually constructible
    (void)probe;
    SUCCEED();
}

// ── 4. Criterion is an abstract Component kind ────────────────────────────────

TEST(ScfPhasePlumbing, CriterionIsAbstractComponentKind) {
    static_assert(std::is_base_of_v<cpu::Component, cpu::Criterion>,
                  "Criterion is a Component kind");
    static_assert(
        std::is_abstract_v<cpu::Criterion>,
        "Criterion is a pure-virtual interface (verdict() unimplemented)");
    EXPECT_TRUE((std::is_base_of_v<cpu::Component, cpu::Criterion>));
    EXPECT_TRUE(std::is_abstract_v<cpu::Criterion>);
}

// ── 5. The driver pulls verdict() once per iteration, AFTER the stages ────────
//
// Exercises BOTH registration forms: an owned criterion (rvalue) and a borrowed one
// (lvalue, so the test can read its poll count back). A Continue-only vote never ends the
// loop, so the budget is exhausted and the loop — not the component — reports NOT
// converged.

TEST(ScfPhasePlumbing, CriterionPolledOncePerIterationAfterStages) {
    cpu::Frame frame = atv_test::make_trivial_frame();

    StageProbe stage;
    ContinueCriterion watcher{&stage};

    ScfPhaseProbe probe;
    probe.on(scf::Phase::fock)
        .add_ref(stage); // a real stage to walk each iteration
    // Owning form must COMPILE and run — a harmless owned Continue vote (nullptr stage:
    // it records nothing, only proves add_criterion(rvalue) is accepted and polled).
    probe.add_criterion(ContinueCriterion{nullptr});
    probe.add_criterion_ref(watcher); // borrowing form — inspected below
    probe.build();

    const int iters = probe.run(frame, /*budget=*/3);

    // The stage walked once per iteration ...
    EXPECT_EQ(stage.launches, iters);
    // ... and the criterion was pulled once per iteration, in lockstep.
    EXPECT_EQ(watcher.polls, iters);
    // ... after the stage of that iteration ran (ordering witness).
    EXPECT_EQ(watcher.stage_at_first_poll, 1)
        << "the driver polled the criterion before running the iteration's "
           "stages";
}

// ── 6. A Converged vote ends the loop, and ONLY the driver flips converged() ──

TEST(ScfPhasePlumbing, ConvergedCriterionStopsLoopViaTheDriverPoll) {
    cpu::Frame frame = atv_test::make_trivial_frame();

    // Continue on polls 0,1 then Converged on poll 2 — the loop ends on iteration 2.
    ConvergeAfter crit{/*at=*/2};

    ScfPhaseProbe probe;
    probe.add_criterion_ref(crit);
    probe.build();

    const int iters = probe.run(frame, /*budget=*/10);

    EXPECT_EQ(iters, 3)
        << "budget was not exhausted; the criterion ended the loop";
    EXPECT_EQ(crit.polls, 3);
    // The component only ever RETURNED Converged; the driver's poll is what set the flag.
    EXPECT_TRUE(probe.converged());
    EXPECT_EQ(probe.stop_reason(), "converge_after");

    // Control: a Continue-only run leaves converged() false — the loop, not any
    // component, owns the verdict.
    StageProbe stage2;
    ContinueCriterion never{&stage2};
    ScfPhaseProbe probe2;
    probe2.add_criterion_ref(never);
    probe2.build();
    const int iters2 = probe2.run(frame, /*budget=*/4);
    EXPECT_EQ(iters2, 4);
    EXPECT_FALSE(probe2.converged());
    EXPECT_TRUE(probe2.stop_reason().empty());
}

// ── 7. Severity composition — Stop beats Converged in the same iteration ──────

TEST(ScfPhasePlumbing, StopBeatsConvergedInComposition) {
    cpu::Frame frame = atv_test::make_trivial_frame();

    // Both vote on the very first iteration: Converged and Stop together.
    AlwaysConverged converged_vote;
    AlwaysStop stop_vote;

    ScfPhaseProbe probe;
    probe.add_criterion_ref(converged_vote); // registered first ...
    probe.add_criterion_ref(stop_vote);      // ... Stop still wins by severity
    probe.build();

    const int iters = probe.run(frame, /*budget=*/5);

    EXPECT_EQ(iters, 1) << "the loop ended on the first iteration";
    // Stop > Converged: a run that must Stop cannot be reported as a success.
    EXPECT_FALSE(probe.converged());
    EXPECT_EQ(probe.stop_reason(), "always_stop")
        << "the stop-voting criterion is credited, regardless of registration "
           "order";
}
