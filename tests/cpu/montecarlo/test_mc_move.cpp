/// @file tests/cpu/mc/test_mc_move.cpp
/// @brief Fast unit tests for the MC move + driver assembly
///        (mc-sampling-02-move): TranslateMove's per-launch behaviour and the
///        MC driver's build-time validation + config-constructor guards.
///
/// The Mover is exercised
/// by calling `launch()` directly against a hand-built context; the driver is
/// exercised only through `build()` (plan compile + validation, no stepping).
/// Mock-only MC::run tests below check its decision/observation boundary.
/// End-to-end physical sampler behaviour (bootstrap energy, accept/reject/rollback,
/// determinism, Metropolis distribution) belongs to the validation tier in
/// sub-spec mc-sampling-03-validate, not to this ms-scale suite.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/kmc.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/polymer_moves.h"
#include "atomiverse/cpu/montecarlo/transaction.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {
constexpr double kEps = 0.0104, kSigma = 3.4, kCut = 8.0, kSkin = 2.0;
} // namespace

// ── TranslateMove::launch — unit behaviour (no driver) ────────────────────

TEST(TranslateMoveUnit, DisplacesOneAtomAndRecordsChangeset) {
    cpu::Frame frame(3);
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_x)[i] = static_cast<double>(i);
        frame.span(ATV::kAtom_y)[i] = 0.0;
        frame.span(ATV::kAtom_z)[i] = 0.0;
    }
    cpu::Frame before = frame;

    std::mt19937_64 rng(123u);
    cpu::mc::Transaction txn;
    cpu::Context ctx;
    ctx.mc.rng = &rng;
    ctx.mc.proposal = &txn;

    cpu::mc::TranslateMove mv{0.5};
    cpu::LaunchContext lctx{ctx, frame, /*stage=*/0, /*step=*/0};
    mv.launch(lctx);

    // Exactly one atom recorded, index in range, verb stamped.
    ASSERT_EQ(txn.size(), 1u);
    EXPECT_EQ(txn.verb, cpu::mc::Transaction::Verb::Displace);
    const int i = txn.touched(0);
    ASSERT_GE(i, 0);
    ASSERT_LT(i, 3);

    // The patch holds the pre-move row; the frame holds the post-move one.
    EXPECT_EQ(txn.patch().span(ATV::kAtom_x)[0], before.span(ATV::kAtom_x)[i]);
    EXPECT_EQ(txn.patch().span(ATV::kAtom_y)[0], before.span(ATV::kAtom_y)[i]);
    EXPECT_EQ(txn.patch().span(ATV::kAtom_z)[0], before.span(ATV::kAtom_z)[i]);

    // Displacement is within the symmetric box [-δ, δ] per axis (the box is
    // degenerate here, so no wrap interferes with the bound).
    EXPECT_LE(
        std::abs(frame.span(ATV::kAtom_x)[i] - before.span(ATV::kAtom_x)[i]),
        0.5);
    EXPECT_LE(
        std::abs(frame.span(ATV::kAtom_y)[i] - before.span(ATV::kAtom_y)[i]),
        0.5);
    EXPECT_LE(
        std::abs(frame.span(ATV::kAtom_z)[i] - before.span(ATV::kAtom_z)[i]),
        0.5);

    // Only atom i moved; the others are untouched.
    for (int k = 0; k < 3; ++k) {
        if (k == i)
            continue;
        EXPECT_EQ(frame.span(ATV::kAtom_x)[k], before.span(ATV::kAtom_x)[k]);
        EXPECT_EQ(frame.span(ATV::kAtom_y)[k], before.span(ATV::kAtom_y)[k]);
        EXPECT_EQ(frame.span(ATV::kAtom_z)[k], before.span(ATV::kAtom_z)[k]);
    }
}

// The wrap back into the cell counts its lattice translation into the image
// (LAMMPS Domain::pbc), so r + H·image moves by exactly the proposed
// displacement; a rejected move (rollback) restores position AND image.
TEST(TranslateMoveUnit, WrapCountsImageAndRollbackRestoresIt) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(4.0);
    frame.span(ATV::kAtom_x)[0] = 0.05;
    frame.span(ATV::kAtom_y)[0] = 3.95;
    frame.span(ATV::kAtom_z)[0] = 2.0;
    auto unwrapped = [&](int a) {
        const double r[3] = {frame.span(ATV::kAtom_x)[0],
                             frame.span(ATV::kAtom_y)[0],
                             frame.span(ATV::kAtom_z)[0]};
        return r[a] + 4.0 * frame.span(ATV::kAtom_image)(0, a);
    };

    std::mt19937_64 rng(2024u);
    cpu::mc::Transaction txn;
    cpu::Context ctx;
    ctx.mc.rng = &rng;
    ctx.mc.proposal = &txn;
    constexpr double kDelta = 1.0;
    cpu::mc::TranslateMove mv{kDelta};
    cpu::LaunchContext lctx{ctx, frame, /*stage=*/0, /*step=*/0};

    int crossed = 0, restored = 0;
    for (int trial = 0; trial < 64; ++trial) {
        double u0[3], r0[3];
        ATV::ImageInt n0[3];
        for (int a = 0; a < 3; ++a) {
            u0[a] = unwrapped(a);
            n0[a] = frame.span(ATV::kAtom_image)(0, a);
        }
        r0[0] = frame.span(ATV::kAtom_x)[0];
        r0[1] = frame.span(ATV::kAtom_y)[0];
        r0[2] = frame.span(ATV::kAtom_z)[0];
        mv.launch(lctx);
        bool image_moved = false;
        for (int a = 0; a < 3; ++a) {
            EXPECT_LE(std::abs(unwrapped(a) - u0[a]), kDelta + 1e-12)
                << "trial " << trial << " axis " << a;
            image_moved |= frame.span(ATV::kAtom_image)(0, a) != n0[a];
        }
        crossed += image_moved ? 1 : 0;
        if (image_moved && restored < 4) { // reject: everything comes back
            txn.rollback(frame);
            ++restored;
            EXPECT_EQ(frame.span(ATV::kAtom_x)[0], r0[0]);
            EXPECT_EQ(frame.span(ATV::kAtom_y)[0], r0[1]);
            EXPECT_EQ(frame.span(ATV::kAtom_z)[0], r0[2]);
            for (int a = 0; a < 3; ++a)
                EXPECT_EQ(frame.span(ATV::kAtom_image)(0, a), n0[a]);
        }
    }
    EXPECT_GT(crossed, 4); // the start sits next to two faces
    EXPECT_EQ(restored, 4);
}

TEST(TranslateMoveUnit, BootstrapPassIsNoOp) {
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 4.0;
    cpu::Frame before = frame;

    std::mt19937_64 rng(1u);
    cpu::mc::Transaction txn;
    cpu::Context ctx;
    ctx.mc.rng = &rng;
    ctx.mc.proposal = &txn;

    cpu::mc::TranslateMove mv{0.5};
    cpu::LaunchContext lctx{ctx, frame, /*stage=*/0, /*step=*/-1}; // bootstrap
    mv.launch(lctx);

    EXPECT_EQ(txn.size(), 0u); // proposed nothing
    EXPECT_EQ(frame.span(ATV::kAtom_x)[0], before.span(ATV::kAtom_x)[0]);
    EXPECT_EQ(frame.span(ATV::kAtom_x)[1], before.span(ATV::kAtom_x)[1]);
}

TEST(TranslateMoveUnit, EmptyFrameProposesNothing) {
    // GCMC can empty the cell (μ very negative), and the shared MoveSelector
    // still hands the trial to TranslateMove. With `nlocal() == 0` the move has
    // no atom to pick: it must stamp `Verb::None` — the same "empty proposal"
    // shape InsertDeleteMove already uses for a deletion from N = 0
    // (src/cpu/montecarlo/insert_delete_move.cpp:103) — so the OutcomePolicy
    // rejects the trial without a Boltzmann draw.
    //
    // PRE-FIX THIS IS UNDEFINED BEHAVIOUR, not merely a wrong verb:
    // src/cpu/montecarlo/translate_move.cpp:46 builds
    // `std::uniform_int_distribution<int> pick(0, n - 1)` with n == 0, whose
    // precondition is a <= b, then indexes the frame with the garbage draw.
    // Observed pre-fix on this Release build: the garbage index escapes as
    // std::out_of_range("Frame pack row is outside capacity") from
    // src/cpu/communication/atom_pack.cpp:182 — a capacity guard, not a
    // memory-safety guarantee (see the capacity > 0 case below).
    cpu::Frame frame(0);
    frame.box.set_cell(Mat3::diagonal(20.0, 20.0, 20.0));

    std::mt19937_64 rng(7u);
    cpu::mc::Transaction txn;
    // The driver reuses ONE Transaction across trials, so the verb read by the
    // policy is whatever the last trial left behind. Seed a stale `Displace` to
    // prove the guard actively stamps `None` rather than leaving the default.
    txn.verb = cpu::mc::Transaction::Verb::Displace;
    cpu::Context ctx;
    ctx.mc.rng = &rng;
    ctx.mc.proposal = &txn;

    cpu::mc::TranslateMove mv{0.5};
    cpu::LaunchContext lctx{ctx, frame, /*stage=*/0, /*step=*/0};
    mv.launch(lctx);

    EXPECT_EQ(txn.verb, cpu::mc::Transaction::Verb::None);
    EXPECT_EQ(txn.size(), 0u);    // nothing touched, nothing to roll back
    EXPECT_EQ(frame.nlocal(), 0); // and the population is untouched
}

TEST(TranslateMoveUnit, EmptiedFrameWithSpareCapacityProposesNothing) {
    // The shape GCMC actually reaches: the chain grew the frame, then deleted
    // its way down to N = 0, so `capacity() > 0` while `nlocal() == 0` and row
    // 0 still holds the last deleted atom's stale data.
    //
    // Pre-fix the index comes from `pick(0, -1)`, whose value is unspecified:
    // on this build it lands past the capacity and surfaces as the packer's
    // std::out_of_range, but nothing bounds it to the used rows. A draw that
    // falls inside a larger capacity is snapshotted, displaced and stamped
    // `Displace` — a trial move on an atom that is not in the system, judged
    // and possibly committed by the policy, with no diagnostic at all.
    cpu::Frame frame(1);
    frame.box.set_cell(Mat3::diagonal(20.0, 20.0, 20.0));
    frame.span(ATV::kAtom_x)[0] = 1.0;
    frame.span(ATV::kAtom_y)[0] = 2.0;
    frame.span(ATV::kAtom_z)[0] = 3.0;
    frame.set_nlocal(0); // the last atom was deleted; the row is stale

    std::mt19937_64 rng(11u);
    cpu::mc::Transaction txn;
    txn.verb = cpu::mc::Transaction::Verb::Displace; // stale verb, as above
    cpu::Context ctx;
    ctx.mc.rng = &rng;
    ctx.mc.proposal = &txn;

    cpu::mc::TranslateMove mv{0.5};
    cpu::LaunchContext lctx{ctx, frame, /*stage=*/0, /*step=*/0};
    mv.launch(lctx);

    EXPECT_EQ(txn.verb, cpu::mc::Transaction::Verb::None);
    EXPECT_EQ(txn.size(), 0u);
    EXPECT_EQ(frame.nlocal(), 0);
    // The out-of-system row must not have been touched either.
    EXPECT_EQ(frame.span(ATV::kAtom_x)[0], 1.0);
    EXPECT_EQ(frame.span(ATV::kAtom_y)[0], 2.0);
    EXPECT_EQ(frame.span(ATV::kAtom_z)[0], 3.0);
}

TEST(TranslateMoveUnit, RejectsNonPositiveDelta) {
    EXPECT_THROW((cpu::mc::TranslateMove{0.0}), std::invalid_argument);
    EXPECT_THROW((cpu::mc::TranslateMove{-0.1}), std::invalid_argument);
}

// ── MC driver — build-time validation + config guards (no run) ────────────

TEST(MCBuildUnit, RejectsNonPositiveTemperature) {
    EXPECT_THROW((cpu::mc::MC{0.0, 1u}), std::invalid_argument);
    EXPECT_THROW((cpu::mc::MC{-300.0, 1u}), std::invalid_argument);
}

TEST(MCBuildUnit, ThrowsWithoutMover) {
    cpu::mc::MC mc{300.0, 1u};
    mc.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    mc.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    EXPECT_THROW(mc.build(), std::runtime_error);
}

TEST(MCBuildUnit, ThrowsWithoutEnergyEvaluator) {
    cpu::mc::MC mc{300.0, 1u};
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    EXPECT_THROW(mc.build(), std::runtime_error);
}

TEST(MCBuildUnit, ThrowsWithPairButNoNeighborList) {
    cpu::mc::MC mc{300.0, 1u};
    mc.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    EXPECT_THROW(mc.build(), std::runtime_error);
}

TEST(MCBuildUnit, SucceedsWhenFullyAssembled) {
    cpu::mc::MC mc{300.0, 1u};
    mc.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    mc.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    EXPECT_NO_THROW(mc.build());
}

TEST(MCBuildUnit, SetMoverIsIdempotentOnSameClass) {
    cpu::mc::MC mc{300.0, 1u};
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    // Singular-by-name: re-setting the same Mover class is a no-op (v1 ships
    // only TranslateMove, so the different-class throw path is not exercisable).
    EXPECT_NO_THROW(mc.set_mover(cpu::mc::TranslateMove{0.5}));
}

namespace {
struct TrialMover : cpu::Mover {
    std::string name() const override { return "trial_mover_probe"; }
    void launch(const cpu::LaunchContext& ctx) override {
        auto& tx = *ctx.runtime.mc.proposal;
        tx.begin(cpu::mc::Transaction::Verb::Displace, ctx.frame);
        tx.log_proposal_ratio = 0.0;
        tx.record_touch(ctx.frame, 0);
        ctx.frame.span(kAtom_x)[0] = 1.0;
    }
};
struct TrialEnergy : cpu::Potential {
    bool fail = false;
    int calls = 0;
    std::string name() const override { return "trial_energy_probe"; }
    void launch(const cpu::LaunchContext& ctx) override {
        EXPECT_TRUE(ctx.runtime.exec.trial_evaluation);
        EXPECT_FALSE(ctx.forces_requested);
        EXPECT_FALSE(ctx.stress_requested);
        ++calls;
        if (fail)
            throw std::runtime_error("trial evaluator failed");
        const double energy = 100.0 * ctx.frame.span(kAtom_x)[0];
        ctx.frame.meta.get(kEnergy) += energy;
        ctx.frame.meta.get(kEnergyPair) = energy;
        ctx.frame.span(kAtom_charge)[0] = energy;
    }
};
struct TrialObserver : cpu::Component {
    int calls = 0;
    std::string name() const override { return "trial_observer_probe"; }
    void launch(const cpu::LaunchContext& ctx) override {
        EXPECT_FALSE(ctx.runtime.exec.trial_evaluation);
        EXPECT_DOUBLE_EQ(ctx.frame.span(kAtom_x)[0], 0.0);
        EXPECT_DOUBLE_EQ(ctx.frame.meta.get(kEnergy), 0.0);
        EXPECT_DOUBLE_EQ(ctx.frame.meta.get(kEnergyPair), 0.0);
        EXPECT_DOUBLE_EQ(ctx.frame.span(kAtom_charge)[0], 0.0);
        ++calls;
    }
};
} // namespace

// Orchestration unit: mocks exercise MC::run's decision/observation boundary,
// with no force field, neighbor build, physical trajectory or stochastic oracle.
TEST(MCUnit, RejectedTrialRestoresOutputsBeforeObservation) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    TrialMover mover;
    TrialEnergy energy;
    TrialObserver observer;
    cpu::mc::MC mc(1.0, 1);
    mc.on(sampling::Phase::propose).add_ref(mover);
    mc.set_evaluator_ref(energy);
    mc.on(sampling::Phase::end_of_step).add_ref(observer);
    mc.build();
    mc.run(frame, 1); // uphill 100 eV at 1 K is deterministically rejected
    EXPECT_EQ(observer.calls, 1); // bootstrap was energy-only
    EXPECT_EQ(energy.calls, 3);   // bootstrap, proposal, restored configuration
    EXPECT_EQ(mc.accepts(), 0);
}

TEST(MCUnit, FailedBootstrapCanBeRetriedWithoutObservingIt) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    TrialMover mover;
    TrialEnergy energy;
    TrialObserver observer;
    cpu::mc::MC mc(1.0, 1);
    mc.on(sampling::Phase::propose).add_ref(mover);
    mc.set_evaluator_ref(energy);
    mc.on(sampling::Phase::end_of_step).add_ref(observer);
    mc.build();
    energy.fail = true;
    EXPECT_THROW(mc.run(frame, 0), std::runtime_error);
    energy.fail = false;
    mc.run(frame, 0);
    EXPECT_EQ(energy.calls, 2);
    EXPECT_EQ(observer.calls, 0);
}

TEST(MCUnit,
     NestedTrialUsesTheSameDecisionBoundaryAndRestoresParentServices) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10);
    TrialMover mover;
    TrialEnergy energy;
    TrialObserver observer;
    cpu::mc::MC mc(1.0, 1);
    mc.on(sampling::Phase::propose).add_ref(mover);
    mc.set_evaluator_ref(energy);
    mc.on(sampling::Phase::end_of_step).add_ref(observer);
    Plan<cpu::Component> outer;
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(outer);
    runtime.set_frame(frame);
    runtime.build_components();
    mc.build_on(runtime);
    const auto saved = runtime.context().loop;
    EXPECT_EQ(mc.run_nested_erased(runtime.context(), frame, 1), 1);
    EXPECT_EQ(observer.calls, 1);
    EXPECT_EQ(energy.calls, 3);
    EXPECT_EQ(mc.accepts(), 0);
    EXPECT_EQ(runtime.context().mc.proposal, nullptr);
    EXPECT_EQ(runtime.context().loop.status, saved.status);
    EXPECT_EQ(runtime.context().loop.metrics, saved.metrics);
    mc.destroy_on(runtime);
    runtime.destroy_components();
}

TEST(KineticEventDraw, ScalesWaitingTimeWithoutChangingTheSelectedEvent) {
    const std::array<double, 3> rates{0, 2, 3};
    const std::array<double, 3> scaled{0, 4, 6};
    std::mt19937_64 rng{7};
    std::mt19937_64 scaled_rng{7};
    cpu::mc::LinearKineticSolver solver;
    const auto event = solver.draw(rates, rng);
    const auto other = solver.draw(scaled, scaled_rng);
    ASSERT_TRUE(event);
    ASSERT_TRUE(other);
    EXPECT_NE(event->index, 0u);
    EXPECT_EQ(event->index, other->index);
    EXPECT_GT(event->waiting_time, 0.0);
    EXPECT_DOUBLE_EQ(event->waiting_time, 2 * other->waiting_time);
}

TEST(KineticEventDraw, ZeroRateLeavesTheRngUnchanged) {
    const std::array<double, 2> rates{0, 0};
    std::mt19937_64 rng{7};
    const auto initial = rng;
    cpu::mc::LinearKineticSolver solver;
    EXPECT_FALSE(solver.draw(rates, rng));
    EXPECT_EQ(rng, initial);
}

namespace {

class TrialEvent final : public cpu::mc::Event {
  public:
    bool fail = false;

    void rates(const cpu::Frame&, std::vector<double>& out) override {
        out.push_back(1.0);
    }

    cpu::mc::EventResult apply(cpu::Frame& frame, std::size_t,
                               cpu::mc::Transaction& transaction) override {
        transaction.record_touch(frame, 0);
        frame.span(kAtom_x)[0] = 2.0;
        transaction.record_touch(frame, 0);
        frame.span(kAtom_x)[0] = 3.0;
        frame.meta.set(cpu::mc::kKineticState, std::int64_t{1});
        if (fail)
            throw std::runtime_error("event could not complete");
        return {42, MetaF64x3{2.0, 0.0, 0.0}};
    }
};

class FixedSolver final : public cpu::mc::KineticSolver {
  public:
    int calls = 0;

  protected:
    cpu::mc::KineticDraw draw_event(std::span<const double>, double,
                                    std::mt19937_64&) override {
        ++calls;
        return {0, 2.0};
    }
};

} // namespace

TEST(KMCUnit, EventFailureRestoresFrameAndClockThroughSharedRun) {
    cpu::Frame frame{1};
    frame.span(kAtom_x)[0] = 1.0;
    frame.meta.insert(cpu::mc::kKineticState, std::int64_t{0});
    cpu::mc::KMC kmc{7u};
    const auto event = std::make_shared<TrialEvent>();
    event->fail = true;
    kmc.set_event(event);
    const auto solver = std::make_shared<FixedSolver>();
    kmc.set_solver(solver);
    cpu::mc::MC& driver = kmc;

    EXPECT_THROW(driver.run(frame, 1), std::logic_error);
    EXPECT_FALSE(frame.meta.contains(cpu::mc::kKineticTime.key));

    kmc.build();
    EXPECT_THROW(driver.run(frame, 1), std::runtime_error);
    EXPECT_EQ(solver->calls, 1);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
    EXPECT_DOUBLE_EQ(frame.meta.get(cpu::mc::kKineticTime), 0.0);
    EXPECT_EQ(frame.meta.get(cpu::mc::kKineticState), 0);
    EXPECT_FALSE(frame.meta.contains(cpu::mc::kKineticEventId.key));

    event->fail = false;
    ASSERT_EQ(driver.run(frame, 1), 1);
    EXPECT_EQ(frame.meta.get(cpu::mc::kKineticEventId), 42);
    EXPECT_DOUBLE_EQ(frame.meta.get(cpu::mc::kKineticWaitingTime), 2.0);
    EXPECT_TRUE(frame.meta.get(cpu::mc::kKineticHasDisplacement));
    EXPECT_EQ(frame.meta.get(cpu::mc::kKineticDisplacement),
              (MetaF64x3{2.0, 0.0, 0.0}));
}

namespace {
Vec3 polymer_point(cpu::Frame& frame, int i) {
    return {frame.span(kAtom_x)[i], frame.span(kAtom_y)[i], frame.span(kAtom_z)[i]};
}

void initialize_polymer(cpu::Frame& frame, const std::vector<Vec3>& points) {
    frame.box = cpu::Box::cubic(30);
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_tag)[i] = i + 11;
        frame.span(kAtom_x)[i] = points[i].x();
        frame.span(kAtom_y)[i] = points[i].y();
        frame.span(kAtom_z)[i] = points[i].z();
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int i = 1; i < frame.nlocal(); ++i)
        frame.add_bond(i + 10, i + 11, 0);
    frame.build_special_list();
}

struct CountingMover : cpu::Mover {
    explicit CountingMover(int& calls) : calls(calls) {}
    int& calls;
    std::string name() const override { return "count_draw"; }
    void launch(const cpu::LaunchContext&) override { ++calls; }
};

struct CandidateEnergy : cpu::Potential {
    std::string name() const override { return "cbmc_candidate_energy"; }
    int calls = 0;
    bool fail = false;
    void launch(const cpu::LaunchContext& ctx) override {
        ++calls;
        EXPECT_TRUE(ctx.runtime.exec.trial_evaluation);
        EXPECT_FALSE(ctx.forces_requested);
        if (fail)
            throw std::runtime_error("candidate score failed");
        ctx.frame.meta.get(kEnergy) +=
            (polymer_point(ctx.frame, 1) - polymer_point(ctx.frame, 0)).norm_squared();
    }
};
} // namespace

TEST(PolymerMoveUnit, SelectorRunsOneLeafAndDoesNotRetryEmptyProposal) {
    cpu::Frame frame(0);
    cpu::Context context;
    std::mt19937_64 rng(335);
    context.mc.rng = &rng;
    cpu::LaunchContext launch{context, frame, 0, 0};
    int a = 0;
    int b = 0;
    auto left = std::make_shared<cpu::mc::MoveSelector>();
    left->add(CountingMover(a), 1);
    cpu::mc::MoveSelector mix;
    mix.add_child(left, 1).add(CountingMover(b), 3);
    EXPECT_THROW(mix.add(CountingMover(b), std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    for (int i = 0; i < 1000; ++i)
        mix.launch(launch);
    EXPECT_EQ(a + b, 1000);
    EXPECT_NEAR(a, 250, 55);
}

TEST(PolymerMoveUnit, TorsionPreservesCrossBoundaryBondsAndRollsBack) {
    cpu::Frame frame(4);
    initialize_polymer(frame,
                       {{28, 15, 15}, {29.5, 15, 15}, {0.5, 16, 15}, {1.5, 16, 16}});
    const cpu::Frame before = frame;
    cpu::Context context;
    std::mt19937_64 rng(1232);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    cpu::mc::TorsionMove move({{11, 12}}, 1.2);
    cpu::LaunchContext launch{context, frame, 0, 0};
    move.launch(launch);
    ASSERT_EQ(txn.size(), 3u);
    EXPECT_DOUBLE_EQ(txn.require_log_proposal_ratio(), 0.0);
    EXPECT_GT(std::abs(frame.span(kAtom_z)[2] - before.span(kAtom_z)[2]), 1e-3);
    const auto length = [](cpu::Frame& f, int i) {
        auto v = polymer_point(f, i) - polymer_point(f, i - 1);
        f.box.min_image(v.x(), v.y(), v.z());
        return v.norm();
    };
    auto before_coords = before;
    for (int i = 1; i < 4; ++i)
        EXPECT_NEAR(length(frame, i), length(before_coords, i), 1e-5);
    txn.rollback(frame);
    for (int i = 0; i < frame.nlocal(); ++i) {
        EXPECT_EQ(frame.span(kAtom_z)[i], before.span(kAtom_z)[i]);
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(frame.span(kAtom_image)(i, d), before.span(kAtom_image)(i, d));
    }
}

TEST(PolymerMoveUnit, ConRotClosesItsAnchorsAndPreservesAngles) {
    cpu::Frame frame(8);
    initialize_polymer(frame, {{3, 6, 10}, {4, 6, 10}, {5, 7, 10}, {6, 7, 12},
                               {7, 9, 13}, {8, 9, 12}, {9, 9, 13}, {10, 8, 12}});
    auto before = frame;
    cpu::Context context;
    std::mt19937_64 rng(973);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    cpu::mc::ConRotMove move({{11, 12, 13, 14, 15, 16, 17, 18}}, 0.02);
    cpu::LaunchContext launch{context, frame, 0, 0};
    move.launch(launch);
    ASSERT_EQ(txn.size(), 4u);
    EXPECT_TRUE(std::isfinite(txn.require_log_proposal_ratio()));
    EXPECT_GT((polymer_point(frame, 2) - polymer_point(before, 2)).norm(), 1e-6);
    for (const int i : {0, 1, 6, 7})
        EXPECT_DOUBLE_EQ((polymer_point(frame, i) - polymer_point(before, i)).norm(), 0);
    for (int i = 1; i < frame.nlocal(); ++i) {
        const auto b = polymer_point(frame, i) - polymer_point(frame, i - 1);
        const auto b0 = polymer_point(before, i) - polymer_point(before, i - 1);
        EXPECT_NEAR(b.norm(), b0.norm(), 1e-5);
        if (i + 1 < frame.nlocal()) {
            const auto next = polymer_point(frame, i + 1) - polymer_point(frame, i);
            const auto next0 = polymer_point(before, i + 1) - polymer_point(before, i);
            EXPECT_NEAR(b.normalized().dot(next.normalized()),
                        b0.normalized().dot(next0.normalized()), 1e-5);
        }
    }
}

TEST(PolymerMoveUnit, ParallelRotationTranslatesOutgoingChain) {
    cpu::Frame frame(7);
    initialize_polymer(frame, {{3, 6, 10}, {4, 6, 10}, {5, 7, 10}, {6, 7, 12},
                               {7, 9, 13}, {8, 9, 12}, {9, 9, 13}});
    auto before = frame;
    cpu::Context context;
    std::mt19937_64 rng(973);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    cpu::mc::ParallelRotationMove move({{11, 12, 13, 14, 15, 16}}, 0.02);
    cpu::LaunchContext launch{context, frame, 0, 0};
    move.launch(launch);
    ASSERT_EQ(txn.size(), 5u);
    const auto shift = polymer_point(frame, 3) - polymer_point(before, 3);
    EXPECT_GT(shift.norm(), 1e-6);
    for (int i = 4; i < frame.nlocal(); ++i) {
        const auto delta = polymer_point(frame, i) - polymer_point(before, i);
        EXPECT_NEAR((delta - shift).norm(), 0, 1e-5);
    }
}

TEST(PolymerMoveUnit, CBMCReplaysEnergyAndRollsBackFailedCandidate) {
    cpu::Frame frame(3);
    initialize_polymer(frame, {{7, 12, 12}, {8, 12, 12}, {9, 12, 12}});
    const auto start = polymer_point(frame, 1);
    CandidateEnergy energy;
    cpu::mc::MC mc(1.0, 975);
    mc.set_evaluator_ref(energy);
    mc.set_mover(cpu::mc::ExtendedCBMCMove({{11, 12, 13}}, 3, 0.5, 1.0));
    mc.build();
    ASSERT_EQ(mc.run(frame, 0), 0);
    ASSERT_EQ(energy.calls, 1);
    energy.fail = true;
    EXPECT_THROW(mc.run(frame, 1), std::runtime_error);
    EXPECT_DOUBLE_EQ((polymer_point(frame, 1) - start).norm(), 0);
    EXPECT_EQ(mc.attempts(), 0);
    energy.fail = false;
    EXPECT_EQ(mc.run(frame, 1), 1);
    EXPECT_GE(energy.calls, 9); // bootstrap, failed score, two trial pools, final score
    const auto energy_ref =
        (polymer_point(frame, 1) - polymer_point(frame, 0)).norm_squared();
    EXPECT_NEAR(mc.current_energy(), energy_ref, 1e-10);
}

TEST(PolymerMoveUnit, GeneralizedReptationRestoresBondAndExclusionListsOnRollback) {
    cpu::Frame frame(6);
    initialize_polymer(frame, {{3, 6, 10}, {4, 6, 10}, {5, 6, 10},
                               {6, 9, 12}, {7, 9, 12}, {8, 9, 12}});
    // Split the chain after row 2. Its bond to row 3 is stored on row 2.
    frame.span(kAtom_num_bond)[2] = 0;
    frame.build_special_list();

    cpu::Context context;
    std::mt19937_64 rng(1934);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    cpu::mc::GeneralizedReptationMove move(0.7);
    move.build({context, frame});
    const auto before = frame;
    cpu::LaunchContext launch{context, frame, 0, 0};
    for (int i = 0; i < 10 && !txn.connectivity_changed; ++i)
        move.launch(launch);
    ASSERT_TRUE(txn.connectivity_changed);
    EXPECT_EQ(txn.size(), 6u);
    EXPECT_TRUE(std::isfinite(txn.require_log_proposal_ratio()));

    const auto nspecial = frame.span(kAtom_nspecial);
    EXPECT_EQ(std::accumulate(frame.span(kAtom_num_bond).data(),
                              frame.span(kAtom_num_bond).data() + frame.nlocal(), 0), 4);
    int endpoints = 0;
    for (int i = 0; i < frame.nlocal(); ++i)
        endpoints += (nspecial(i, 0) == 1);
    EXPECT_EQ(endpoints, 4);

    txn.rollback(frame);
    for (int i = 0; i < frame.nlocal(); ++i) {
        EXPECT_EQ(frame.span(kAtom_x)[i], before.span(kAtom_x)[i]);
        EXPECT_EQ(frame.span(kAtom_num_bond)[i], before.span(kAtom_num_bond)[i]);
        EXPECT_EQ(frame.span(kAtom_bond_atom)(i, 0), before.span(kAtom_bond_atom)(i, 0));
        for (int col = 0; col < 3; ++col)
            EXPECT_EQ(frame.span(kAtom_nspecial)(i, col), before.span(kAtom_nspecial)(i, col));
    }
}

TEST(PolymerMoveUnit, ReptationAccountsForBothEndpointDensities) {
    cpu::Frame frame(4);
    initialize_polymer(frame, {{6, 6, 10}, {7, 6, 10}, {8, 6, 10}, {9, 6, 10}});
    cpu::Context context;
    std::mt19937_64 rng(191);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    constexpr double sigma = 0.7;
    cpu::mc::ReptationMove move({{11, 12, 13, 14}}, sigma);
    cpu::LaunchContext launch{context, frame, 0, 0};
    move.launch(launch);
    ASSERT_EQ(txn.size(), 4u);
    double sum_squared = 0;
    for (int i = 1; i < 4; ++i)
        sum_squared += (polymer_point(frame, i) - polymer_point(frame, i - 1)).norm_squared();
    EXPECT_NEAR(txn.require_log_proposal_ratio(),
                (sum_squared - 3.0) / (2 * sigma * sigma), 1e-5);
}

TEST(PolymerMoveUnit, AtomIdentityExchangeIsAnInvolutionAtFixedComposition) {
    cpu::Frame frame(2);
    frame.span(kAtom_tag)[0] = 1;
    frame.span(kAtom_tag)[1] = 2;
    frame.span(kAtom_mass)[0] = 1;
    frame.span(kAtom_mass)[1] = 2;
    frame.span(kAtom_x)[0] = 3;
    frame.span(kAtom_x)[1] = 7;
    frame.span(kAtom_image)(0, 0) = 1;
    cpu::Context context;
    std::mt19937_64 rng(33);
    cpu::mc::Transaction txn;
    context.mc.rng = &rng;
    context.mc.proposal = &txn;
    cpu::mc::AtomIdentityExchangeMove move({{1, 2}});
    cpu::LaunchContext launch{context, frame, 0, 0};
    move.launch(launch);
    EXPECT_EQ(frame.span(kAtom_x)[0], 7);
    EXPECT_EQ(frame.span(kAtom_image)(1, 0), 1);
    EXPECT_EQ(frame.span(kAtom_mass)[0], 1);
    EXPECT_EQ(frame.span(kAtom_mass)[1], 2);
    move.launch(launch);
    EXPECT_EQ(frame.span(kAtom_x)[0], 3);
    EXPECT_EQ(frame.span(kAtom_image)(0, 0), 1);
}
