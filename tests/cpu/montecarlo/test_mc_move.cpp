/// @file tests/cpu/mc/test_mc_move.cpp
/// @brief Fast unit tests for the MC move + driver assembly
///        (mc-sampling-02-move): TranslateMove's per-launch behaviour and the
///        MC driver's build-time validation + config-constructor guards.
///
/// The Mover is exercised
/// by calling `launch()` directly against a hand-built context; the driver is
/// exercised only through `build()` (plan compile + validation, no stepping).
/// Mock-only McBase::run tests below check its decision/observation boundary.
/// End-to-end physical sampler behaviour (bootstrap energy, accept/reject/rollback,
/// determinism, Metropolis distribution) belongs to the validation tier in
/// sub-spec mc-sampling-03-validate, not to this ms-scale suite.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/mc.h"
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

// Orchestration unit: mocks exercise McBase::run's decision/observation boundary,
// with no force field, neighbor build, physical trajectory or stochastic oracle.
TEST(McBaseUnit, RejectedTrialRestoresOutputsBeforeObservation) {
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

TEST(McBaseUnit, FailedBootstrapCanBeRetriedWithoutObservingIt) {
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

TEST(McBaseUnit,
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
