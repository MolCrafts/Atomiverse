/// @file tests/cpu/mc/test_mc_move.cpp
/// @brief Fast unit tests for the MC move + driver assembly
///        (mc-sampling-02-move): TranslateMove's per-launch behaviour and the
///        MC driver's build-time validation + config-constructor guards.
///
/// These are true unit tests — they do NOT drive `MC::run()` (a full
/// backend/plan pipeline, i.e. an integration/e2e path). The Mover is exercised
/// by calling `launch()` directly against a hand-built context; the driver is
/// exercised only through `build()` (plan compile + validation, no stepping).
/// End-to-end sampler behaviour (bootstrap energy, accept/reject/rollback,
/// determinism, Metropolis distribution) belongs to the validation tier in
/// sub-spec mc-sampling-03-validate, not to this ms-scale suite.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/mc/mc.h"
#include "atomiverse/cpu/mc/transaction.h"
#include "atomiverse/cpu/mc/translate_move.h"
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
    mc.add_potential(cpu::PairLJCut{kEps, kSigma, kCut});
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
    mc.add_potential(cpu::PairLJCut{kEps, kSigma, kCut});
    mc.set_mover(cpu::mc::TranslateMove{0.2});
    EXPECT_THROW(mc.build(), std::runtime_error);
}

TEST(MCBuildUnit, SucceedsWhenFullyAssembled) {
    cpu::mc::MC mc{300.0, 1u};
    mc.add_potential(cpu::PairLJCut{kEps, kSigma, kCut});
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
