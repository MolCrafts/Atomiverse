/// @file tests/cpu/mc/test_gcmc_npt_movers.cpp
/// @brief Unit tests for the mc-gcmc-npt Movers: `VolumeMove` (isotropic
///        rescale + full patch), `InsertDeleteMove` (variable-N proposals via
///        the lifted comm/ primitives), and `MoveSelector` (one move type per
///        sweep, drawn at the configured probabilities — ac-006).

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/insert_delete_move.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/transaction.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/montecarlo/volume_move.h"

using namespace ATV;
using Verb = cpu::mc::Transaction::Verb;

namespace {

/// @brief Owns the context plumbing a Mover expects from an MC preset.
struct MoverHarness {
    cpu::Frame frame;
    cpu::Context ctx{};
    cpu::mc::Transaction txn{};
    std::mt19937_64 rng{7u};

    explicit MoverHarness(int n) : frame(n) {
        for (int i = 0; i < n; ++i) {
            frame.span(ATV::kAtom_x)[i] = 1.0 + i;
            frame.span(ATV::kAtom_y)[i] = 2.0 + i;
            frame.span(ATV::kAtom_z)[i] = 3.0 + i;
            frame.span(ATV::kAtom_mass)[i] = 39.948;
            frame.span(ATV::kAtom_type)[i] = 1;
        }
        frame.box.set_cell(Mat3::diagonal(10.0, 10.0, 10.0));
        ctx.mc.proposal = &txn;
        ctx.mc.rng = &rng;
    }

    cpu::LaunchContext launch(int step = 0) {
        return cpu::LaunchContext{ctx, frame, /*stage=*/0, step};
    }
};

} // namespace

// ── VolumeMove ──────────────────────────────────────────────────────────────

TEST(VolumeMove, RejectsNonPositiveDvMax) {
    EXPECT_THROW((cpu::mc::VolumeMove{0.0}), std::invalid_argument);
    EXPECT_THROW((cpu::mc::VolumeMove{-1.0}), std::invalid_argument);
}

TEST(VolumeMove, RescalesBoxAndEveryCoordinateIsotropically) {
    MoverHarness h{3};
    const cpu::Frame before = h.frame;
    cpu::mc::VolumeMove mv{100.0};
    mv.launch(h.launch());

    ASSERT_EQ(h.txn.verb, Verb::Box);
    ASSERT_EQ(h.txn.size(), 3u); // every row snapshotted
    const double s =
        h.frame.box.length(0) / before.box.length(0); // isotropic factor
    EXPECT_NEAR(h.frame.box.length(1), before.box.length(1) * s, 1e-12);
    EXPECT_NEAR(h.frame.box.length(2), before.box.length(2) * s, 1e-12);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(h.frame.span(ATV::kAtom_x)[i],
                  static_cast<StorageReal>(before.span(ATV::kAtom_x)[i] * s));
        EXPECT_EQ(h.frame.span(ATV::kAtom_y)[i],
                  static_cast<StorageReal>(before.span(ATV::kAtom_y)[i] * s));
        EXPECT_EQ(h.frame.span(ATV::kAtom_z)[i],
                  static_cast<StorageReal>(before.span(ATV::kAtom_z)[i] * s));
    }
    // ΔV bounded by δV.
    const double v0 = before.box.volume();
    EXPECT_LE(std::abs(h.frame.box.volume() - v0), 100.0 + 1e-9);

    // And a rejection restores everything bitwise.
    h.txn.rollback(h.frame);
    EXPECT_EQ(h.frame.span(ATV::kAtom_x)[1], before.span(ATV::kAtom_x)[1]);
    EXPECT_EQ(h.frame.box.cell()(2, 2), before.box.cell()(2, 2));
}

TEST(VolumeMove, ClampedDrawProposesNothing) {
    MoverHarness h{2};
    h.frame.box.set_cell(Mat3::diagonal(1.0, 1.0, 1.0)); // V = 1 Å³
    cpu::mc::VolumeMove mv{1e6}; // most draws land below V' <= 0
    int none_seen = 0;
    for (int k = 0; k < 50; ++k) {
        mv.launch(h.launch());
        if (h.txn.verb == Verb::None) {
            ++none_seen;
        } else {
            h.txn.rollback(h.frame); // keep V at 1 Å³ for the next draw
        }
    }
    EXPECT_GT(none_seen, 0); // ~50% of draws are negative-volume clamps
}

TEST(VolumeMove, BootstrapPassIsNoOp) {
    MoverHarness h{2};
    const cpu::Frame before = h.frame;
    cpu::mc::VolumeMove mv{10.0};
    mv.launch(h.launch(/*step=*/-1));
    EXPECT_EQ(h.txn.size(), 0u);
    EXPECT_EQ(h.frame.box.cell()(0, 0), before.box.cell()(0, 0));
}

// ── InsertDeleteMove ────────────────────────────────────────────────────────

TEST(InsertDeleteMove, RejectsBadConstruction) {
    EXPECT_THROW((cpu::mc::InsertDeleteMove{0, 1.0}), std::invalid_argument);
    EXPECT_THROW((cpu::mc::InsertDeleteMove{1, 0.0}), std::invalid_argument);
}

TEST(InsertDeleteMove, InsertionAppendsAZeroedRowInsideTheCell) {
    MoverHarness h{3};
    cpu::mc::InsertDeleteMove mv{2, 39.948};
    // Drive until an insertion happens (coin is 50/50).
    while (true) {
        const int n_before = h.frame.nlocal();
        mv.launch(h.launch());
        if (h.txn.verb == Verb::Insert) {
            const int slot = n_before;
            ASSERT_EQ(h.frame.nlocal(), n_before + 1);
            EXPECT_GE(h.frame.span(kAtom_x)[slot], 0.0);
            EXPECT_LT(h.frame.span(kAtom_x)[slot], 10.0);
            EXPECT_GE(h.frame.span(kAtom_y)[slot], 0.0);
            EXPECT_LT(h.frame.span(kAtom_y)[slot], 10.0);
            EXPECT_EQ(h.frame.span(kAtom_vx)[slot], 0.0);
            EXPECT_EQ(h.frame.span(kAtom_fz)[slot], 0.0);
            EXPECT_EQ(h.frame.span(kAtom_charge)[slot], 0.0);
            EXPECT_EQ(h.frame.span(kAtom_num_bond)[slot], 0);
            EXPECT_EQ(h.frame.span(kAtom_mass)[slot], StorageReal(39.948));
            EXPECT_EQ(h.frame.span(kAtom_type)[slot], 2);
            EXPECT_EQ(h.frame.span(kAtom_tag)[slot], 3);
            break;
        }
        h.txn.rollback(h.frame); // undo the deletion and redraw
    }
}

TEST(InsertDeleteMove, DeletionCompactsAndPatchResurrects) {
    MoverHarness h{4};
    const cpu::Frame before = h.frame;
    cpu::mc::InsertDeleteMove mv{1, 39.948};
    while (true) {
        mv.launch(h.launch());
        if (h.txn.verb == Verb::Delete) {
            EXPECT_EQ(h.frame.nlocal(), 3);
            h.txn.rollback(h.frame);
            ASSERT_EQ(h.frame.nlocal(), 4);
            for (int i = 0; i < 4; ++i) {
                EXPECT_EQ(h.frame.span(ATV::kAtom_x)[i],
                          before.span(ATV::kAtom_x)[i])
                    << i;
                EXPECT_EQ(h.frame.span(ATV::kAtom_tag)[i],
                          before.span(ATV::kAtom_tag)[i])
                    << i;
            }
            break;
        }
        h.txn.rollback(h.frame); // undo the insertion and redraw
    }
}

TEST(InsertDeleteMove, DeletionFromEmptyFrameProposesNothing) {
    MoverHarness h{0};
    cpu::mc::InsertDeleteMove mv{1, 39.948};
    int none_seen = 0;
    for (int k = 0; k < 40; ++k) {
        mv.launch(h.launch());
        if (h.txn.verb == Verb::None)
            ++none_seen; // the N = 0 deletion branch
        else
            h.txn.rollback(h.frame); // reject the insertion, stay at N = 0
    }
    EXPECT_GT(none_seen, 0);
    EXPECT_EQ(h.frame.nlocal(), 0);
}

// ── MoveSelector (ac-006) ───────────────────────────────────────────────────

TEST(MoveSelector, RejectsBadConfiguration) {
    EXPECT_THROW(cpu::mc::MoveSelector{}.add(cpu::mc::TranslateMove{0.2}, 0.0),
                 std::invalid_argument);
    MoverHarness h{1};
    cpu::mc::MoveSelector empty;
    cpu::BuildContext bctx{h.ctx, h.frame};
    EXPECT_THROW(empty.build(bctx), std::runtime_error);
}

TEST(MoveSelector, FiresExactlyOneChildPerSweepAtConfiguredProbabilities) {
    MoverHarness h{16};
    cpu::mc::MoveSelector sel = cpu::mc::MoveSelector{}
                                    .add(cpu::mc::TranslateMove{0.05}, 0.75)
                                    .add(cpu::mc::VolumeMove{1.0}, 0.25);

    const int n_sweeps = 20000;
    int displace = 0, box = 0;
    for (int k = 0; k < n_sweeps; ++k) {
        sel.launch(h.launch());
        if (h.txn.verb == Verb::Displace)
            ++displace;
        else if (h.txn.verb == Verb::Box)
            ++box;
        h.txn.rollback(h.frame); // hold the state fixed across sweeps
    }
    // Exactly one child fired per sweep…
    EXPECT_EQ(displace + box, n_sweeps);
    // …at the configured 3:1 ratio (4σ band, σ = sqrt(p(1−p)/n)).
    const double p = static_cast<double>(displace) / n_sweeps;
    const double sigma = std::sqrt(0.75 * 0.25 / n_sweeps);
    EXPECT_NEAR(p, 0.75, 4.0 * sigma);
}

#include "atomiverse/cpu/montecarlo/gcmc.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

TEST(MoveSelector, TemporaryChainIsOwnedByTheDriver) {
    // Regression pin for a real dangling-pointer bug: `MoveSelector{}.add(…)`
    // used to yield an LVALUE reference to the temporary, so Driver::add took
    // its non-owning branch and kept a raw pointer into a dead object —
    // role() then read a trashed vtable and build() threw "Mover missing"
    // (release builds jumped to address 0). The ref-qualified `add` overloads
    // keep the chain an rvalue, forcing the owning branch. This test drives
    // exactly that temporary-chain path through build().
    cpu::mc::Gcmc mc{120.0, -0.1, 39.948, 1u};
    mc.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 8.5});
    mc.set_neighbor(cpu::md::NeighborList<>{8.5, 1.0, 1, 0, true});
    mc.set_mover(cpu::mc::MoveSelector{}
                     .add(cpu::mc::TranslateMove{0.2}, 0.5)
                     .add(cpu::mc::InsertDeleteMove{1, 39.948}, 0.5));
    EXPECT_NO_THROW(mc.build());
}

TEST(MoveSelector, BootstrapPassDrawsNothing) {
    MoverHarness h{4};
    cpu::mc::MoveSelector sel =
        cpu::mc::MoveSelector{}.add(cpu::mc::TranslateMove{0.1}, 1.0);
    const auto rng_before = h.rng; // no RNG consumption on bootstrap
    sel.launch(h.launch(/*step=*/-1));
    EXPECT_EQ(h.txn.size(), 0u);
    EXPECT_TRUE(rng_before == h.rng);
}
