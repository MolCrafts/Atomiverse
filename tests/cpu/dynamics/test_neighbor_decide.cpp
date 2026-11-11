/// @file tests/cpu/dynamics/test_neighbor_decide.cpp
/// @brief `NeighborList::decide_reneighbor` is LAMMPS `Neighbor::decide`.

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"

using namespace ATV;

namespace {

cpu::Frame two_atoms() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 1.0;
    f.span(kAtom_z)[0] = 1.0;
    f.span(kAtom_x)[1] = 1.5;
    f.span(kAtom_y)[1] = 1.0;
    f.span(kAtom_z)[1] = 1.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_mass)[1] = 1.0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

} // namespace

TEST(TestNeighborDecide, DelayTenEveryOneCheckNoRebuildsOnTenthDecide) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(/*cutoff=*/2.0, /*skin=*/0.4, /*every=*/1,
                               /*delay=*/10, /*check=*/false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    ASSERT_EQ(nl.n_rebuilds(), 1);

    for (int s = 1; s <= 9; ++s) {
        const bool nflag = nl.decide_reneighbor(ctx, frame, true);
        EXPECT_FALSE(nflag) << "step " << s;
        ctx.md.reneighbor = nflag;
        if (nflag)
            nl.launch({ctx, frame, 0, s});
        EXPECT_EQ(nl.n_rebuilds(), 1) << "step " << s;
    }
    const bool tenth = nl.decide_reneighbor(ctx, frame, true);
    EXPECT_TRUE(tenth);
    ctx.md.reneighbor = tenth;
    nl.launch({ctx, frame, 0, 10});
    EXPECT_EQ(nl.n_rebuilds(), 2);
}

// LAMMPS e8c4c278a8 `neigh_modify` defaults (src/neighbor.cpp:122): every 1
// delay 0 check yes. A deferred check is not safe for MD: a bead that moves
// past half the skin inside the delay window meets partners the list lacks.
TEST(TestNeighborDecide, DefaultChecksDisplacementOnTheFirstStep) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(/*cutoff=*/2.0, /*skin=*/0.4);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
    frame.span(kAtom_x)[0] += 0.25; // > skin / 2
    EXPECT_TRUE(nl.decide_reneighbor(ctx, frame, true));
}

TEST(TestNeighborDecide, PartitionDirtyForcesReneighborDuringDelay) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 10, false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    cpu::md::mpi::DomainDecomposition dd;
#ifdef ATV_HAS_MPI
    // A communicator is part of the spatial-DD contract (no collective here).
    dd.cart_comm = MPI_COMM_SELF;
#endif
    ctx.md.dd = &dd;
    ctx.mpi_size = 2;
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
    dd.partition_dirty = true;
    EXPECT_TRUE(nl.decide_reneighbor(ctx, frame, true));
}

TEST(TestNeighborDecide,
     LifecycleBuildDoesNotEnumerateAndLaunchAlwaysRebuilds) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 10, false);
    nl.build({ctx, frame});
    EXPECT_EQ(nl.n_rebuilds(), 0);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 0);
    EXPECT_TRUE(nl.decide_reneighbor(ctx, frame, true));
    nl.launch({ctx, frame, 0, 0});
    EXPECT_EQ(nl.n_rebuilds(), 1);
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
    nl.launch({ctx, frame, 0, 1});
    EXPECT_EQ(nl.n_rebuilds(), 2);
}

TEST(TestNeighborDecide, DisplacementChecksOwnersAndDoesNotFoldWholeBoxMotion) {
    auto frame = two_atoms();
    frame.set_nlocal(1);
    frame.set_nghost(1);
    cpu::Context context;
    cpu::md::NeighborList<> neighbors(2.0, 0.4, 1, 0, true);
    neighbors.build({context, frame});
    neighbors.launch({context, frame, 0, 0});
    frame.span(kAtom_x)[1] += 0.5;
    EXPECT_FALSE(neighbors.decide_reneighbor(context, frame, true));
    // Integrators preserve continuous coordinates between rebuilds. Folding
    // this displacement by PBC would hide a real stale neighbor list.
    frame.span(kAtom_x)[0] += 10.0;
    EXPECT_TRUE(neighbors.decide_reneighbor(context, frame, true));
}

TEST(TestNeighborDecide, HalfAndFullViewsCoexistAndRefreshDuringDelay) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 10, false);
    nl.set_atom_sort(1000, 0.0, false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    ASSERT_NE(ctx.md.pair_list, nullptr);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 1);
    nl.configure_lists(true, true);
    EXPECT_TRUE(nl.decide_reneighbor(ctx, frame, true));
    nl.launch({ctx, frame, 0, 0});
    ASSERT_NE(ctx.md.pair_list, nullptr);
    ASSERT_NE(ctx.md.full_pair_list, nullptr);
    EXPECT_NE(ctx.md.pair_list, ctx.md.full_pair_list);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 1);
    EXPECT_EQ(ctx.md.full_pair_list->total_pairs, 2);
    EXPECT_EQ(ctx.md.full_pair_list->n_rows, 2);
    nl.configure_lists(true, false);
    nl.launch({ctx, frame, 0, 1});
    EXPECT_EQ(ctx.md.full_pair_list, nullptr);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 1);
    nl.destroy({ctx, frame});
    EXPECT_EQ(ctx.md.pair_list, nullptr);
    EXPECT_EQ(ctx.md.atom_sort, nullptr);
}

#ifdef ATV_HAS_MPI
TEST(TestNeighborDecide, FullListGhostRowsAreIndependentAndCanBeRemoved) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::mpi::DomainDecomposition dd;
    frame.set_nlocal(1);
    frame.set_nghost(1);
    dd.dims[0] = 2;
    dd.subhi[0] = 0.5;
    dd.subhi[1] = dd.subhi[2] = 1.0;
    // A distribution predicate only: this unit does not call communication.
    dd.cart_comm = MPI_COMM_WORLD;
    ctx.mpi_size = 2;
    ctx.md.dd = &dd;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 0, false);
    nl.configure_lists(false, true);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    for (bool ghost_rows : {false, true, false}) {
        nl.configure_lists(false, true, ghost_rows);
        ctx.md.reneighbor = true;
        nl.launch({ctx, frame, 0, 0});
        ASSERT_NE(ctx.md.full_pair_list, nullptr);
        const auto& list = *ctx.md.full_pair_list;
        EXPECT_EQ(list.inum, 1);
        EXPECT_EQ(list.n_rows, ghost_rows ? 2 : 1);
        EXPECT_EQ(list.total_pairs, ghost_rows ? 2 : 1);
        ASSERT_EQ(list.numneigh[0], 1);
        EXPECT_EQ(list.neighbors[list.firstneigh[0]], 1);
    }
    EXPECT_THROW(nl.configure_lists(true, false, true), std::invalid_argument);
    nl.destroy({ctx, frame});
}
#endif

TEST(TestNeighborDecide, RuntimeReplacesConsumerRequirementsOnLiveRemoval) {
    class Consumer : public cpu::Component, public cpu::NeighborConsumer {
      public:
        explicit Consumer(cpu::NeighborListMode mode) : mode_(mode) {}
        std::string name() const override { return "neighbor_consumer"; }
        void launch(const cpu::LaunchContext&) override {}
        AccReal neighbor_cutoff() const override { return 2.0; }
        cpu::NeighborListMode neighbor_list_mode() const override {
            return mode_;
        }
        const cpu::NeighborConsumer* as_neighbor_consumer() const override {
            return this;
        }

      private:
        cpu::NeighborListMode mode_;
    } half(cpu::NeighborListMode::Half), full(cpu::NeighborListMode::Full);
    auto frame = two_atoms();
    cpu::md::NeighborList<> nl(2.0, 0.4);
    PlanBuilder<cpu::Component> builder;
    builder.add({0, 0, &nl});
    builder.add({0, 1, &half});
    const auto initial = builder.build(1);
    builder.add({0, 2, &full});
    const auto combined = builder.build(1);
    Plan<cpu::Component> addition;
    addition.lifecycle_only = {&full};
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(initial);
    runtime.set_frame(frame);
    runtime.build_components();
    auto& ctx = runtime.context();
    nl.launch({ctx, frame, 0, 0});
    EXPECT_EQ(ctx.md.full_pair_list, nullptr);
    runtime.build_plan(addition);
    runtime.load(combined);
    nl.launch({ctx, frame, 0, 0});
    ASSERT_NE(ctx.md.full_pair_list, nullptr);
    ASSERT_NE(ctx.md.pair_list, nullptr);
    EXPECT_EQ(ctx.md.full_pair_list->total_pairs, 2);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 1);
    runtime.destroy_plan(addition);
    runtime.load(initial);
    EXPECT_EQ(ctx.md.full_pair_list, nullptr);
    EXPECT_EQ(ctx.md.pair_list->total_pairs, 1);
    runtime.destroy_components();
}

TEST(TestNeighborDecide, CellScalingBypassesDelayAndEvery) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 10, 20, true);
    nl.set_atom_sort(1000, 0.0, false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
    // 5% of a 10 Å edge is 0.5 Å, past the 0.4 Å skin. A 0.1% Berendsen
    // step is not: see SubSkinIsotropicScaleStaysOnCadence.
    frame.box.scale_isotropic(1.05);
    EXPECT_TRUE(nl.decide_reneighbor(ctx, frame, true));
    ctx.md.reneighbor = true;
    nl.launch({ctx, frame, 0, 0});
    EXPECT_FALSE(nl.strategy().grid_stale(frame));
    EXPECT_EQ(nl.n_rebuilds(), 2);
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
}

TEST(TestNeighborDecide, SubSkinIsotropicScaleStaysOnCadence) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 10, true);
    nl.set_atom_sort(1000, 0.0, false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    frame.box.scale_isotropic(1.001);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_x)[i] *= 1.001;
        frame.span(kAtom_y)[i] *= 1.001;
        frame.span(kAtom_z)[i] *= 1.001;
    }
    EXPECT_TRUE(nl.strategy().grid_stale(frame));
    EXPECT_LT(nl.strategy().lattice_drift(frame), nl.strategy().skin());
    EXPECT_FALSE(nl.decide_reneighbor(ctx, frame, true));
    EXPECT_EQ(nl.n_rebuilds(), 1);
}

namespace {

/// Rebuild once, change the cell, move owner 0 along x; decide.
bool decide_after(const Mat3& cell, double move) {
    auto frame = two_atoms();
    cpu::Context ctx;
    cpu::md::NeighborList<> nl(2.0, 0.4, 1, 0, true);
    nl.set_atom_sort(1000, 0.0, false);
    nl.build({ctx, frame});
    nl.launch({ctx, frame, 0, 0});
    frame.box = cpu::Box(cell);
    frame.span(kAtom_x)[0] += move;
    return nl.decide_reneighbor(ctx, frame, true);
}

} // namespace

TEST(TestNeighborDecide, CellStrainShrinksTheDisplacementTrigger) {
    // LAMMPS Neighbor::check_distance boxcheck: hi corner moves
    // |(0.1, 0.1, 0.1)| = 0.173, trigger 0.5 (0.4 - 0.173) = 0.113.
    const Mat3 strained = Mat3::diagonal(10.1, 10.1, 10.1);
    EXPECT_TRUE(decide_after(strained, 0.15));
    EXPECT_FALSE(decide_after(strained, 0.05));
    EXPECT_FALSE(decide_after(Mat3::diagonal(10, 10, 10), 0.15));
}

TEST(TestNeighborDecide, TiltChangeShrinksTheTriggerByTwoCorners) {
    // An xy change of 0.15 moves four corners by 0.15: trigger
    // 0.5 (0.4 - 0.3) = 0.05.
    Mat3 tilted = Mat3::diagonal(10, 10, 10);
    tilted(0, 1) = 0.15;
    EXPECT_TRUE(decide_after(tilted, 0.08));
    EXPECT_FALSE(decide_after(tilted, 0.04));
}

TEST(TestNeighborDecide, CellChangeThatConsumesTheSkinRebuildsWithoutMotion) {
    // Every lattice vector moves 0.3 < skin (no lattice-drift rebuild), but
    // the hi corner moves 0.52 > skin.
    EXPECT_TRUE(decide_after(Mat3::diagonal(10.3, 10.3, 10.3), 0.0));
}
