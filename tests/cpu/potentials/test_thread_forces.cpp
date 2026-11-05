/// @file tests/cpu/potentials/test_thread_forces.cpp
/// @brief The Runtime's shared per-thread force accumulators and the
///        row-identity revision bonded caches key on.
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ATV;

namespace {

/// Disables dynamic team sizing for the scope so `num_threads(2)` is honoured.
struct FixedTeams {
#ifdef _OPENMP
    int saved = omp_get_dynamic();
    FixedTeams() { omp_set_dynamic(0); }
    ~FixedTeams() { omp_set_dynamic(saved); }
#endif
};

void add_in_team(cpu::md::ThreadForces& arena, int team, double base) {
#pragma omp parallel num_threads(team)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const cpu::md::ThreadForce tf = arena.thread(tid);
        tf.fx[0] += base + tid;
        tf.fy[1] += 1.0;
    }
}

} // namespace

TEST(ThreadForces, LaunchesShareOneReductionPerSlot) {
    FixedTeams fixed;
    cpu::Frame f(2);
    f.span(kAtom_fx)[0] = 1000.0;
    cpu::md::ThreadForces arena;
    arena.reserve(2, f.capacity());

    // Two kernels of one slot: nothing reaches the frame until the flush.
    arena.bind(f, /*forces_requested=*/true);
    add_in_team(arena, 2, 10.0); // 10 + 11
    arena.bind(f, true);
    add_in_team(arena, 2, 20.0); // 20 + 21
#ifdef _OPENMP
    EXPECT_EQ(f.span(kAtom_fx)[0], 1000.0);
    EXPECT_TRUE(arena.pending());
#endif
    arena.flush();
    EXPECT_FALSE(arena.pending());
    EXPECT_EQ(f.span(kAtom_fx)[0], 1000.0 + 10 + 11 + 20 + 21);
    EXPECT_EQ(f.span(kAtom_fy)[1], 4.0);

    // The next slot starts from zeroed buffers; a second flush adds nothing.
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 1062.0);
    arena.bind(f, true);
    add_in_team(arena, 2, 1.0);
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 1062.0 + 1 + 2);
}

TEST(ThreadForces, TeamOfOneWritesTheFrameDirectly) {
    FixedTeams fixed;
    cpu::Frame f(2);
    cpu::md::ThreadForces arena;
    arena.bind(f, true);
    add_in_team(arena, 1, 5.0);
    EXPECT_EQ(f.span(kAtom_fx)[0], 5.0);
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 5.0);
}

TEST(ThreadForces, EnergyOnlyPassIsDiscarded) {
#ifndef _OPENMP
    GTEST_SKIP() << "private buffers exist only with OpenMP teams";
#else
    FixedTeams fixed;
    cpu::Frame f(2);
    cpu::md::ThreadForces arena;
    arena.reserve(2, f.capacity());
    arena.bind(f, /*forces_requested=*/false);
    add_in_team(arena, 2, 7.0);
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 0.0);
    EXPECT_EQ(f.span(kAtom_fy)[1], 0.0);
#endif
}

TEST(ThreadForces, ForcePassWithoutRuntimeRunsSerially) {
    cpu::Frame f(2);
    cpu::Context ctx; // no Runtime: thread_forces is null
    const cpu::LaunchContext lc{ctx, f, 0, 0};
    const cpu::md::ForcePass forces(lc);
    EXPECT_EQ(forces.team(), 1);
    forces.thread(0).fz[1] += 3.0;
    EXPECT_EQ(f.span(kAtom_fz)[1], 3.0);
}

TEST(FrameRowRevision, ChangesOnRowIdentityNotOnPositions) {
    cpu::Frame f(4);
    const auto r0 = f.row_revision();
    f.span(kAtom_x)[0] = 1.5; // position refresh
    EXPECT_EQ(f.row_revision(), r0);
    f.set_nghost(0);
    const auto r1 = f.row_revision();
    EXPECT_NE(r1, r0);
    f.copy_row(1, 2);
    const auto r2 = f.row_revision();
    EXPECT_NE(r2, r1);
    f.set_nlocal(3);
    EXPECT_NE(f.row_revision(), r2);
    // A different Frame assigned at the same address gets a fresh revision.
    const auto before = f.row_revision();
    f = cpu::Frame(4);
    EXPECT_NE(f.row_revision(), before);
}

TEST(BondedTupleCache, StaleExactlyWhenRowsChange) {
    cpu::Frame f(3);
    cpu::Context ctx;
    const cpu::LaunchContext lc{ctx, f, 0, 0};
    cpu::md::BondedTupleCache cache;
    EXPECT_TRUE(cache.stale(lc));
    cache.mark_built(lc);
    EXPECT_FALSE(cache.stale(lc));
    f.span(kAtom_y)[2] = 4.0; // positions move every step
    EXPECT_FALSE(cache.stale(lc));
    f.copy_row(0, 1); // sort / exchange compaction
    EXPECT_TRUE(cache.stale(lc));
    cache.mark_built(lc);
    f.set_nghost(0); // borders rebuild
    EXPECT_TRUE(cache.stale(lc));
}
