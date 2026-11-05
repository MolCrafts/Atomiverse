/// @file tests/cpu/potentials/test_chunk_forces.cpp
/// @brief The Runtime's shared chunk-owned force accumulators and the
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

/// One launch of a two-chunk kernel: chunk c adds `base + c` to fx[0].
void add_in_chunks(cpu::md::ChunkForces& arena, cpu::Frame& f, double base) {
    arena.bind(f, /*forces_requested=*/true);
    arena.open(2, f.n_total());
#pragma omp parallel for num_threads(2) schedule(dynamic, 1)
    for (int c = 0; c < 2; ++c) {
        const cpu::md::ForceRows rows = arena.slot(c);
        rows.fx[0] += base + c;
        rows.fy[1] += 1.0;
    }
}

} // namespace

TEST(ChunkForces, LaunchesShareOneReductionPerSlot) {
    FixedTeams fixed;
    cpu::Frame f(2);
    f.span(kAtom_fx)[0] = 1000.0;
    cpu::md::ChunkForces arena;
    arena.reserve(2, f.capacity());

    // Two kernels of one slot: nothing reaches the frame until the flush.
    add_in_chunks(arena, f, 10.0); // 10 + 11
    add_in_chunks(arena, f, 20.0); // 20 + 21
    EXPECT_EQ(f.span(kAtom_fx)[0], 1000.0);
    EXPECT_TRUE(arena.pending());
    arena.flush();
    EXPECT_FALSE(arena.pending());
    EXPECT_EQ(f.span(kAtom_fx)[0], 1000.0 + 10 + 11 + 20 + 21);
    EXPECT_EQ(f.span(kAtom_fy)[1], 4.0);

    // The next slot starts from zeroed buffers; a second flush adds nothing.
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 1062.0);
    add_in_chunks(arena, f, 1.0);
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 1062.0 + 1 + 2);
}

TEST(ChunkForces, ResultFollowsChunksNotTheThreadThatRanThem) {
    // Values whose sum depends on the order they are added in: chunk order
    // fixes it, whichever order the chunks executed in.
    const double v[3] = {1.0e16, 1.0, -1.0e16};
    auto run = [&](bool reversed) {
        cpu::Frame f(1);
        cpu::md::ChunkForces arena;
        arena.bind(f, true);
        arena.open(3, 1);
        for (int n = 0; n < 3; ++n) {
            const int c = reversed ? 2 - n : n;
            arena.slot(c).fx[0] += v[c];
            arena.sums(c)[0] = v[c];
        }
        const double energy = arena.total(0);
        arena.flush();
        return std::pair{f.span(kAtom_fx)[0], energy};
    };
    const auto forward = run(false);
    const auto backward = run(true);
    EXPECT_EQ(forward.first, backward.first);
    EXPECT_EQ(forward.second, backward.second);
    // Chunk order: (1e16 + 1) - 1e16, not 1e16 - 1e16 + 1.
    EXPECT_EQ(forward.second, (v[0] + v[1]) + v[2]);
}

TEST(ChunkForces, LaterLaunchWideningTheRowsZeroesTheNewRows) {
    cpu::Frame f(4);
    cpu::md::ChunkForces arena;
    arena.bind(f, true);
    arena.open(2, 2); // first kernel touches rows [0, 2)
    arena.slot(1).fx[1] += 1.0;
    arena.bind(f, true);
    arena.open(2, 4); // second kernel needs every row
    const cpu::md::ForceRows rows = arena.slot(1);
    EXPECT_EQ(rows.fx[3], 0.0);
    rows.fx[3] += 2.0;
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[1], 1.0);
    EXPECT_EQ(f.span(kAtom_fx)[3], 2.0);
}

TEST(ChunkForces, TeamOfOneWritesTheFrameDirectly) {
    cpu::Frame f(2);
    cpu::md::ChunkForces arena;
    cpu::Context ctx;
    ctx.md.chunk_forces = &arena;
    const cpu::LaunchContext lc{ctx, f, 0, 0};
    cpu::md::ForcePass forces(lc);
    forces.open(1);
    EXPECT_TRUE(forces.direct());
    forces.rows(0).fx[0] += 5.0;
    forces.store(0, 3.0, cpu::md::Voigt6{});
    EXPECT_EQ(f.span(kAtom_fx)[0], 5.0);
    EXPECT_EQ(forces.energy(), 3.0);
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 5.0);
}

TEST(ChunkForces, EnergyOnlyPassIsDiscarded) {
    cpu::Frame f(2);
    cpu::md::ChunkForces arena;
    arena.bind(f, /*forces_requested=*/false);
    arena.open(2, f.n_total());
    arena.slot(0).fx[0] += 7.0;
    arena.slot(1).fy[1] += 7.0;
    arena.flush();
    EXPECT_EQ(f.span(kAtom_fx)[0], 0.0);
    EXPECT_EQ(f.span(kAtom_fy)[1], 0.0);
}

TEST(ChunkForces, ForcePassWithoutRuntimeRunsSerially) {
    cpu::Frame f(2);
    cpu::Context ctx; // no Runtime: chunk_forces is null
    const cpu::LaunchContext lc{ctx, f, 0, 0};
    cpu::md::ForcePass forces(lc);
    EXPECT_EQ(forces.team(), 1);
    forces.open(4, 4, f.n_total()); // a team asks for nothing without one
    EXPECT_TRUE(forces.direct());
    EXPECT_EQ(forces.chunks(), 1);
    forces.rows(0).fz[1] += 3.0;
    EXPECT_EQ(f.span(kAtom_fz)[1], 3.0);
}

TEST(ChunkForces, ForEachChunkTilesTheTermsAndSumsInChunkOrder) {
    FixedTeams fixed;
    cpu::Frame f(1);
    cpu::md::ChunkForces arena;
    cpu::Context ctx;
    ctx.md.chunk_forces = &arena;
    const cpu::LaunchContext lc{ctx, f, 0, 0, true, true};
    const int n = 5 * cpu::md::ForcePass::kMinTermsPerThread + 3;
    auto run = [&] {
        f.span(kAtom_fx)[0] = 0.0;
        cpu::md::ForcePass forces(lc);
        forces.for_each_chunk(
            n, [&](int c, int begin, int end, cpu::md::ForceRows rows) {
                double e = 0.0;
                for (int t = begin; t < end; ++t) {
                    e += 1.0 / (t + 1.0);
                    rows.fx[0] += 1.0 / (t + 1.0);
                }
                forces.store(c, e, cpu::md::Voigt6{});
            });
        const double e = forces.energy();
        arena.flush();
        return std::pair{e, f.span(kAtom_fx)[0]};
    };
    const auto first = run();
    for (int rep = 0; rep < 8; ++rep) {
        const auto again = run();
        EXPECT_EQ(again.first, first.first);
        EXPECT_EQ(again.second, first.second);
    }
    double serial = 0.0;
    for (int t = 0; t < n; ++t)
        serial += 1.0 / (t + 1.0);
    EXPECT_NEAR(first.first, serial, 1e-12);
    EXPECT_NEAR(first.second, serial, 1e-12);
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
