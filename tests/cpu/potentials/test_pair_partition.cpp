/// @file tests/cpu/potentials/test_pair_partition.cpp
/// @brief Unit: `md::PairRowPartition` (team size, balanced rows, touched-row
///        bound) and the threaded PairLJCut / PairCoulCut paths it drives.
///
/// Mirrors `include/atomiverse/cpu/potentials/pair_partition.h` and the
/// parallel regions of `pair_lj_cut.cpp` / `pair_coul_cut.cpp`. A molecule
/// with an all-pairs half list is the case the partition exists for: row i
/// holds N-1-i partners, so a static split over rows gives the first thread
/// most of the work, and ghost rows the list never names must not be swept.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_partition.h"

using namespace ATV;

namespace {

/// All-pairs half list over rows [0, n): row i holds i+1 .. n-1.
cpu::md::PairList all_pairs(int n) {
    cpu::md::PairList p;
    p.inum = n;
    p.n_rows = n;
    p.ilist.resize(n);
    p.numneigh.resize(n);
    p.firstneigh.resize(n);
    for (int i = 0; i < n; ++i) {
        p.ilist[i] = i;
        p.firstneigh[i] = static_cast<int>(p.neighbors.size());
        for (int j = i + 1; j < n; ++j)
            p.neighbors.push_back(j);
        p.numneigh[i] = n - 1 - i;
    }
    p.total_pairs = static_cast<int>(p.neighbors.size());
    p.version = 1;
    return p;
}

std::int64_t pairs_in(const cpu::md::PairList& p, int begin, int end) {
    std::int64_t s = 0;
    for (int ii = begin; ii < end; ++ii)
        s += p.numneigh[p.ilist[ii]];
    return s;
}

} // namespace

TEST(PairRowPartition, RangesTileTheRowsAndBalanceThePairs) {
    const auto p = all_pairs(700);
    cpu::md::PairRowPartition part;
    part.rebuild(p.inum, p.ilist.data(), p.numneigh.data(), p.firstneigh.data(),
                 p.neighbors.data());
    ASSERT_EQ(part.total(), p.total_pairs);
    for (int team : {1, 2, 3, 4, 7}) {
        int next = 0;
        const double share = static_cast<double>(part.total()) / team;
        for (int t = 0; t < team; ++t) {
            const auto [b, e] = part.range(t, team);
            EXPECT_EQ(b, next) << "rows must tile without gap or overlap";
            EXPECT_LE(b, e);
            next = e;
            // Each share is within one row (the longest, N-1 pairs) of equal.
            EXPECT_NEAR(static_cast<double>(pairs_in(p, b, e)), share,
                        static_cast<double>(p.inum));
        }
        EXPECT_EQ(next, p.inum);
    }
}

TEST(PairRowPartition, TeamFollowsThePairCount) {
    cpu::md::PairRowPartition part;
    const auto small = all_pairs(40); // 780 pairs
    part.rebuild(small.inum, small.ilist.data(), small.numneigh.data(),
                 small.firstneigh.data(), small.neighbors.data());
    EXPECT_EQ(part.team(10), 1) << "a small list runs on one thread";

    const auto big = all_pairs(700); // 244,650 pairs
    part.rebuild(big.inum, big.ilist.data(), big.numneigh.data(),
                 big.firstneigh.data(), big.neighbors.data());
    const int by_work = static_cast<int>(
        part.total() / cpu::md::PairRowPartition::kMinPairsPerThread);
    EXPECT_EQ(part.team(64), by_work);
    EXPECT_EQ(part.team(2), 2) << "never more than the threads available";
    EXPECT_EQ(part.team(0), 1);
}

TEST(PairRowPartition, TouchBoundIsTheHighestNamedRow) {
    // Rows 0..2 name partners up to 5; rows past that (ghosts) are untouched.
    cpu::md::PairList p;
    p.inum = 3;
    p.ilist = {0, 1, 2};
    p.numneigh = {2, 1, 0, 0, 0, 0, 0, 0};
    p.firstneigh = {0, 2, 3, 3, 3, 3, 3, 3};
    p.neighbors = {1, 5, 2};
    p.total_pairs = 3;
    cpu::md::PairRowPartition part;
    part.rebuild(p.inum, p.ilist.data(), p.numneigh.data(), p.firstneigh.data(),
                 p.neighbors.data());
    EXPECT_EQ(part.n_touch(), 6);
}

#ifdef _OPENMP
namespace {

/// A compact random-ish molecule: N atoms on a jittered lattice, two LJ types,
/// alternating charges, bonded in a chain so specials exist. Extra ghost
/// capacity rows that no pair names sit past N.
cpu::Frame molecule(int n, int ghosts) {
    cpu::Frame f(n);
    std::uint64_t s = 12345;
    auto rnd = [&s] {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
    };
    const int side = static_cast<int>(std::ceil(std::cbrt(n)));
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = 1.6 * (i % side) + 0.3 * rnd();
        f.span(kAtom_y)[i] = 1.6 * ((i / side) % side) + 0.3 * rnd();
        f.span(kAtom_z)[i] = 1.6 * (i / (side * side)) + 0.3 * rnd();
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_type)[i] = i % 2;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_charge)[i] = (i % 2 ? -0.3 : 0.3) + 0.05 * rnd();
    }
    f.box = cpu::Box::cubic(200.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int i = 0; i + 1 < n; ++i)
        f.add_bond(i, i + 1, 0);
    f.build_special_list();
    f.grow(n + ghosts);
    f.set_nghost(ghosts);
    return f;
}

struct Result {
    double energy;
    std::vector<double> fx, fy, fz;
};

template <class Make>
Result run_kernel(Make make, int threads, cpu::Frame f,
                  const cpu::md::PairList& pairs) {
    const int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    // Threaded kernels scatter into the Runtime's chunk buffers; publish one
    // as a Runtime would, and flush it as the end of a slot walk would.
    cpu::md::ChunkForces arena;
    ctx.md.chunk_forces = &arena;
    auto pot = make();
    pot.build({ctx, f});
    f.meta.get(kEnergy) = 0;
    const int rows = f.n_total();
    for (int i = 0; i < rows; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    pot.launch({ctx, f, 0, 0, true, true});
    arena.flush();
    pot.destroy({ctx, f});
    omp_set_num_threads(saved);
    Result r{f.meta.get(kEnergy), {}, {}, {}};
    for (int i = 0; i < rows; ++i) {
        r.fx.push_back(f.span(kAtom_fx)[i]);
        r.fy.push_back(f.span(kAtom_fy)[i]);
        r.fz.push_back(f.span(kAtom_fz)[i]);
    }
    return r;
}

void expect_close(const Result& a, const Result& b) {
    EXPECT_NEAR(a.energy, b.energy, 1e-10 * std::abs(b.energy));
    ASSERT_EQ(a.fx.size(), b.fx.size());
    for (std::size_t i = 0; i < a.fx.size(); ++i) {
        EXPECT_NEAR(a.fx[i], b.fx[i], 1e-10);
        EXPECT_NEAR(a.fy[i], b.fy[i], 1e-10);
        EXPECT_NEAR(a.fz[i], b.fz[i], 1e-10);
    }
}

constexpr int kAtoms = 560;   // 156,520 all-pairs: a team of 3
constexpr int kGhosts = 2000; // rows no pair names

} // namespace

TEST(PairRowPartition, ThreadedLJTableWithSpecialsMatchesOneThread) {
    const auto pairs = all_pairs(kAtoms);
    auto make = [] {
        return cpu::PairLJCut({{0, 0, 0.004, 3.4}, {1, 1, 0.007, 3.0}}, 60.0)
            .set_special_bonds(0.0, 0.0, 0.5);
    };
    const auto one = run_kernel(make, 1, molecule(kAtoms, kGhosts), pairs);
    const auto many = run_kernel(make, 4, molecule(kAtoms, kGhosts), pairs);
    expect_close(many, one);
    // And the threaded run is reproducible bit for bit.
    const auto again = run_kernel(make, 4, molecule(kAtoms, kGhosts), pairs);
    EXPECT_EQ(again.energy, many.energy);
    EXPECT_EQ(again.fx, many.fx);
}

TEST(PairRowPartition, ThreadedCoulombWithSpecialsMatchesOneThread) {
    const auto pairs = all_pairs(kAtoms);
    for (bool specials : {false, true}) {
        auto make = [specials] {
            cpu::PairCoulCut c(60.0);
            if (specials)
                c.set_special_bonds(0.0, 0.0, 0.8333);
            return c;
        };
        const auto one = run_kernel(make, 1, molecule(kAtoms, kGhosts), pairs);
        const auto many = run_kernel(make, 4, molecule(kAtoms, kGhosts), pairs);
        expect_close(many, one);
    }
}
#endif
