/// @file tests/cpu/potentials/test_force_determinism.cpp
/// @brief Threaded force kernels are bit-reproducible run to run at a fixed
///        thread count (OMP 1 / 4 / 8): chunk-owned force slots and partial
///        sums, combined in chunk order (`md::ChunkForces`).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/pair_morse.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ATV;

namespace {

constexpr int kThreadCounts[] = {1, 4, 8};
constexpr int kRepeats = 4;

/// Jittered lattice of `n` atoms, spacing `a`, chained by bonds of type 0.
cpu::Frame chain(int n, double a, double box) {
    cpu::Frame f(n);
    std::uint64_t s = 977;
    auto rnd = [&s] {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
    };
    const int side = static_cast<int>(std::ceil(std::cbrt(n)));
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = a * (i % side) + 0.2 * a * rnd();
        f.span(kAtom_y)[i] = a * ((i / side) % side) + 0.2 * a * rnd();
        f.span(kAtom_z)[i] = a * (i / (side * side)) + 0.2 * a * rnd();
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i;
    }
    f.box = cpu::Box::cubic(box);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int i = 0; i + 1 < n; ++i)
        f.add_bond(i, i + 1, 0);
    return f;
}

/// All-pairs half list over rows [0, n).
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

struct Snapshot {
    double energy = 0.0;
    double stress[6] = {};
    std::vector<double> fx, fy, fz;
    bool threaded = false; ///< forces went through the chunk slots

    bool operator==(const Snapshot& o) const {
        for (int k = 0; k < 6; ++k)
            if (stress[k] != o.stress[k])
                return false;
        return energy == o.energy && fx == o.fx && fy == o.fy && fz == o.fz;
    }
};

/// One launch of `pot` with a published chunk arena, flushed as the end of
/// a Runtime slot walk would; `pot` must already be built.
template <class Pot>
Snapshot evaluate(Pot& pot, cpu::Context& ctx, cpu::md::ChunkForces& arena,
                  cpu::Frame& f) {
    const int rows = f.n_total();
    for (int i = 0; i < rows; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    f.meta.get(kEnergy) = 0;
    for (int k = 0; k < 6; ++k)
        f.meta.get(kStress)[k] = 0;
    pot.launch({ctx, f, 0, 0, true, true});
    Snapshot s;
    s.threaded = arena.pending();
    arena.flush();
    s.energy = f.meta.get(kEnergy);
    for (int k = 0; k < 6; ++k)
        s.stress[k] = f.meta.get(kStress)[k];
    s.fx.assign(f.span(kAtom_fx).data(), f.span(kAtom_fx).data() + rows);
    s.fy.assign(f.span(kAtom_fy).data(), f.span(kAtom_fy).data() + rows);
    s.fz.assign(f.span(kAtom_fz).data(), f.span(kAtom_fz).data() + rows);
    return s;
}

/// Build `make()` at each thread count and require every repeat to match
/// the first launch bit for bit.
template <class Make>
void expect_reproducible(Make make, cpu::Frame& f, cpu::Context& ctx) {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    for (const int threads : kThreadCounts) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        if (threads > 1)
            continue;
#endif
        cpu::md::ChunkForces arena;
        ctx.md.chunk_forces = &arena;
        auto pot = make();
        pot.build({ctx, f});
        const Snapshot first = evaluate(pot, ctx, arena, f);
        EXPECT_EQ(first.threaded, threads > 1) << threads << " threads";
        EXPECT_NE(first.energy, 0.0);
        for (int rep = 1; rep < kRepeats; ++rep)
            EXPECT_TRUE(evaluate(pot, ctx, arena, f) == first)
                << threads << " threads, repeat " << rep;
        pot.destroy({ctx, f});
        ctx.md.chunk_forces = nullptr;
    }
#ifdef _OPENMP
    omp_set_num_threads(saved);
    omp_set_dynamic(saved_dynamic);
#endif
}

} // namespace

TEST(ForceDeterminism, PairLoopStyleIsBitReproduciblePerThreadCount) {
    // 900 atoms, all pairs inside the cutoff: 404,550 pairs, enough for a
    // team of eight (PairRowPartition::kMinPairsPerThread).
    constexpr int kAtoms = 900;
    cpu::Frame f = chain(kAtoms, 1.1, 500.0);
    const auto pairs = all_pairs(kAtoms);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    expect_reproducible([] { return cpu::PairMorse(0.3, 1.2, 1.1, 60.0); }, f,
                        ctx);
}

TEST(ForceDeterminism, BondedStyleIsBitReproduciblePerThreadCount) {
    // 40,000 bonds: enough for a team of eight (ForcePass::kMinTermsPerThread).
    constexpr int kAtoms = 40001;
    cpu::Frame f = chain(kAtoms, 1.0, 500.0);
    cpu::Context ctx;
    expect_reproducible([] { return cpu::BondHarmonic({2.5}, {0.9}); }, f, ctx);
}
