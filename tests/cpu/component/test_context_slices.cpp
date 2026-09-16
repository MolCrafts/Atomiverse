/// @file tests/cpu/component/test_context_slices.cpp
/// @brief RED gate for arch-refactor-05-context: `cpu::Context` regrouped
///        into flat core fields (n_threads / mpi_rank / mpi_size / timer /
///        outer_status) plus four nested pure-aggregate service slices —
///        `DynamicsServices md`, `SamplingServices mc`,
///        `OptimizeServices opt`, `AnalysisServices analysis`.
///
/// Layout-only: no physics, no numerics. Pins
///   1. the exact per-slice field mapping (compile-level static_asserts),
///   2. default-constructed slices are all-null / all-false
///      ("field unset = service unavailable" semantics carried over verbatim),
///   3. core fields stay flat with today's defaults,
///   4. the header guards: DynamicsServices / SamplingServices /
///      AnalysisServices trivially copyable (OptimizeServices exempt —
///      pre-existing std::function), Context non-polymorphic,
///   5. slices are plain writable aggregates (assign → read back).
///
/// RED: fails to COMPILE against the current flat `cpu::Context`
/// (no `md`/`mc`/`opt`/`analysis` members, no slice types). GREEN after the
/// include/atomiverse/cpu/component/context.h regrouping lands.

#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <type_traits>

#include "atomiverse/core/iteration_status.h"
#include "atomiverse/cpu/asynchronous/analysis_executor.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/montecarlo/transaction.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

using namespace ATV;

// ── Header guards: trivially-copyable slices, non-polymorphic Context ──────
// (Category: domain validation of the layout contract — zero-cost grouping.)

TEST(ContextSlices, SliceStructsAreTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<cpu::DynamicsServices>,
                  "DynamicsServices must stay a trivially copyable aggregate");
    static_assert(std::is_trivially_copyable_v<cpu::SamplingServices>,
                  "SamplingServices must stay a trivially copyable aggregate");
    static_assert(std::is_trivially_copyable_v<cpu::AnalysisServices>,
                  "AnalysisServices must stay a trivially copyable aggregate");
    // OptimizeServices is exempt: it carries the pre-existing std::function
    // reeval_forces (present in Context before this refactor).
    SUCCEED();
}

TEST(ContextSlices, ContextIsNonPolymorphic) {
    static_assert(!std::is_polymorphic_v<cpu::Context>,
                  "cpu::Context must have no vtable — constant-offset field "
                  "access is a chain-level hot-path invariant");
    SUCCEED();
}

// ── Exact field mapping (compile-level) ────────────────────────────────────
// (Category: basics — the slice layout the spec dictates, field by field.)

TEST(ContextSlices, DynamicsSliceFieldMapping) {
    static_assert(
        std::is_same_v<decltype(cpu::Context::md), cpu::DynamicsServices>);
    static_assert(std::is_same_v<decltype(cpu::DynamicsServices::dd),
                                 cpu::md::mpi::DomainDecomposition*>);
    static_assert(std::is_same_v<decltype(cpu::DynamicsServices::pair_list),
                                 const cpu::md::PairList*>);
    static_assert(
        std::is_same_v<decltype(cpu::DynamicsServices::is_last_step), bool>);
    SUCCEED();
}

TEST(ContextSlices, SamplingSliceFieldMapping) {
    static_assert(
        std::is_same_v<decltype(cpu::Context::mc), cpu::SamplingServices>);
    static_assert(std::is_same_v<decltype(cpu::SamplingServices::proposal),
                                 cpu::mc::Transaction*>);
    static_assert(
        std::is_same_v<decltype(cpu::SamplingServices::rng), std::mt19937_64*>);
    SUCCEED();
}

TEST(ContextSlices, OptimizeSliceFieldMapping) {
    static_assert(
        std::is_same_v<decltype(cpu::Context::opt), cpu::OptimizeServices>);
    static_assert(std::is_same_v<decltype(cpu::OptimizeServices::reeval_forces),
                                 std::function<void(bool)>>);
    static_assert(
        std::is_same_v<decltype(cpu::OptimizeServices::line_search_trial),
                       bool>);
    SUCCEED();
}

TEST(ContextSlices, AnalysisSliceFieldMapping) {
    static_assert(std::is_same_v<decltype(cpu::Context::analysis),
                                 cpu::AnalysisServices>);
    static_assert(std::is_same_v<decltype(cpu::AnalysisServices::executor),
                                 cpu::AnalysisExecutor*>);
    SUCCEED();
}

// ── Core fields stay flat ──────────────────────────────────────────────────
// (Category: basics — direct reachability at compile level + defaults.)

TEST(ContextSlices, CoreFieldsStayFlatWithDefaults) {
    static_assert(std::is_same_v<decltype(cpu::Context::n_threads), int>);
    static_assert(std::is_same_v<decltype(cpu::Context::mpi_rank), int>);
    static_assert(std::is_same_v<decltype(cpu::Context::mpi_size), int>);
    static_assert(
        std::is_same_v<decltype(cpu::Context::timer), ATV::TimerRegistry*>);
    // The Context does NOT own an IterationStatus any more — it BORROWS the active
    // loop's. That indirection is the whole fix for a bug that shipped twice: with one
    // status on the Context, a nested loop's convergence stopped its parent (probe 2:
    // LBFGS wrote `converged` unconditionally; probe 3: the first converged NEB image
    // would have stopped the chain). A pointer that points at the loop cannot reach the
    // loop above it.
    static_assert(
        std::is_same_v<decltype(cpu::Context::loop), cpu::LoopServices>);
    static_assert(std::is_same_v<decltype(cpu::LoopServices::status),
                                 ATV::IterationStatus*>);
    static_assert(std::is_same_v<decltype(cpu::LoopServices::metrics),
                                 ATV::ConvergenceMetrics*>);

    cpu::Context ctx;
    EXPECT_EQ(ctx.n_threads, 0);
    EXPECT_EQ(ctx.mpi_rank, 0);
    EXPECT_EQ(ctx.mpi_size, 1);
    EXPECT_EQ(ctx.timer, nullptr);
    // Null until a Driver runs: a bare Context is attached to no loop at all.
    EXPECT_EQ(ctx.loop.status, nullptr);
    EXPECT_EQ(ctx.loop.metrics, nullptr);
}

// ── Default-constructed slices: all-null / all-false ───────────────────────
// (Category: edge case — "unset slice = service unavailable"; components
//  null-check before use, verbatim carry-over of today's semantics.)

TEST(ContextSlices, DefaultDynamicsSliceIsNull) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.md.dd, nullptr);
    EXPECT_EQ(ctx.md.pair_list, nullptr);
    EXPECT_FALSE(ctx.md.is_last_step);
}

TEST(ContextSlices, DefaultSamplingSliceIsNull) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.mc.proposal, nullptr);
    EXPECT_EQ(ctx.mc.rng, nullptr);
}

TEST(ContextSlices, DefaultOptimizeSliceIsNull) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.opt.reeval_forces, nullptr); // std::function null-comparison
    EXPECT_FALSE(static_cast<bool>(ctx.opt.reeval_forces));
    EXPECT_FALSE(ctx.opt.line_search_trial);
}

TEST(ContextSlices, DefaultAnalysisSliceIsNull) {
    cpu::Context ctx;
    EXPECT_EQ(ctx.analysis.executor, nullptr);
}

// ── Slices are writable aggregates: assign → read back ────────────────────
// (Category: basics/integration — publisher-side writes land where the
//  consumer-side reads look, through real service objects where cheap.)

TEST(ContextSlices, DynamicsSliceIsWritable) {
    cpu::Context ctx;

    ctx.md.is_last_step = true;
    EXPECT_TRUE(ctx.md.is_last_step);

    cpu::md::PairList pl;
    ctx.md.pair_list = &pl;
    ASSERT_EQ(ctx.md.pair_list, &pl);
    EXPECT_EQ(ctx.md.pair_list->inum, 0); // const read-through path
}

TEST(ContextSlices, SamplingSliceIsWritable) {
    cpu::Context ctx;

    cpu::mc::Transaction txn;
    ctx.mc.proposal = &txn;
    EXPECT_EQ(ctx.mc.proposal, &txn);

    // Seeded single-RNG-stream reproducibility through the slice pointer.
    std::mt19937_64 rng(42u);
    std::mt19937_64 reference(42u);
    ctx.mc.rng = &rng;
    ASSERT_EQ(ctx.mc.rng, &rng);
    EXPECT_EQ((*ctx.mc.rng)(), reference());
}

TEST(ContextSlices, OptimizeSliceIsWritable) {
    cpu::Context ctx;

    int calls = 0;
    bool last_need_grad = false;
    ctx.opt.reeval_forces = [&](bool need_grad) {
        ++calls;
        last_need_grad = need_grad;
    };
    ASSERT_NE(ctx.opt.reeval_forces, nullptr);
    ctx.opt.reeval_forces(true);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(last_need_grad);

    ctx.opt.line_search_trial = true;
    EXPECT_TRUE(ctx.opt.line_search_trial);
}

TEST(ContextSlices, AnalysisSliceIsWritable) {
    cpu::Context ctx;

    cpu::AnalysisExecutor exec; // default-constructed, no init() — cheap
    ctx.analysis.executor = &exec;
    EXPECT_EQ(ctx.analysis.executor, &exec);
}

// ── Pure-aggregate copy semantics ──────────────────────────────────────────
// (Category: basics — copying a Context copies slice values; no identity,
//  no back-references, exactly what "zero-cost grouping" promises.)

TEST(ContextSlices, ContextCopyPreservesSliceValues) {
    cpu::Context ctx;
    cpu::md::PairList pl;
    cpu::mc::Transaction txn;
    std::mt19937_64 rng(7u);

    ctx.md.pair_list = &pl;
    ctx.md.is_last_step = true;
    ctx.mc.proposal = &txn;
    ctx.mc.rng = &rng;
    ctx.opt.line_search_trial = true;

    cpu::Context copy = ctx;
    EXPECT_EQ(copy.md.pair_list, &pl);
    EXPECT_TRUE(copy.md.is_last_step);
    EXPECT_EQ(copy.mc.proposal, &txn);
    EXPECT_EQ(copy.mc.rng, &rng);
    EXPECT_TRUE(copy.opt.line_search_trial);
}
