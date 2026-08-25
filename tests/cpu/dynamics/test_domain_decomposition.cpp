/// @file tests/cpu/md/test_domain_decomposition.cpp
/// @brief RED tests for single-rank `DomainDecomposition` state (Phase 1).
///
/// Spec: .claude/specs/parallel-md-engine.md § A.1 + § B (Context fields).
/// The `Backend` must expose a non-null `Context::dd` in single-rank mode with
/// identity topology, zero ghosts, and an identity tag→local map.

#include <gtest/gtest.h>

#include <cstdint>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame make_small_frame(int n) {
    cpu::Frame frame(n);
    frame.box = ATV::cpu::Box::cubic(10.0);
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.1 * i;
        frame.span(ATV::kAtom_y)[i] = 0.2 * i;
        frame.span(ATV::kAtom_z)[i] = 0.3 * i;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
    }
    return frame;
}

cpu::Runtime make_backend_for(cpu::Frame& frame) {
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    return backend;
}

} // namespace

// ── single-rank defaults ─────────────────────────────────────────────

TEST(DomainDecomposition, SingleRankDefaults) {
    auto frame = make_small_frame(64);
    auto backend = make_backend_for(frame);

    cpu::Context& ctx = backend.context();
    ASSERT_NE(ctx.md.dd, nullptr) << "Context::dd must be non-null after "
                                     "Backend::init() even in single-rank mode";

    EXPECT_EQ(ctx.md.dd->dims[0], 1);
    EXPECT_EQ(ctx.md.dd->dims[1], 1);
    EXPECT_EQ(ctx.md.dd->dims[2], 1);
    EXPECT_EQ(ctx.md.dd->coords[0], 0);
    EXPECT_EQ(ctx.md.dd->coords[1], 0);
    EXPECT_EQ(ctx.md.dd->coords[2], 0);
    EXPECT_EQ(ctx.md.dd->periods[0], 1);
    EXPECT_EQ(ctx.md.dd->periods[1], 1);
    EXPECT_EQ(ctx.md.dd->periods[2], 1);
    EXPECT_EQ(ctx.md.dd->n_local, frame.nlocal());
    EXPECT_EQ(ctx.md.dd->n_ghost, 0);
    EXPECT_EQ(ctx.md.dd->capacity, frame.nlocal());
    EXPECT_EQ(ctx.md.dd->neighbor_generation, static_cast<std::uint64_t>(0));
}

// ── send lists empty ─────────────────────────────────────────────────
//
// The send lists moved off the decomposition onto the halo's published
// swap table (`ctx.md.swap_list`, owned by `HaloSend`). Single-rank keeps
// the same criterion — nothing is ever shipped — expressed at the new
// address: no halo component runs, so no table is published at all.

TEST(DomainDecomposition, SendListsEmpty) {
    auto frame = make_small_frame(32);
    auto backend = make_backend_for(frame);

    cpu::Context& ctx = backend.context();
    ASSERT_NE(ctx.md.dd, nullptr);
    EXPECT_EQ(ctx.md.swap_list, nullptr)
        << "single-rank mode must publish no swap table, so there are no "
           "send lists to ship from";
}

// ── identity tag→local map ───────────────────────────────────────────
//
// The tag→local map exists solely for bonded-tuple resolution, so it is
// maintained ONLY for frames carrying bonded topology. Two cases:
// bonded frame → identity map; bare frame → map stays empty (and the
// per-atom hash-node memory is never paid).

TEST(DomainDecomposition, TagToLocalIdentityMap) {
    const int n = 48;
    auto frame = make_small_frame(n);
    // Give the frame bonded topology so the map is maintained.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n);
    for (int i = 0; i + 1 < n; i += 2)
        frame.add_bond(i, i + 1, 0);
    auto backend = make_backend_for(frame);

    cpu::Context& ctx = backend.context();
    ASSERT_NE(ctx.md.dd, nullptr);

    EXPECT_EQ(static_cast<int>(ctx.md.dd->tag_to_local.size()), n);
    for (int i = 0; i < n; ++i) {
        auto it = ctx.md.dd->tag_to_local.find(frame.span(ATV::kAtom_tag)[i]);
        ASSERT_NE(it, ctx.md.dd->tag_to_local.end())
            << "tag " << frame.span(ATV::kAtom_tag)[i]
            << " missing from tag_to_local";
        EXPECT_EQ(it->second, i);
    }
}

TEST(DomainDecomposition, TagToLocalSkippedWithoutBondedTopology) {
    auto frame = make_small_frame(48);
    auto backend = make_backend_for(frame);

    cpu::Context& ctx = backend.context();
    ASSERT_NE(ctx.md.dd, nullptr);
    EXPECT_TRUE(ctx.md.dd->tag_to_local.empty())
        << "tag_to_local must not be populated for a frame without "
           "bonded topology";
    EXPECT_TRUE(frame.meta.get(cpu::kTagIndex).empty())
        << "the local tag index must be released without bonded topology";
}

// ── mpi_rank / mpi_size set in Context ───────────────────────────────

TEST(DomainDecomposition, MpiFieldsInContext) {
    auto frame = make_small_frame(8);
    auto backend = make_backend_for(frame);

    cpu::Context& ctx = backend.context();
    EXPECT_EQ(ctx.mpi_rank, 0);
    EXPECT_EQ(ctx.mpi_size, 1);
}
