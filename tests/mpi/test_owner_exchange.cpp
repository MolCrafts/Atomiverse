/// @file tests/mpi/test_owner_exchange.cpp
/// @brief Phase 0.2 (msu-pe-validation) / P3.1 -- `Comm::exchange`.
///
/// Verifies that an owner whose post-integration position falls outside its
/// rank's slab is transferred to the rank whose slab actually contains the
/// new position. `ctx.md.comm->exchange()` must:
///   - drop the migrating atom from this rank's owner range,
///   - append it (with all per-atom fields preserved) on the destination rank,
///   - bump `dd->neighbor_generation`,
///   - rebuild `dd->tag_to_local` so downstream consumers can resolve tags
///     to the new local indices.

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

const ColKey<StorageReal> kMigrationAux{"atoms", "migration_aux",
                                        kDynamicFieldId, ColDType::Storage};

// 8 atoms uniformly spaced along x for a 1D-like setup with 2 ranks
// along x. The y, z positions are arbitrary but all in [0, box_len).
cpu::Frame make_chain_frame(double box_len, int n_atoms) {
    cpu::Frame frame(n_atoms);
    const double dx = box_len / static_cast<double>(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        frame.span(ATV::kAtom_x)[i] = (i + 0.5) * dx;
        frame.span(ATV::kAtom_y)[i] = box_len * 0.5;
        frame.span(ATV::kAtom_z)[i] = box_len * 0.5;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
        frame.span(ATV::kAtom_type)[i] = 0;
        frame.span(ATV::kAtom_vx)[i] = 0.0;
        frame.span(ATV::kAtom_vy)[i] = 0.0;
        frame.span(ATV::kAtom_vz)[i] = 0.0;
    }
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// ── 2-rank: a single atom drifts past +x boundary, must end up on neighbor ──

TEST(Exchange, AtomDriftsAcrossPosXBoundary) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank-specific test; world_size=" << world_size;
    }

    auto frame = make_chain_frame(20.0, 8); // 8 atoms in a 20 Å box
    frame.block("atoms").insert(
        "migration_aux",
        cpu::ColumnSpec{ColDType::Storage, 2, ColGroup::Core, 0,
                        kColMigrate | kColExchange, kDynamicFieldId});
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kMigrationAux)(i, 0) = StorageReal(100 + i);
        frame.span(kMigrationAux)(i, 1) = StorageReal(-200 - i);
    }
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Single-direction migration: only rank 0 displaces one of its owners
    // past its +x slab boundary so the net flow is rank 0 -> rank 1. Rank 1
    // is the pure receiver. (If both ranks pushed, the net n_local change
    // would be zero on each rank and the test would be trivial.)
    TagInt moved_tag = TagInt(-1);
    StorageReal moved_aux[2]{};
    if (my_rank == 0) {
        // Take the rightmost owner and shove it 1 Å past subhi[0].
        int rightmost = 0;
        for (int i = 1; i < dd->n_local; ++i) {
            if (frame.span(ATV::kAtom_x)[i]
                > frame.span(ATV::kAtom_x)[rightmost])
                rightmost = i;
        }
#if defined(ATV_ID_BIGBIG)
        frame.span(ATV::kAtom_tag)[rightmost] = TagInt{9007199254740993LL};
#else
        frame.span(ATV::kAtom_tag)[rightmost] = TagInt{700000003};
#endif
        moved_tag = frame.span(ATV::kAtom_tag)[rightmost];
        moved_aux[0] = frame.span(kMigrationAux)(rightmost, 0);
        moved_aux[1] = frame.span(kMigrationAux)(rightmost, 1);
        const double Lx = frame.box.cell()(0, 0);
        const double new_x = dd->subhi[0] + 1.0;
        frame.span(ATV::kAtom_x)[rightmost] =
            new_x - Lx * std::floor(new_x / Lx);
    }
    const int n_local_before = dd->n_local;
    const std::uint64_t gen_before = dd->neighbor_generation;

    // Broadcast the moved tag from rank 0 so all ranks can probe their
    // tag_to_local for it post-migration.
    const MPI_Datatype tag_mpi =
        std::is_same_v<TagInt, std::int32_t> ? MPI_INT32_T : MPI_INT64_T;
    const MPI_Datatype storage_mpi =
        std::is_same_v<StorageReal, float> ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Bcast(&moved_tag, 1, tag_mpi, 0, MPI_COMM_WORLD);
    MPI_Bcast(moved_aux, 2, storage_mpi, 0, MPI_COMM_WORLD);

    ASSERT_NE(backend.context().md.comm, nullptr);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    backend.context().md.comm->exchange(lctx);

    const int n_local_after = dd->n_local;
    if (my_rank == 0) {
        EXPECT_EQ(n_local_after, n_local_before - 1)
            << "sender rank 0 did not lose its migrating atom";
    } else {
        EXPECT_EQ(n_local_after, n_local_before + 1)
            << "receiver rank 1 did not gain the migrated atom";
    }

    EXPECT_GT(dd->neighbor_generation, gen_before)
        << "neighbor_generation must bump on a successful migration";

    // Globally the atom count must be conserved.
    int local_count = dd->n_local;
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_EQ(global_count, 8) << "atom count must be globally conserved";

    // The migrated tag must now be OWNED by exactly one rank. Scan the
    // owned rows directly: `tag_to_local` is maintained only for frames
    // with bonded topology (it exists solely for bonded-tuple
    // resolution), and this frame has none — assert that contract too.
    EXPECT_TRUE(dd->tag_to_local.empty())
        << "tag_to_local must stay empty for a frame without bonded topology";
    int i_have_it = 0;
    for (int i = 0; i < dd->n_local; ++i) {
        if (frame.span(ATV::kAtom_tag)[i] == moved_tag) {
            i_have_it = 1;
            break;
        }
    }
    int total_owners_of_tag = 0;
    MPI_Allreduce(&i_have_it, &total_owners_of_tag, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_EQ(total_owners_of_tag, 1)
        << "tag " << moved_tag
        << " must have exactly one owner across all ranks";

    // The tag must now belong to the receiver (rank 1).
    if (my_rank == 1) {
        EXPECT_EQ(i_have_it, 1) << "rank 1 did not receive the migrating tag";
        for (int i = 0; i < dd->n_local; ++i) {
            if (frame.span(ATV::kAtom_tag)[i] != moved_tag)
                continue;
            EXPECT_EQ(frame.span(kMigrationAux)(i, 0), moved_aux[0]);
            EXPECT_EQ(frame.span(kMigrationAux)(i, 1), moved_aux[1]);
        }
    }

    backend.shutdown();
}

// ── single-rank no-op: exchange must leave the frame untouched ──

TEST(Exchange, SingleRankIsNoOp) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1) {
        GTEST_SKIP() << "single-rank guard";
    }

    auto frame = make_chain_frame(20.0, 8);
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;
    const int n_before = dd->n_local;
    const std::uint64_t gen_before = dd->neighbor_generation;

    // Move atom 0 past where the slab boundary would be (box edge in single-rank).
    frame.span(ATV::kAtom_x)[0] = 19.5;

    ASSERT_NE(backend.context().md.comm, nullptr);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    backend.context().md.comm->exchange(lctx);

    EXPECT_EQ(dd->n_local, n_before)
        << "single-rank migration must not change n_local";
    EXPECT_EQ(dd->neighbor_generation, gen_before)
        << "single-rank migration must not bump neighbor_generation";
    EXPECT_DOUBLE_EQ(frame.span(ATV::kAtom_x)[0], 19.5)
        << "single-rank migration must not move atoms";

    backend.shutdown();
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
